from pathlib import Path
import os
import shutil
import torch
import torch.nn as nn
import hydra
from omegaconf import OmegaConf, DictConfig
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import torchinfo
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# srcフォルダから必要なモジュールをインポート
from src.dataset import AnnotatedDatasetFolder, collate_fn_skip_none, pil_loader
from src.model import ResNet, SimpleCNN
from src.trainer import Trainer, LossEvaluator
from src.train_id import print_config, generate_train_id
from src.extension import ModelSaver, HistorySaver, HistoryLogger, IntervalTrigger, LearningCurvePlotter, MinValueTrigger
from src.util import set_random_seed

def denormalize(tensor, mean, std):
    """正規化されたテンソルを元の範囲 [0, 1] に戻す"""
    if tensor.ndim == 3:
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
    return tensor

def adjust_exposure(image_tensor, ev_value):
    """
    予測されたEV値に基づいて画像の明るさを補正する（簡易版）
    image_tensor: [C, H, W] 形式で、範囲が [0, 1] のテンソル
     """
    # 明るさの補正係数を計算 (EV値が+1なら明るさ2倍、-1なら1/2倍)
    correction_factor = 2.0 ** ev_value
    # 画像の各ピクセル値に補正係数を乗算
    corrected_image = image_tensor * correction_factor
    # 値が [0, 1] の範囲に収まるようにクリッピング
    return torch.clamp(corrected_image, 0, 1)

# --- 評価のコアロジック ---
def run_evaluation(train_cfg, test_cfg, train_id, seed, device):
    """単一の学習済みモデルに対する評価処理を行う関数"""
    net = hydra.utils.instantiate(train_cfg.model).to(device)
    model_path = Path("./outputs/train/history") / train_id / "best_model.pth"
    net.load_state_dict(torch.load(model_path, map_location=device))

    # config.yamlのtestセクションを元にデータセットを準備
    annotation_file_path = test_cfg.dataset.test.annotation_file
    if str(annotation_file_path).lower() == "null" or annotation_file_path is None:
        annotation_file_path = None
    
    # データ拡張(transform)は、config.yamlの定義に基づき、ここで直接インスタンス化
    test_transforms = hydra.utils.instantiate(test_cfg.dataset.test.transform)

    dataset = AnnotatedDatasetFolder(
        root=test_cfg.dataset.test.root,
        annotation_file=annotation_file_path,
        loader=pil_loader,
        transform=test_transforms
    )
    # 評価用データローダーを作成
    test_loader = torch.utils.data.DataLoader(
        dataset, 
        shuffle=False, 
        batch_size=test_cfg.dataloader.batch_size, # 評価時のバッチサイズを使用
        num_workers=test_cfg.dataloader.num_workers, 
        collate_fn=collate_fn_skip_none
    )

    net.eval()
    criterion = torch.nn.MSELoss()
    evaluator = LossEvaluator(criterion, criterion_name="MSE")
    evaluator.initialize()

    predictions = []
    best_image_info = {
        'min_error': float('inf'),
        'original_image': None,
        'pred_ev': None,
        'true_ev': None, 
        'filename': None
    }

    has_labels = annotation_file_path is not None

    with torch.no_grad(): # 勾配計算を無効にして、メモリ消費を抑え、計算を高速化
        for batch in test_loader:
            if not batch or batch[0] is None: continue
            inputs, targets, filenames = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            # 正解ラベルがある場合のみ、誤差の計算と比較を行う
            if has_labels:
                targets = targets.to(device)
                evaluator.eval_batch(outputs, targets)
                abs_errors = torch.abs(outputs - targets).squeeze()
                if abs_errors.dim() == 0: abs_errors = [abs_errors]
                for i, error in enumerate(abs_errors):
                    if error.item() < best_image_info['min_error']:
                        best_image_info.update({
                            'min_error': error.item(),
                            'original_image': inputs[i].cpu(),
                            'pred_ev': outputs[i].item(),
                            'true_ev': targets[i].item(),
                            'filename': filenames[i]
                        })

            target_items = targets.cpu().tolist() if has_labels else [None] * len(filenames)
            for filename, target, output in zip(filenames, target_items, outputs.cpu()):
                predictions.append([filename, target, output.item()])

    if has_labels:
        result = evaluator.finalize()
        mse_value = result['loss/MSE']
        rmse_value = torch.sqrt(torch.tensor(mse_value)).item()
        result['loss/RMSE'] = rmse_value 

        print(f"\n[INFO] Train ID {train_id} の評価結果 (モデル: {train_cfg.model.name})")
        for key, value in result.items():
            print(f"{key}: {value:.4f}")

        log_path = Path("./outputs/test_results") / f"{train_id}_result.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w",encoding="utf-8") as f:
            f.write(f"Train ID {train_id} の評価結果\n")
            f.write(f"Model: {train_cfg.model.name}\n")
            for key, value in result.items():
                f.write(f"{key}:{value:.4f}\n")

    csv_path = Path(f"./outputs/predictions/{train_id}_predictions.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_ev", "pred_ev"])
        writer.writerows(predictions)
    print(f"予測結果を {csv_path} に保存しました")

    if has_labels and 'original_image' in best_image_info:
        mean = test_cfg.dataset.test.transform.transforms[-1].mean
        std = test_cfg.dataset.test.transform.transforms[-1].std
        original_img_denorm = denormalize(best_image_info['original_image'].clone(), mean, std)
        corrected_img = adjust_exposure(original_img_denorm, best_image_info['pred_ev'])

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_img_denorm.permute(1, 2, 0))
        axes[0].set_title(f"補正前 (Original)\nTrue EV: {best_image_info['true_ev']:.2f}")
        axes[0].axis('off')
        axes[1].imshow(corrected_img.permute(1, 2, 0))
        axes[1].set_title(f"補正後 (Corrected)\nPredicted EV: {best_image_info['pred_ev']:.2f}")
        axes[1].axis('off')
        fig.suptitle(f"最高精度の画像: {best_image_info['filename']}\n(Error: {best_image_info['min_error']:.4f})")
        plt.show()

        save_dir = Path(f"./outputs/best_predictions/{train_id}")
        save_dir.mkdir(parents=True, exist_ok=True)
        vutils.save_image(original_img_denorm, save_dir / f"{best_image_info['filename']}_original.png")
        vutils.save_image(corrected_img, save_dir / f"{best_image_info['filename']}_corrected.png")
        print(f"補正前後の画像を{save_dir} に保存しました")
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # コマンドライン引数をHydraのconfigオブジェクトから取得
    args = cfg.args 
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    p = Path(args.history_dir)
    # 学習履歴フォルダをループ
    for q in sorted(p.glob('**/config.yaml')):
        train_cfg = OmegaConf.load(str(q))
        train_id = q.parent.name
        
        if not (q.parent / "best_model.pth").exists(): continue
        
        result_file = Path("./outputs/test_results") / f"{train_id}_result.txt"
        if result_file.exists() and args.skip_tested:
            print(f"Train ID {train_id} ：テスト済み")
            continue
        
        print("-" * 50)
        print(f"Train ID: {train_id} のモデルをテスト")

        try:
            # 評価のコア処理を呼び出す
            run_evaluation(train_cfg, cfg, train_id, args.seed, device)
        except Exception as e:
            print(f"例外が発生したため、Train ID {train_id} をスキップ: {e}")

#if __name__ == "__main__":
    # argparse を使って、Hydraが直接管理しない引数を定義
    #parser = argparse.ArgumentParser(description="学習済みモデルをテスト")
    #parser.add_argument('--history_dir', type=str, default="./outputs/train/history")
    #parser.add_argument('--seed', type=int, default=42)
    #parser.add_argument('--skip_tested', action="store_true")
    
    # Hydraが解析しないコマンドライン引数のみをパース
    #args, _ = parser.parse_known_args()
    
    # パースした引数をHydraの設定にマージするための登録
    #OmegaConf.register_new_resolver("args", lambda: args)
    
    #main()



    