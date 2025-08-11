from pathlib import Path
import os
import shutil
import argparse
import torch
from omegaconf import OmegaConf, DictConfig
import csv
import hydra
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# srcフォルダから必要なモジュールをインポート
from src.dataset import AnnotatedDatasetFolder, collate_fn_skip_none, pil_loader
from src.model import ResNet, SimpleCNN
from src.trainer import LossEvaluator
from src.train_id import print_config
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

# main関数の引数をcfgのみにする 
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # Hydraのconfigオブジェクトから引数を取得
    args = cfg.args
    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 履歴フォルダのパスを修正
    p = Path(args.history_dir)
    for q in sorted(p.glob('**/config.yaml')):
        train_cfg = OmegaConf.load(str(q))
        train_id = q.parent.name
        
        if not (q.parent / "best_model.pth").exists(): continue
        
        # 結果ファイルのパスを修正
        result_file = Path("../outputs/test_results") / f"{train_id}_result.txt"
        if result_file.exists() and args.skip_tested:
            print(f"Train ID {train_id} はすでにテスト済みです")
            continue
        
        print("-" * 50)
        print(f"Train ID: {train_id} のモデルをテストします")

        try:
            # 評価処理に必要な情報を引数として渡す
            run_evaluation(train_cfg, train_id, args.seed, device)
        except Exception as e:
            print(f"例外が発生したため、Train ID {train_id} をスキップしました: {e}")

# 評価のコア処理を別の関数に分離
def run_evaluation(cfg, train_id, seed, device):
    # Hydra経由でモデルをインスタンス化
    net = hydra.utils.instantiate(cfg.model).to(device)
    # outputsへのパスを修正
    model_path = Path("../outputs/train/history") / train_id / "best_model.pth"
    net.load_state_dict(torch.load(model_path, map_location=device))
    
    if cfg.dataset.test.annotation_file == "null":
        cfg.dataset.test.annotation_file = None
        
    dataset = hydra.utils.instantiate(cfg.dataset.test)
    test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=64, num_workers=2, collate_fn=collate_fn_skip_none)

    net.eval()
    criterion = torch.nn.MSELoss()
    evaluator = LossEvaluator(criterion, criterion_name="MSE") # 修正点: MSEで評価
    evaluator.initialize()

    predictions = []
    best_image_info = {
        'min_error': float('inf'), 'original_image': None, 'pred_ev': None,
        'true_ev': None, 'filename': None
    }
    # annotation_fileが指定されているかどうかのフラグ
    has_labels = cfg.dataset.test.annotation_file is not None

    with torch.no_grad():
        for batch in test_loader:
            if not batch or batch[0] is None: continue
            inputs, targets, filenames = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            
            # 正解ラベルがある場合のみ評価と最良画像の追跡を行う
            if has_labels:
                evaluator.eval_batch(outputs, targets)
                # 最も精度の良い画像を特定
                abs_errors = torch.abs(outputs - targets).squeeze()
                if abs_errors.dim() == 0: abs_errors = [abs_errors]
                for i, error in enumerate(abs_errors):
                    if error.item() < best_image_info['min_error']:
                        best_image_info.update({
                            'min_error': error.item(), 
                            # 画像データをCPUに移して保存
                            'original_image': inputs[i].cpu(),
                            'pred_ev': outputs[i].item(), 
                            'true_ev': targets[i].item(),
                            'filename': filenames[i]
                        })
            # 予測結果を保存
            for filename, target, output in zip(filenames, targets, outputs):
                predictions.append([filename, target.item(), output.item()])

    # 正解ラベルがある場合、最終的な精度を表示・保存        
    if has_labels:
        result = evaluator.finalize()
        mse_value = result['loss/MSE']
        rmse_value = torch.sqrt(torch.tensor(mse_value)).item()
        result['loss/RMSE'] = rmse_value
        
        print(f"\n[INFO] Train ID {train_id} の評価結果 (モデル: {cfg.model.name})")
        for key, value in result.items():
            print(f"{key}: {value:.4f}")

        log_path = Path("../outputs/test_results") / f"{train_id}_result.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            f.write(f"Train ID {train_id} の評価結果\n")
            f.write(f"Model: {cfg.model.name}\n")
            for key, value in result.items():
                f.write(f"{key}:{value:.4f}\n")

    # 全ての予測結果をCSVに保存
    csv_path = Path(f"../outputs/predictions/{train_id}_predictions.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_ev", "pred_ev"])
        writer.writerows(predictions)
    print(f"予測結果を {csv_path} に保存しました")

    # 最も精度が良かった画像の表示と保存
    if best_image_info['original_image'] is not None:
        # 正規化を解除して元の画像に戻す
        mean = cfg.dataset.test.transform.transforms[-1].mean
        std = cfg.dataset.test.transform.transforms[-1].std
        original_img_denorm = denormalize(best_image_info['original_image'].clone(), mean, std)
        # 正規化を解除して元の画像に戻す
        corrected_img = adjust_exposure(original_img_denorm, best_image_info['pred_ev'])

        # 表示
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_img_denorm.permute(1, 2, 0))
        axes[0].set_title(f"補正前 (Original)\nTrue EV: {best_image_info['true_ev']:.2f}")
        axes[0].axis('off')

        axes[1].imshow(corrected_img.permute(1, 2, 0))
        axes[1].set_title(f"補正後 (Corrected)\nPredicted EV: {best_image_info['pred_ev']:.2f}")
        axes[1].axis('off')

        fig.suptitle(f"最も精度が良かった画像: {best_image_info['filename']}\n(Error: {best_image_info['min_error']:.4f})")
        plt.show()

        # 保存
        save_dir = Path(f"../outputs/best_predictions/{train_id}")
        save_dir.mkdir(parents=True, exist_ok=True)
        vutils.save_image(original_img_denorm, save_dir / f"{best_image_info['filename']}_original.png")
        vutils.save_image(corrected_img, save_dir / f"{best_image_info['filename']}_corrected.png")
        print(f"補正前後の画像を {save_dir} に保存しました")

if __name__ == "__main__":
    # argparseとHydraを連携
    parser = argparse.ArgumentParser(description="学習済みモデルをテスト")
    parser.add_argument('--history_dir', type=str, default="../outputs/train/history")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip_tested', action="store_true")
    
    # Hydraが解析しないコマンドライン引数をパース
    args, _ = parser.parse_known_args()
    
    # パースした引数をHydraの設定にマージするための登録
    OmegaConf.register_new_resolver("args", lambda: args)
    
    main()