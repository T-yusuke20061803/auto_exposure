# 評価用コード
from pathlib import Path
import torch, csv, datetime, shutil
import torchvision.utils as vutils
from torchvision.transforms import v2
import hydra
from omegaconf import DictConfig

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import time

from src.dataset import AnnotatedDatasetFolder, pil_loader, collate_fn_skip_none
from src.model import SimpleCNN, ResNet, RegressionEfficientNet, RegressionMobileNet
from src.trainer import LossEvaluator
from src.util import set_random_seed

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def adjust_exposure(image_tensor, ev_value):
    # sRGB (非線形) -> Linear (線形)
    # 一般的なガンマ値 2.2 を使用
    linear_image = torch.pow(image_tensor, 2.2)
    correction_factor = 2.0 ** ev_value
    corrected_linear_image = linear_image * correction_factor
    # Linear (線形) -> sRGB (非線形) に戻す
    # 逆ガンマ (1 / 2.2) を適用
    corrected_srgb_image = torch.pow(corrected_linear_image, 1.0/2.2)
    # 最終結果を [0, 1] にクリップして返す
    return torch.clamp(corrected_srgb_image, 0, 1)

def plot_ev_predictions(csv_file, output_dir):
    try:
        df = pd.read_csv(csv_file)
        #散布図
        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=df, x="true_ev", y="pred_ev", s=50, alpha=0.7)

        # 理想直線（y=x）を描画
        min_val = min(df.true_ev.min(), df.pred_ev.min())
        max_val = max(df.true_ev.max(), df.pred_ev.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal (y=x)")

        plt.xlabel("True EV (正解値)")
        plt.ylabel("Predicted EV (予測値)")
        plt.title("散布図：露出値（正解値+予測値）")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "露出値_scatter.png")
        plt.close()
        

        #誤差分布のヒストグラム
        df["diff"] = df["pred_ev"] - df["true_ev"]
        plt.figure(figsize=(6,4))
        sns.histplot(df["diff"], kde=True, bins=30)
        plt.xlabel("予測誤差（予測値ー正解値）")
        plt.title("予測誤差の分布")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "予測誤差_histogram.png")
        plt.close()

        print(f"可視化グラフ保存完了: {output_dir}")
    except Exception as e:
        print(f"警告: グラフの描画に失敗しました。{e}")
 
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    set_random_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #モデル別の最新の学習結果を探す
    model_name = cfg.model.name
    history_dir = Path(f"outputs/train_reg/history/{model_name}")

# Train ID を config.yaml から取得 
 # Train ID 自動検出 or 指定
    if "train_id" in cfg and cfg.train_id != "your_train_id_here":
        train_id = cfg.train_id
    else:
        if not history_dir.exists():
            raise FileNotFoundError(f"モデル '{model_name}' の学習履歴ディレクトリが見つかりません。")
        # best_model.pthが存在するディレクトリの中から、更新日時が最新のものを探す
        run_dirs = [d for d in history_dir.iterdir() if d.is_dir() and (d / "best_model.pth").exists()]
        if not run_dirs:
            raise FileNotFoundError(f"モデル '{model_name}' の学習済みモデル（best_model.pth）が見つかりません。")


        latest_model_dir = max((p for p in history_dir.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
        train_id = latest_model_dir.name

        print(f"[INFO] train_id が指定されていないため最新を使用します: {train_id}")

    #モデルとパス
    #model_dir = next(Path("outputs/train_reg/history").rglob(f"{train_id}"))
    model_path = history_dir / train_id / "best_model.pth"
    config_path = history_dir / train_id / "config.yaml"


    if not model_path.exists():
        raise FileNotFoundError(f"最良モデルが見つかりません: {model_path}")
    if not config_path.exists():
        print("[WARN] config.yaml が見つかりません。train_id のみで識別します。")
    
    print(f"[INFO] 使用モデル: {model_path}")

    #モデル構築
    if cfg.model.name.lower() == "simplecnn":
        net = SimpleCNN(**cfg.model.params).to(device)
    elif cfg.model.name.lower() == "resnet":
        net = ResNet(**cfg.model.params).to(device)
    elif cfg.model.name.lower() == "efficientnet":
        net = RegressionEfficientNet(**cfg.model.params).to(device)
    elif cfg.model.name.lower() == "mobilenet":
        net = RegressionMobileNet(**cfg.model.params).to(device)
    else:
        raise ValueError(f"未対応のモデルです: {cfg.model.name}")
    
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

#評価時においてはデータ拡張を行わない
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(cfg.dataset.test.transform.resize),
        v2.CenterCrop(cfg.dataset.test.transform.center_crop),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=cfg.dataset.test.transform.normalize.mean,
            std=cfg.dataset.test.transform.normalize.std
        ),
    ])
    
    print(f"[INFO] テストCSV: {cfg.dataset.test.csv_file}")
    print(f"[INFO] テスト画像ルート: {cfg.dataset.test.root}")

# test.csv を読み込む　質問2
    dataset = AnnotatedDatasetFolder(
        root=cfg.dataset.test.root,
        csv_file=cfg.dataset.test.csv_file,
        loader=pil_loader,
        transform=transform
    )
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        collate_fn=collate_fn_skip_none
    )
    #評価処理
    criterion = torch.nn.MSELoss()
    evaluator = LossEvaluator(criterion, "MSE")
    evaluator.initialize()

    predictions = []
    best_image_info = {"min_error": float("inf")}

    #推論速度計測用 start
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets, filenames) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            evaluator.eval_batch(outputs, targets)
            errors = torch.abs(outputs - targets).squeeze()
            if errors.dim() == 0: # バッチサイズが1の場合
                errors = errors.unsqueeze(0)

            for i, (filename, target, output) in enumerate(zip(filenames, targets, outputs)):
                predictions.append([filename, target.item(), output.item()])
                if errors[i] < best_image_info["min_error"]:
                    best_image_info.update({
                        "min_error": errors[i].item(),
                        "original": inputs[i].cpu(),
                        "pred_ev": output.item(),
                        "true_ev": target.item(),
                        "filename": filename
                    })

    end_time = time.time()
    total_inference_time = end_time - start_time
    avg_inference_time_ms = (total_inference_time / len(dataset)) * 1000 if len(dataset) > 0 else 0


    #  評価結果計算
    result = evaluator.finalize()
    #MSEのキーを柔軟に取得
    # MSE 取得（キーが "loss")
    mse_value = result.get("loss/MSE",result.get("loss", None))
    if mse_value is None:
        raise KeyError(f"MSEが見つかりません: {result}")

    rmse_value = float(torch.sqrt(torch.tensor(mse_value)))
    result["loss/MSE"] = mse_value
    result["loss/RMSE"] = rmse_value

     # ターミナルに分かりやすく表示 
    print("\n=== 最良モデルの検証結果 ===")
    print(f"Train ID: {train_id}")
    print(f"Model:{cfg.model.name}")
    print(f"Test Data Size: {len(dataset)} 件")
    print(f"MSE:  {result['loss/MSE']:.5f}")
    print(f"RMSE: {result['loss/RMSE']:.5f}")
    print(f"推論速度: {avg_inference_time_ms:.3f} ms/枚")


    #保存処理
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # モデル別フォルダ構造に整理
    model_name = cfg.model.name
    output_root = Path("outputs/train_reg/history") / model_name / f"{train_id}"
    output_root.mkdir(parents=True, exist_ok=True)

    result_dir = output_root / "result"
    csv_dir = output_root/ "csv_result"
    bestpred_dir = output_root/ "best_predictions"

    result_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    bestpred_dir.mkdir(parents=True, exist_ok=True)
 
    # --- 学習済みモデルや設定ファイルは既に存在しているためコピー不要 ---
    # shutil.copy(model_path, output_root / "best_model.pth")
    # shutil.copy(config_path, output_root / "config.yaml")

    # --- learning_curve.png も学習時に生成済みなのでコピー不要 ---
    # learning_curve_src = output_root / "learning_curve.png"
    # if not learning_curve_src.exists():
    #     print(f"learning_curve.png が見つかりません: {learning_curve_src}")

    # ログ保存 (GitHubで追跡されるディレクトリへ)
    #log_dir = Path("outputs/train_reg/history") / train_id / "result"
    #log_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{train_id}_result.txt"
    with open(result_path, "w") as f:
        f.write(f"=== 最良モデルの検証結果 ===\n")
        f.write(f"Train ID: {train_id}\n")
        f.write(f"Model:{cfg.model.name}\n")
        f.write(f"Size: {len(dataset)} 件\n")
        f.write(f"MSE:  {result['loss/MSE']:.5f}\n")
        f.write(f"RMSE: {result['loss/RMSE']:.5f}\n")
        f.write(f"推論速度: {avg_inference_time_ms:.2f} ms/枚\n")

    # 予測結果保存
    #pred_dir = Path("outputs/train_reg/history") / train_id / "csv_result"
    #pred_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{train_id}_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_ev", "pred_ev"])
        writer.writerows(predictions)
    print(f"\n予測結果を {csv_path} に保存しました")

    # 補正画像保存
    if "original" in best_image_info:
        mean, std = cfg.dataset.test.transform.normalize.mean, cfg.dataset.test.transform.normalize.std
        denorm_img = denormalize(best_image_info["original"], mean, std)
        corrected_img = adjust_exposure(denorm_img, best_image_info["pred_ev"])

        #img_dir = Path("outputs/train_reg/history") / train_id / "best_predictions"
        #img_dir.mkdir(parents=True, exist_ok=True)
        # 元のファイル名から拡張子 (.jpgなど) を取り除く
        base_filename = Path(best_image_info['filename']).stem

        original_path = bestpred_dir / f"{base_filename}_補正前.png"
        corrected_path = bestpred_dir / f"{base_filename}_補正後.png"

        vutils.save_image(denorm_img, original_path)
        vutils.save_image(corrected_img, corrected_path)

        print(f"補正前後の画像を {output_root} に保存しました")
        #可視化関数呼び出し
        plot_ev_predictions(csv_path, output_root)

if __name__ == "__main__":
    main()
