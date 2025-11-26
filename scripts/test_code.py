# 評価用コード
from pathlib import Path
import torch, csv, datetime, shutil
import torchvision.utils as vutils
from torchvision.transforms import v2
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio.v3 as iio
import time

from src.dataset import AnnotatedDatasetFolder, pil_loader, imageio_loader, dng_loader, collate_fn_skip_none, LogTransform
from src.model import SimpleCNN, ResNet, ResNetRegression, RegressionEfficientNet, RegressionMobileNet
from src.trainer import LossEvaluator
from src.util import set_random_seed

# --- 既存ヘルパー関数 ---
def denormalize(tensor, mean, std, inplace=False):
    tensor_copy = tensor if inplace else tensor.clone()
    for t, m, s in zip(tensor_copy, mean, std):
        t.mul_(s).add_(m)
    return tensor_copy

def adjust_exposure(image_tensor, ev_value):
    correction_factor = 2.0 ** ev_value
    corrected_linear_image = image_tensor * correction_factor 
    tone_mapped = corrected_linear_image / (corrected_linear_image + 1.0)
    corrected_srgb_image = torch.pow(tone_mapped, 1.0/2.2)
    return torch.clamp(corrected_srgb_image, 0.0, 1.0)

def plot_ev_predictions(csv_file, output_dir):
    try:
        df = pd.read_csv(csv_file)
        plt.figure(figsize=(6, 6))
        plt.scatter(df["true_ev"], df["pred_ev"], s=50, alpha=0.7)
        min_val = min(df.true_ev.min(), df.pred_ev.min())
        max_val = max(df.true_ev.max(), df.pred_ev.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal (y=x)")
        plt.xlabel("True EV"); plt.ylabel("Predicted EV"); plt.legend(); plt.grid(True)
        plt.savefig(Path(output_dir) / "scatter_ev.png"); plt.close()

        df["diff"] = df["pred_ev"] - df["true_ev"]
        plt.figure(figsize=(6,4))
        plt.hist(df["diff"], bins=30, alpha=0.7)
        plt.title(f"RMSE={np.sqrt((df['diff']**2).mean()):.3f}")
        plt.grid(True)
        plt.savefig(Path(output_dir) / "error_histogram.png"); plt.close()
        print(f"可視化グラフ保存完了: {output_dir}")
    except Exception as e:
        print(f"警告: グラフ描画失敗 {e}")


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    set_random_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = cfg.model.name
    history_dir = Path(f"outputs/train_reg/history/{model_name}")

    if "train_id" in cfg and cfg.train_id != "your_train_id_here":
        train_id = cfg.train_id
    else:
        if not history_dir.exists(): raise FileNotFoundError(f"履歴なし: {model_name}")
        run_dirs = [d for d in history_dir.iterdir() if d.is_dir() and (d / "best_model.pth").exists()]
        if not run_dirs: raise FileNotFoundError("学習済みモデルなし")
        latest_model_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
        train_id = latest_model_dir.name
        print(f"[INFO] train_id 自動取得: {train_id}")

    model_path = history_dir / train_id / "best_model.pth"
    print(f"[INFO] 使用モデル: {model_path}")

    # モデル
    if cfg.model.name.lower() == "simplecnn": net = SimpleCNN(**cfg.model.params).to(device)
    elif cfg.model.name.lower() == "resnet": net = ResNetRegression(**cfg.model.params).to(device)
    elif cfg.model.name.lower() == "efficientnet": net = RegressionEfficientNet(**cfg.model.params).to(device)
    elif cfg.model.name.lower() == "mobilenet": net = RegressionMobileNet(**cfg.model.params).to(device)
    else: raise ValueError(f"未対応モデル: {cfg.model.name}")
    
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # データセット
    transform = v2.Compose([
        v2.Resize(cfg.dataset.test.transform.resize),
        v2.CenterCrop(cfg.dataset.test.transform.center_crop),
        LogTransform(),
        v2.Normalize(**cfg.dataset.test.transform.normalize),
    ])
    
    print(f"[INFO] テストCSV: {cfg.dataset.test.csv_file}")
    dataset = AnnotatedDatasetFolder(
        root=cfg.dataset.test.root,
        csv_file=cfg.dataset.test.csv_file,
        loader=imageio_loader, 
        transform=transform
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.dataloader.batch_size, shuffle=False, collate_fn=collate_fn_skip_none)

    # 評価
    criterion_mse = torch.nn.MSELoss()
    criterion_mae = torch.nn.L1Loss()
    evaluator_mse = LossEvaluator(criterion_mse, "mse")
    evaluator_mae = LossEvaluator(criterion_mae, "mae")
    evaluator_mse.initialize(); evaluator_mae.initialize()

    predictions = []
    
    # 保存設定
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path("outputs/train_reg/history") / model_name / f"{train_id}_{timestamp}"
    
    result_dir = output_root / "result"
    csv_dir = output_root / "csv_result"
    bestpred_dir = output_root / "best_predictions"
    sample_dir   = output_root / "sample_predictions"
    
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    bestpred_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    best_image_info = {"min_error": float("inf")}
    count_high_ev = 0
    count_low_ev = 0

    start_time = time.time()
    print("\n=== 推論開始 (Sample画像探索中) ===")

    with torch.no_grad():
        for batch_idx, (inputs, targets, filenames) in enumerate(loader):
            if batch_idx == 0: print("Tensor range:", inputs.min().item(), "〜", inputs.max().item())
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            
            evaluator_mse.eval_batch(outputs, targets)
            evaluator_mae.eval_batch(outputs, targets)
            
            errors = torch.abs(outputs - targets).squeeze()
            if errors.dim() == 0: errors = errors.unsqueeze(0)

            # バッチ内ループ
            for i, (filename, target, output) in enumerate(zip(filenames, targets, outputs)):
                t_val = target.item()
                p_val = output.item()
                predictions.append([filename, t_val, p_val])

                # 1. [Best] 最小誤差の更新
                if errors[i] < best_image_info["min_error"]:
                    best_image_info.update({
                        "min_error": errors[i].item(),
                        "original": inputs[i].cpu(),
                        "pred_ev": p_val,
                        "true_ev": t_val,
                        "filename": filename
                    })

                # 2. [Sample] High/Low EV 保存判定
                save_prefix = None
                if t_val > 1.5 and count_high_ev < 3:
                    save_prefix = f"HighEV{count_high_ev+1}"
                    count_high_ev += 1
                elif t_val < -1.5 and count_low_ev < 3:
                    save_prefix = f"LowEV{count_low_ev+1}"
                    count_low_ev += 1
                
                # 保存実行 (関数を使わずインライン記述)
                if save_prefix is not None:
                    # --- データ復元 ---
                    mean = cfg.dataset.train.transform.normalize.mean
                    std  = cfg.dataset.train.transform.normalize.std
                    denorm_img = denormalize(inputs[i].cpu(), mean, std)
                    
                    # (2^x - 1) / 65535
                    temp = torch.pow(2.0, denorm_img) - 1.0
                    linear_img = temp
                    linear_img = torch.clamp(linear_img, min=0.0)
                    
                    # --- 画像生成 ---
                    base_filename = Path(filename).stem
                    img_orig = adjust_exposure(linear_img, 0.0)
                    img_pred = adjust_exposure(linear_img, p_val)
                    img_true = adjust_exposure(linear_img, t_val)
                    img_diff = torch.abs(img_pred - img_orig) * 10.0

                    # --- 保存 ---
                    path_pred = sample_dir / f"{save_prefix}_補正後(EV{p_val:.2f})_{base_filename}.png"
                    vutils.save_image(img_orig, sample_dir / f"{save_prefix}_補正前{base_filename}.png")
                    vutils.save_image(img_pred, path_pred)
                    vutils.save_image(img_true, sample_dir / f"{save_prefix}_正解補正後(EV{t_val:.2f})_{base_filename}.png")
                    vutils.save_image(img_diff, sample_dir / f"{save_prefix}_差分強調画像{base_filename}.png")

                    print(f"[Save Sample] {save_prefix}: {base_filename}")

                    # --- ★追加: 輝度・Bit深度確認ログ ---
                    try:
                        check = iio.imread(path_pred)
                        print(f"  Bit深度: {check.dtype}")
                    except: pass

                    mean_orig = img_orig.mean().item()
                    mean_pred = img_pred.mean().item()
                    diff_val  = mean_pred - mean_orig
                    
                    print(f"  予測EV: {p_val:.4f} / 正解EV: {t_val:.4f}")
                    print(f"  平均輝度: {mean_orig:.4f} -> {mean_pred:.4f}")
                    print(f"  輝度差分: {diff_val:+.5f}")
                    
                    if abs(diff_val) > 0.0001:
                        print("  判定: 変化あり")
                    else:
                        print("  判定: 変化なし")

    # --- ループ終了後 (Best画像保存) ---
    end_time = time.time()
    
    # 3. [Best] 保存処理
    if "original" in best_image_info:
        print("\n[Save] 最小誤差(Best)画像を保存します...")
        
        # --- データ復元 ---
        mean = cfg.dataset.train.transform.normalize.mean
        std  = cfg.dataset.train.transform.normalize.std
        denorm_img = denormalize(best_image_info["original"], mean, std)
        
        temp = torch.pow(2.0, denorm_img) - 1.0
        linear_img = temp / 65535.0
        linear_img = torch.clamp(linear_img, min=0.0)
        
        p_val = best_image_info["pred_ev"]
        t_val = best_image_info["true_ev"]
        base_filename = Path(best_image_info["filename"]).stem

        # --- 画像生成 ---
        img_orig = adjust_exposure(linear_img, 0.0)
        img_pred = adjust_exposure(linear_img, p_val)
        img_true = adjust_exposure(linear_img, t_val)
        img_diff = torch.abs(img_pred - img_orig) * 10.0

        # --- 保存 ---
        path_pred = bestpred_dir / f"誤差最小_補正後(EV{p_val:.2f})_{base_filename}.png"
        vutils.save_image(img_orig, bestpred_dir / f"誤差最小_補正前{base_filename}.png")
        vutils.save_image(img_pred, path_pred)
        vutils.save_image(img_true, bestpred_dir / f"誤差最小_正解補正後(EV{t_val:.2f})_{base_filename}.png")
        vutils.save_image(img_diff, bestpred_dir / f"誤差最小_誤差強調画像{base_filename}.png")

        # ---輝度確認ログ ---
        mean_orig = img_orig.mean().item()
        mean_pred = img_pred.mean().item()
        diff_val  = mean_pred - mean_orig

        print(f"  予測EV: {p_val:.4f}")
        print(f"  平均輝度: {mean_orig:.4f} -> {mean_pred:.4f}")
        print(f"  輝度差分: {diff_val:+.5f}")
        
        pred_ev_best = p_val
        true_ev_best = t_val
    else:
        pred_ev_best = 0.0; true_ev_best = 0.0

    # 結果集計
    result_mse = evaluator_mse.finalize()
    result_mae = evaluator_mae.finalize()
    rmse_value = float(np.sqrt(result_mse["mse"]))
    total_params = sum(p.numel() for p in net.parameters()) / 1e6
    avg_time = ((end_time - start_time) / len(dataset)) * 1000

    csv_path = csv_dir / f"{train_id}_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_ev", "pred_ev"])
        writer.writerows(predictions)
    print(f"\n予測結果保存: {csv_path}")

    print("\n=== 検証結果 ===")
    print(f"Train ID: {train_id}")
    print(f"Size: {len(dataset)}")
    print(f"MSE: {result_mse['mse']:.5f}")
    print(f"RMSE: {rmse_value:.5f}")
    print(f"MAE: {result_mae['mae']:.5f}")    
    print(f"Params: {total_params:.2f} M")
    print(f"Speed: {avg_time:.3f} ms/img")

    result_path = result_dir / f"{train_id}_result.txt"
    with open(result_path, "w") as f:
        f.write(f"=== 最良モデルの検証結果 ===\n")
        f.write(f"Train ID: {train_id}\n")
        f.write(f"MSE: {result_mse['mse']:.5f}\n")
        f.write(f"RMSE: {rmse_value:.5f}\n")
        f.write(f"MAE: {result_mae['mae']:.5f}\n") 
        f.write(f"Best Sample -> Pred: {pred_ev_best:.4f} / True: {true_ev_best:.4f}")

    plot_ev_predictions(csv_path, output_root)
    print(f"\n完了: {output_root}")

if __name__ == "__main__":
    main()