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

# 既存ヘルパー関数
def denormalize(tensor, mean, std, inplace=False):
    tensor_copy = tensor if inplace else tensor.clone()
    for t, m, s in zip(tensor_copy, mean, std):
        t.mul_(s).add_(m)
    return tensor_copy

def adjust_exposure(image_tensor, ev_value):
    # sRGB (非線形) -> Linear (線形),一般的なガンマ値 2.2 を使用
    correction_factor = 2.0 ** ev_value
    corrected_linear_image = image_tensor * correction_factor 
    # トーンマッピング（Reinhard）
    tone_mapped = corrected_linear_image / (corrected_linear_image + 0.25)
    # Linear (線形) -> sRGB (非線形) に戻す,ガンマ補正 (1 / 2.2) を適用
    corrected_srgb_image = torch.pow(tone_mapped, 1.0/2.2)
    #最終結果を [0, 1] にクリップして返す
    return torch.clamp(corrected_srgb_image, 0.0, 1.0)
    return corrected_srgb_image

def plot_ev_predictions(csv_file, output_dir):
    try:
        df = pd.read_csv(csv_file)
        #散布図
        plt.figure(figsize=(6, 6))
        plt.scatter(df["true_ev"], df["pred_ev"], s=50, alpha=0.7)

        # 理想直線（y=x）を描画
        min_val = min(df.true_ev.min(), df.pred_ev.min())
        max_val = max(df.true_ev.max(), df.pred_ev.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal (y=x)")

        plt.xlabel("True EV ") #正解値
        plt.ylabel("Predicted EV ") #予測値
        plt.title("Predicted vs. True EV Scatter Plot")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "scatter_ev.png")
        plt.close()
        
        #誤差分布のヒストグラム
        df["diff"] = df["pred_ev"] - df["true_ev"]
        plt.figure(figsize=(6,4))
        plt.hist(df["diff"], bins=30, alpha=0.7)
        plt.xlabel("Prediction Error (Predicted - True) [EV]")
        plt.title(f"Prediction Error Distribution (RMSE={np.sqrt((df['diff']**2).mean()):.3f})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "error_histogram.png")
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

    # Train ID を config.yaml から取得 Train ID 自動検出 or 指定
    if "train_id" in cfg and cfg.train_id != "your_train_id_here":
        train_id = cfg.train_id
    else:
        if not history_dir.exists(): raise FileNotFoundError(f"履歴ディレクトリなし: {model_name}")
        # best_model.pthが存在するディレクトリの中から、更新日時が最新のものを探す
        run_dirs = [d for d in history_dir.iterdir() if d.is_dir() and (d / "best_model.pth").exists()]
        if not run_dirs: 
            raise FileNotFoundError(f"モデル '{model_name}' の学習済みモデル無し")
        latest_model_dir = max((p for p in history_dir.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
        train_id = latest_model_dir.name
        print(f"[INFO] train_id 自動取得: {train_id}")

    #モデルとパス
    model_path = history_dir / train_id / "best_model.pth"
    config_path = history_dir / train_id / "config.yaml"
    print(f"[INFO] 使用モデル: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"最良モデルが見つかりません: {model_path}")
    if not config_path.exists():
        print("[WARN] config.yaml が見つかりません。train_id のみで識別します。")

    # モデル構築
    if cfg.model.name.lower() == "simplecnn": 
        net = SimpleCNN(**cfg.model.params).to(device)
    elif cfg.model.name.lower() == "resnet": 
        net = ResNetRegression(**cfg.model.params).to(device)
    elif cfg.model.name.lower() == "efficientnet": 
        net = RegressionEfficientNet(**cfg.model.params).to(device)
    elif cfg.model.name.lower() == "mobilenet": 
        net = RegressionMobileNet(**cfg.model.params).to(device)
    else: 
        raise ValueError(f"未対応モデル: {cfg.model.name}")
    
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    #評価時においてはデータ拡張を行わない
    transform = v2.Compose([
        v2.Resize(cfg.dataset.test.transform.resize),
        v2.CenterCrop(cfg.dataset.test.transform.center_crop),
        LogTransform(),
        v2.Normalize(**cfg.dataset.test.transform.normalize),
    ])
    
    print(f"[INFO] テストCSV: {cfg.dataset.test.csv_file}")
    print(f"[INFO] テスト画像ルート: {cfg.dataset.test.root}")

    # test.csv を読み込む
    dataset = AnnotatedDatasetFolder(
        root=cfg.dataset.test.root,
        csv_file=cfg.dataset.test.csv_file,
        loader=imageio_loader, 
        transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        collate_fn=collate_fn_skip_none
    )

    # 評価指標
    criterion_mse = torch.nn.MSELoss()
    criterion_mae = torch.nn.L1Loss()
    criterion_smooth = torch.nn.SmoothL1Loss(beta=1.0)
    evaluator_mse = LossEvaluator(criterion_mse, "mse")
    evaluator_mae = LossEvaluator(criterion_mae, "mae")
    evaluator_smooth = LossEvaluator(criterion_smooth, "smooth_l1")
    evaluator_mse.initialize()
    evaluator_mae.initialize()
    evaluator_smooth.initialize()

    predictions = []
    
    #  保存用ディレクトリ設定 (秒数追加でID重複回避)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path("outputs/train_reg/history") / model_name / f"{train_id}_{timestamp}"
    # モデル別フォルダ構造に整理
    model_name = cfg.model.name
    output_root = Path("outputs/train_reg/history") / model_name / f"{train_id}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    
    result_dir = output_root / "result"
    csv_dir = output_root / "csv_result"
    bestpred_dir = output_root / "best_predictions"     # 誤差最小用
    sample_dir   = output_root / "sample_predictions"   # 変化大サンプル用
    
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    bestpred_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 保存用変数
    best_image_info = {"min_error": float("inf")}
    count_high_ev = 0
    count_low_ev = 0

    #推論速度計測用
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets, filenames) in enumerate(loader):
            if batch_idx == 0: print("Tensor range:", inputs.min().item(), "〜", inputs.max().item())
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            
            evaluator_mse.eval_batch(outputs, targets)
            evaluator_mae.eval_batch(outputs, targets)
            evaluator_smooth.eval_batch(outputs, targets)
            
            errors = torch.abs(outputs - targets).squeeze()
            if errors.dim() == 0: 
                errors = errors.unsqueeze(0)

            # --- バッチ内のループ処理 ---
            for i, (filename, target, output) in enumerate(zip(filenames, targets, outputs)):
                t_val = target.item()
                p_val = output.item()
                predictions.append([filename, t_val, p_val])

                # [Best] 誤差最小データを保持
                if errors[i] < best_image_info["min_error"]:
                    best_image_info.update({
                        "min_error": errors[i].item(),
                        "original": inputs[i].cpu(),
                        "pred_ev": p_val,
                        "true_ev": t_val,
                        "filename": filename
                    })

                # [Sample] 変化が大きい画像の保存 (High/Low EV)
                save_prefix = None
                if t_val > 1.5 and count_high_ev < 3:
                    save_prefix = f"HighEV{count_high_ev+1}"
                    count_high_ev += 1
                elif t_val < -1.5 and count_low_ev < 3:
                    save_prefix = f"LowEV{count_low_ev+1}"
                    count_low_ev += 1
                
                # 条件に合致した場合、その場で保存 (sample_predictionsへ)
                if save_prefix is not None:
                    # データ復元
                    mean = cfg.dataset.train.transform.normalize.mean
                    std  = cfg.dataset.train.transform.normalize.std
                    denorm_img = denormalize(inputs[i].cpu(), mean, std)
                    
                    # (2^x - 1) / 65535
                    temp = torch.pow(2.0, denorm_img) - 1.0
                    if temp.max() > 100.0:
                        linear_img = temp / 65535.0
                    else:
                        linear_img = temp
                    linear_img = torch.clamp(linear_img, min=0.0)
                    
                    # 画像生成
                    base_filename = Path(filename).stem
                    img_orig = adjust_exposure(linear_img, 0.0)
                    img_pred = adjust_exposure(linear_img, p_val)
                    img_true = adjust_exposure(linear_img, t_val)
                    img_diff = torch.abs(img_pred - img_orig) * 10.0

                    # 保存
                    path_pred = sample_dir / f"{save_prefix}_補正後(EV{p_val:.2f})_{base_filename}.png"
                    vutils.save_image(img_orig, sample_dir / f"{save_prefix}_補正前{base_filename}.png")
                    vutils.save_image(img_pred, path_pred)
                    vutils.save_image(img_true, sample_dir / f"{save_prefix}_正解補正後(EV{t_val:.2f})_{base_filename}.png")
                    vutils.save_image(img_diff, sample_dir / f"{save_prefix}_誤差強調画像{base_filename}.png")

                    print(f"[Save Sample] {save_prefix}: {base_filename} (EV {t_val:.2f})")
                    # 輝度・Bit深度確認
                    try:
                        check = iio.imread(path_pred)
                        print(f"  Bit深度: {check.dtype}")
                    except: pass

                    mean_orig = img_orig.mean().item()
                    mean_pred = img_pred.mean().item()
                    diff_val  = mean_pred - mean_orig
                    
                    print(f"  予測EV: {p_val:.4f}/ 正解EV: {t_val:.4f}")
                    print(f"  平均輝度（補正前）: {mean_orig:.5f}")
                    print(f"  平均輝度（補正後）: {mean_pred:.5f}")
                    print(f"  輝度差分: {diff_val:+.5f}")
                    
                    if abs(diff_val) > 0.0001:
                        print("  判定: 変化あり")
                    else:
                        print("  判定: 変化なし")

    # --- ループ終了後 (Best画像保存) ---
    end_time = time.time()
    
    # 3. [Best] 保存処理 (best_predictionsへ)
    if "original" in best_image_info:
        print("\n[Save] 最小誤差(Best)画像を保存")
        
        mean = cfg.dataset.train.transform.normalize.mean
        std  = cfg.dataset.train.transform.normalize.std
        denorm_img = denormalize(best_image_info["original"], mean, std)
        
        temp = torch.pow(2.0, denorm_img) - 1.0
        linear_img = temp / 65535.0
        linear_img = torch.clamp(linear_img, min=0.0)

        max_val = linear_img.max().item()
        print(f"\n[Check] 復元されたRawデータの最大値: {max_val:.2f}")
        # トーンマップ用には [0,1] スケールが必要
        if max_val > 100.0: # 明らかに1.0より大きい場合
            linear_img_normalized = linear_img / 65535.0
            print(" -> 16bitスケールと判定: 表示用に 1/65535を実施。")
        else:
            linear_img_normalized = linear_img
            print(" -> 0-1スケールと判定: そのまま処理")
        
        base_ev = 0.0
        pred_ev = best_image_info["pred_ev"] # 変数名を元のログに合わせる
        true_ev = best_image_info["true_ev"] # 変数名を元のログに合わせる
        base_filename = Path(best_image_info["filename"]).stem

        # 画像生成
        baseline_srgb_img = adjust_exposure(linear_img_normalized, base_ev)
        pred_corrected_img = adjust_exposure(linear_img_normalized, pred_ev)
        true_corrected_img = adjust_exposure(linear_img_normalized, true_ev)
        diff_tensor = torch.abs(pred_corrected_img - baseline_srgb_img) * 10.0

        # 保存
        original_path = bestpred_dir / f"{base_filename}誤差最小_補正前(EV {base_ev:+.4f}).png"
        pred_path = bestpred_dir / f"{base_filename}誤差最小_補正後(Pred EV {pred_ev:+.4f}).png"
        true_path = bestpred_dir / f"{base_filename}誤差最小_正解補正後(True EV {true_ev:+.4f}).png"
        path_diff = bestpred_dir / f"{base_filename}誤差最小_差分強調画像.png"

        vutils.save_image(baseline_srgb_img, original_path)
        vutils.save_image(pred_corrected_img, pred_path)
        vutils.save_image(true_corrected_img, true_path)
        vutils.save_image(diff_tensor, path_diff)

        # ログ出力の復元
        print("確認事項：画像bit深度")
        try:
            loaded_img = iio.imread(pred_path)
            print(f"  保存ファイル: {pred_path.name}")
            print(f"  データ型(dtype): {loaded_img.dtype}")
            if loaded_img.dtype == np.uint8:
                print("  → 結果: 8-bit")
            elif loaded_img.dtype == np.uint16:
                print("  → 結果: 16-bit")
            else:
                print(f"  → 結果: その他 ({loaded_img.dtype}) です。")
        except Exception as e:
            print(f"  確認エラー: {e}")

        # 輝度確認ログ
        print("確認事項：補正実施の数値確認")
        mean_orig = img_orig.mean().item()
        mean_pred = img_pred.mean().item()
        diff_val  = mean_pred - mean_orig

        print(f"  予測EV: {pred_ev:.4f}")
        print(f"  平均輝度（補正前）: {mean_orig:.5f}")
        print(f"  平均輝度（補正後）: {mean_pred:.5f}")
        print(f"  輝度差分: {diff_val:+.5f}")
        
        print("DEBUG img:", torch.isnan(baseline_srgb_img).any(), baseline_srgb_img.min(), baseline_srgb_img.max())
        print("denorm_img:", denorm_img.min().item(), denorm_img.max().item())
        print("linear_img:", linear_img.min().item(), linear_img.max().item())
        print("baseline_srgb_img:", baseline_srgb_img.min().item(), baseline_srgb_img.max().item())

        print(f"補正前後の画像を {output_root} に保存しました")

        pred_ev_best = pred_ev
        true_ev_best = true_ev
    else:
        pred_ev_best = 0.0; true_ev_best = 0.0

    # --- 結果集計とログ出力 ---
    total_inference_time = end_time - start_time
    avg_inference_time_ms = (total_inference_time / len(dataset)) * 1000 if len(dataset) > 0 else 0

    result_mse = evaluator_mse.finalize()
    result_mae = evaluator_mae.finalize()
    result_smooth = evaluator_smooth.finalize()

    # 値の取得
    mse_value = result_mse.get("mse")
    mae_value = result_mae.get("mae")
    smooth_value = result_smooth.get("smooth_l1")
    rmse_value = float(np.sqrt(mse_value))

    # パラメータ数計算
    total_params = sum(p.numel() for p in net.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6

    # CSV保存
    csv_path = csv_dir / f"{train_id}_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_ev", "pred_ev"])
        writer.writerows(predictions)
    print(f"\n予測結果保存: {csv_path}")

    # Result辞書
    result = {}
    result["MSE"] = mse_value
    result["RMSE"] = rmse_value
    result["MAE"] = mae_value
    result["SmoothL1"] = smooth_value

    # ★元のログ出力形式を完全復元
    print("\n=== 最良モデルの検証結果 ===")
    print(f"Train ID: {train_id}")
    print(f"Model:{cfg.model.name}")
    print(f"Test Data Size: {len(dataset)} 件")
    print(f"MSE:{result['MSE']:.5f}")
    print(f"RMSE:{result['RMSE']:.5f}")
    print(f"MAE:{result['MAE']:.5f}")    
    print(f"SmoothL1:{result['SmoothL1']:.5f}")
    print(f"総パラメータ数: {total_params:.2f} M (学習対象: {trainable_params:.2f} M)")
    print(f"推論速度: {avg_inference_time_ms:.3f} ms/枚")
    print(f"Pred EV: {pred_ev:.4f} / True EV: {true_ev:.4f}")

    result_path = result_dir / f"{train_id}_result.txt"
    with open(result_path, "w") as f:
        f.write(f"=== 最良モデルの検証結果 ===\n")
        f.write(f"Train ID: {train_id}\n")
        f.write(f"Model:{cfg.model.name}\n")
        f.write(f"Size: {len(dataset)} 件\n")
        f.write(f"MSE:{result['MSE']:.5f}\n")
        f.write(f"RMSE:{result['RMSE']:.5f}\n")
        f.write(f"MAE:{result['MAE']:.5f}\n") 
        f.write(f"SmoothL1:{result['SmoothL1']:.5f}\n")
        f.write(f"総パラメータ数: {total_params:.2f} M (学習対象: {trainable_params:.2f} M)\n")
        f.write(f"推論速度: {avg_inference_time_ms:.2f} ms/枚\n")
        f.write(f"  Pred EV: {pred_ev:.4f} / True EV: {true_ev:.4f}")

    plot_ev_predictions(csv_path, output_root)
    print(f"\n完了: {output_root}")

if __name__ == "__main__":
    main()