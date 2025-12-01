# 評価用コード
from pathlib import Path
import torch, csv, datetime, shutil
import torchvision.utils as vutils
from torchvision.transforms import v2
import hydra
from omegaconf import DictConfig
import scipy.stats #KLダイバージェイス計算用
import shutil


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio.v3 as iio

import time

from src.dataset import AnnotatedDatasetFolder, pil_loader,imageio_loader, dng_loader, collate_fn_skip_none, LogTransform
from src.model import SimpleCNN, ResNet,ResNetRegression, RegressionEfficientNet, RegressionMobileNet
from src.trainer import LossEvaluator
from src.util import set_random_seed

def denormalize(tensor, mean, std, inplace=False):
    tensor_copy = tensor if inplace else tensor.clone()
    for t, m, s in zip(tensor_copy, mean, std):
        t.mul_(s).add_(m)
    return tensor_copy

def adjust_exposure(image_tensor, ev_value):
    print(f"\n[DEBUG] EV: {ev_value:.4f}")
    print(f"  Input Min: {image_tensor.min():.6f}, Max: {image_tensor.max():.6f}, Mean: {image_tensor.mean():.6f}")
    # sRGB (非線形) -> Linear (線形)
    # 一般的なガンマ値 2.2 を使用
    #linear_image = torch.pow(image_tensor, 2.2)
    correction_factor = 2.0 ** ev_value
    corrected_linear_image = image_tensor * correction_factor #linear_image -> image_tensor
    # トーンマッピング (変更日11/30)
    # 変更前 : tone_mapped = corrected_linear_image / (corrected_linear_image + 1.0)
    # 変更後 (Code1): Clipping ( 1.0を超えたら切り捨て )
    # これにより、明るい部分は白飛びするようになります（アノテーション画面と同じ見た目）
    tone_mapped = torch.clamp(corrected_linear_image, 0.0, 1.0)
    # Linear (線形) -> sRGB (非線形) に戻す
    # ガンマ補正 (1 / 2.2) を適用
    corrected_srgb_image = torch.pow(tone_mapped, 1.0/2.2)
    # 最終結果を [0, 1] にクリップして返す
    return torch.clamp(corrected_srgb_image, 0.0, 1.0)

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
        plt.savefig(Path(output_dir) / "誤差分布_histogram.png")
        plt.close()

        #モデル予測値のみヒストグラム
        plt.figure(figsize=(6,4))
        plt.hist(df["pred_ev"], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel("Predicted EV")
        plt.ylabel("Frequency")
        plt.title("Predicted EV Distribution")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "予測値_ev_histogram.png")
        plt.close()

        #正解値のみヒストグラム
        plt.figure(figsize=(6,4))
        plt.hist(df["true_ev"], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel("Predicted EV")
        plt.ylabel("Frequency")
        plt.title("Predicted EV Distribution")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "正価値_ev_histogram.png")
        plt.close()

        # ヒストグラムの範囲を統一して計算
        # 範囲決定
        range_min = min(df["true_ev"].min(), df["pred_ev"].min())
        range_max = max(df["true_ev"].max(), df["pred_ev"].max())
        bins = np.linspace(range_min, range_max, 31) # 30ビン 論文を同じにすること

        # 度数分布を取得
        hist_true, _ = np.histogram(df["true_ev"], bins=bins, density=False)
        hist_pred, _ = np.histogram(df["pred_ev"], bins=bins, density=False)

        # 確率分布に変換 (和を1にする)
        prob_true = hist_true / (hist_true.sum() + 1e-10)
        prob_pred = hist_pred / (hist_pred.sum() + 1e-10)

        # ゼロ除算回避のための微小値加算
        epsilon = 1e-10
        prob_true = prob_true + epsilon
        prob_pred = prob_pred + epsilon
        
        # 再正規化
        prob_true /= prob_true.sum()
        prob_pred /= prob_pred.sum()

        # KLダイバージェンス計算: entropy(pk, qk) = sum(pk * log(pk / qk))
        # P:正解分布、Q:予測分布 とするのが一般的
        kl_value = scipy.stats.entropy(prob_true, prob_pred)

        # 描画
        plt.figure(figsize=(8, 5))
        # alphaで透過させて重ねる
        plt.hist(df["true_ev"], bins=bins, alpha=0.5, label='True EV', color='blue', density=True)
        plt.hist(df["pred_ev"], bins=bins, alpha=0.5, label='Pred EV', color='orange', density=True)
        
        plt.xlabel("EV")
        plt.ylabel("Density")
        plt.title(f"True vs Pred Distribution (KL Divergence = {kl_value:.4f})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "distribution_comparison_kl.png")
        plt.close()

        print(f"可視化グラフ保存完了: {output_dir}")
    except Exception as e:
        import traceback
        traceback.print_exc()
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


        latest_model_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
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
        net = ResNetRegression(**cfg.model.params).to(device) #ResNer -> ResNetRegression(事前学習ver)
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
        #v2.ToImage(),
        v2.Resize(cfg.dataset.test.transform.resize),
        v2.CenterCrop(cfg.dataset.test.transform.center_crop),
        #v2.ToDtype(torch.float32, scale=True),
        LogTransform(),
        v2.Normalize(**cfg.dataset.test.transform.normalize),
    ])
    
    print(f"[INFO] テストCSV: {cfg.dataset.test.csv_file}")
    print(f"[INFO] テスト画像ルート: {cfg.dataset.test.root}")

# test.csv を読み込む　質問2
    dataset = AnnotatedDatasetFolder(
        root=cfg.dataset.test.root,
        csv_file=cfg.dataset.test.csv_file,
        loader=imageio_loader, # imageio_loader から dng_loader
        transform=transform
    )
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        collate_fn=collate_fn_skip_none
    )
    #評価処理
    criterion_mse = torch.nn.MSELoss()
    criterion_mae = torch.nn.L1Loss() # MAE (L1)
    criterion_smooth = torch.nn.SmoothL1Loss(beta=1.0) # SmoothL1

    evaluator_mse = LossEvaluator(criterion_mse, "mse")
    evaluator_mae = LossEvaluator(criterion_mae, "mae")
    evaluator_smooth = LossEvaluator(criterion_smooth, "smooth_l1")

    evaluator_mse.initialize()
    evaluator_mae.initialize()
    evaluator_smooth.initialize()

    predictions = []
    best_image_info = {"min_error": float("inf")}

    ev_groups = {
    "EV0.5": [],
    "EV1.0": [],
    "EV1.5": [],
    "EV2.0": []
    }

    #推論速度計測用 start
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets, filenames) in enumerate(loader):
            if batch_idx == 0:
                print("Tensor range:", inputs.min().item(), "〜", inputs.max().item())

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            evaluator_mse.eval_batch(outputs, targets)
            evaluator_mae.eval_batch(outputs, targets)
            evaluator_smooth.eval_batch(outputs, targets)

            errors = torch.abs(outputs - targets).squeeze()
            if errors.dim() == 0: # バッチサイズが1の場合
                errors = errors.unsqueeze(0)

            for i, (filename, target, output) in enumerate(zip(filenames, targets, outputs)):
                t_val = target.item()
                p_val = output.item()
                predictions.append([filename, t_val, p_val])
                
                if errors[i] < best_image_info["min_error"]:
                    best_image_info.update({
                        "min_error": errors[i].item(),
                        "original": inputs[i].cpu(),
                        "pred_ev": p_val,
                        "true_ev": t_val,
                        "filename": filename
                    })
                item_data = {
                    "filename": filename,
                    "original": inputs[i].cpu(),
                    "true_ev": t_val,
                    "pred_ev": p_val
                }

                # --- EV値絶対値による分類 ---
                abs_ev = abs(t_val)
                error = abs(p_val - t_val)

                item_data["error"] = error

                if 0.5 <= abs_ev < 1.0:
                    ev_groups["EV0.5"].append(item_data)
                elif 1.0 <= abs_ev < 1.5:
                    ev_groups["EV1.0"].append(item_data)
                elif 1.5 <= abs_ev < 2.0:
                    ev_groups["EV1.5"].append(item_data)
                elif abs_ev >= 2.0:
                    ev_groups["EV2.0"].append(item_data)



    end_time = time.time()
    total_inference_time = end_time - start_time
    avg_inference_time_ms = (total_inference_time / len(dataset)) * 1000 if len(dataset) > 0 else 0


    #  評価結果計算
    result_mse = evaluator_mse.finalize()
    result_mae = evaluator_mae.finalize()
    result_smooth = evaluator_smooth.finalize()

    mse_value = result_mse.get("mse")
    mae_value = result_mae.get("mae")
    smooth_value = result_smooth.get("smooth_l1")

    if mse_value is None:
        raise KeyError("MSE計算エラー")
    
    rmse_value = float(np.sqrt(mse_value))
    
    result = {}
    result["MSE"] = mse_value
    result["RMSE"] = rmse_value
    result["MAE"] = mae_value
    result["SmoothL1"] = smooth_value

    # --- 総パラメータ数を計算 ---
    total_params = sum(p.numel() for p in net.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6

    #保存処理
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # モデル別フォルダ構造に整理
    model_name = cfg.model.name
    output_root = Path("outputs/train_reg/history") / model_name / f"{train_id}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    result_dir = output_root / "result"
    csv_dir = output_root/ "csv_result"
    bestpred_dir = output_root/ "best_predictions"
    ev_save_root = output_root / "extreme_predictions" / "EV_Groups"
    
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    bestpred_dir.mkdir(parents=True, exist_ok=True)
    ev_save_root.mkdir(parents=True, exist_ok=True)

 
    csv_path = csv_dir / f"{train_id}_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_ev", "pred_ev"])
        writer.writerows(predictions)
    print(f"\n予測結果保存:{csv_path} ")

    selected_samples = []

    for group_name, items in ev_groups.items():
        items.sort(key=lambda x: x["error"])  # 誤差が小さい順
        top2 = items[:2]  # 2枚だけ選択

        save_dir = ev_save_root / group_name
        save_dir.mkdir(exist_ok=True)

        for idx, sample in enumerate(top2, 1):
            sample["group"] = group_name
            sample["rank"] = idx
            sample["save_dir"] = save_dir
            selected_samples.append(sample)

    # 共通処理で画像生成とログ出力
    mean_params = cfg.dataset.train.transform.normalize.mean
    std_params  = cfg.dataset.train.transform.normalize.std

    for sample in selected_samples:
        save_dir = sample["save_dir"]

        s_filename = Path(sample["filename"]).stem
        s_true = sample["true_ev"]
        s_pred = sample["pred_ev"]
        s_label = f"{sample['group']}{sample['rank']}"

        # 復元処理
        denorm = denormalize(sample["original"], mean_params, std_params)
        temp = torch.pow(2.0, denorm) - 1.0
        if temp.max() > 100.0:
            linear = temp / 65535.0
        else:
            linear = temp
        linear = torch.clamp(linear, min=0.0)

        # 画像生成
        img_orig = adjust_exposure(linear, 0.0)
        img_pred_inv = adjust_exposure(linear, -s_pred) #修正：符号反転（修正前img_pred = adjust_exposure(linear, s_pred)）
        img_pred_raw = adjust_exposure(linear, s_pred)
        img_true_raw = adjust_exposure(linear, s_true)
        img_true_inv = adjust_exposure(linear, -s_true)
        img_diff = torch.abs(img_pred_inv - img_orig) * 10.0

        # 保存
        orig_path = save_dir / f"{s_filename}_補正前.png"
        pred_path_raw = save_dir / f"{s_filename}_補正後(Pred EV {s_pred:.4f}).png"
        true_path_raw = save_dir / f"{s_filename}_正解補正後(True EV {s_true:.4f}).png"
        pred_path_inv = save_dir / f"{s_filename}_補正後(反転)(Pred EV {-s_pred:.4f}).png"
        true_path_inv = save_dir / f"{s_filename}_正解補正後（反転）(True EV {-s_true:.4f}).png"
        diff_path = save_dir / f"{s_filename}_差分強調.png"

        vutils.save_image(img_orig, orig_path)
        vutils.save_image(img_true_inv, true_path_inv)
        vutils.save_image(img_true_raw, true_path_raw) 
        vutils.save_image(img_pred_inv, pred_path_inv)
        vutils.save_image(img_pred_raw, pred_path_raw) 
        vutils.save_image(img_diff, diff_path)

    
        # --- まとめて比較画像の保存 (グリッド表示) ---
        # 配列順序: 
        # [元画像, 正解(反転), 正解(生)]
        # [差分,   予測(反転), 予測(生)]
        compare_path = save_dir / f"{s_filename}_ALL_Comparison.png"
        comparison_list = [
            img_orig, img_true_inv, img_true_raw,
            img_diff, img_pred_inv, img_pred_raw
        ]
        
        # nrow=3 にすることで、横3枚で折り返しになり、2行3列の画像が生成されます
        vutils.save_image(comparison_list, compare_path, nrow=3, padding=5, normalize=False)

        # ビット深度確認
        dtype_str = "unknown"
        try:
            check_img = iio.imread(pred_path_inv)  # ← 保存したファイルパスを使う
            dtype_str = str(check_img.dtype)
        except:
            pass

        print(f"[SAVE] {s_label} → {save_dir}")
        print(f"  {s_filename}: Pred {s_pred:+.3f}, True {s_true:+.3f}")


        mean_orig = img_orig.mean().item()
        mean_pred = img_pred_inv.mean().item()
        diff_val  = mean_pred - mean_orig
        judge = "変化あり" if abs(diff_val) > 0.0001 else "変化なし"

        # 指定のログフォーマットで出力
        print(f"[Save Sample] {s_label}: {s_filename}")
        print(f"  Bit深度: {dtype_str}")
        print(f"  予測EV: {s_pred:.4f} / 正解EV: {s_true:.4f}")
        print(f"  平均輝度: {mean_orig:.4f} -> {mean_pred:.4f}")
        print(f"  輝度差分: {diff_val:+.5f}")
        print(f"  判定: {judge}")

    # 補正画像保存 
    if "original" in best_image_info:
        mean = cfg.dataset.train.transform.normalize.mean
        std  = cfg.dataset.train.transform.normalize.std

        #補正前の画像(EV=0 のsRGB画像として保存) <- ".png"で保存すると画像全体が暗くなるため、視覚的に比較しやすくするため
        denorm_img = denormalize(best_image_info["original"], mean, std)
    #対数修正その2
        # Log -> 線形 への「逆変換」
        #(x = 2^y - 1)
        temp = torch.pow(2.0, denorm_img) - 1.0
        linear_img = temp / 65535.0
        # (計算誤差でマイナスになるのを防ぐ)
        linear_img = torch.clamp(linear_img, min=0.0)

        #linear_img_normalized = linear_img / 65535.0
        
        # 確認用ログ（これで 0.0 〜 1.0 くらいになっていればOK）
        print(f"\n[Check] 0-1正規化後の最大値: {linear_img.max().item():.4f}")

        # max値が1.0に近い場合はすでに正規化されているので65535で割ってはいけない
        #max_val = linear_img.max().item()
        #print(f"\n[Check] 復元されたRawデータの最大値: {max_val:.2f}")
        
        # あなたの前処理コードでは 0-65535 なので、それに対応
        # トーンマップ用には [0,1] スケールが必要
        #if max_val > 100.0: # 明らかに1.0より大きい場合
            #linear_img_normalized = linear_img / 65535.0
            #print(" -> 16bitスケールと判定: 表示用に 1/65535を実施。")
        #else:
            #linear_img_normalized = linear_img
            #print(" -> 0-1スケールと判定: そのまま処理")

        # (補正前のEV値は 0.0 で固定)
        base_ev = 0.0 
        pred_ev = best_image_info["pred_ev"]
        true_ev = best_image_info["true_ev"]

        # adjust_exposure には「線形」の linear_img を渡す
        baseline_srgb_img = adjust_exposure(linear_img, base_ev) #対数修正その3:denorm_img -> linear_img
        #モデル予測値で補正した画像
        # モデル予測値 (2パターン)
        pred_corrected_img_inv = adjust_exposure(linear_img, -pred_ev) # 反転
        pred_corrected_img_raw = adjust_exposure(linear_img, pred_ev)  # そのまま
        #正解ラベル値で補正した画像
        # 正解ラベル (2パターン)
        true_corrected_img_inv = adjust_exposure(linear_img, -true_ev) # 反転
        true_corrected_img_raw = adjust_exposure(linear_img, true_ev)  # そのまま
        # 元のファイル名から拡張子 (.jpgなど) を取り除く
        base_filename = Path(best_image_info['filename']).stem


        original_path = bestpred_dir / f"{base_filename}_補正前(EV {base_ev:.4f}).png"
        pred_raw_path = bestpred_dir / f"{base_filename}_補正後(Pred EV {pred_ev:.4f}).png"
        true_raw_path = bestpred_dir / f"{base_filename}_正解補正後(True EV {true_ev:.4f}).png"
        pred_inv_path = bestpred_dir / f"{base_filename}_補正後(反転)(Pred EV {-pred_ev:.4f}).png"
        true_inv_path = bestpred_dir / f"{base_filename}_正解補正後（反転）(True EV {-true_ev:.4f}).png"
        path_diff = bestpred_dir / f"{base_filename}_差分強調画像.png"

        vutils.save_image(baseline_srgb_img, original_path)
        vutils.save_image(pred_corrected_img_inv, pred_inv_path)
        vutils.save_image(true_corrected_img_inv, true_inv_path)
        vutils.save_image(pred_corrected_img_raw, pred_raw_path)
        vutils.save_image(true_corrected_img_raw, true_raw_path)

        # 視覚的に変化を確認するための「差分強調画像」を作成,(補正後 - 補正前) の絶対値を 10倍 して保存
        diff_tensor = torch.abs(pred_corrected_img_inv - baseline_srgb_img) * 10.0
        vutils.save_image(diff_tensor, path_diff)

        # --- まとめて比較画像の保存 ---
        compare_path = bestpred_dir / f"{base_filename}_ALL_Comparison.png"
        comparison_list = [
            baseline_srgb_img, true_corrected_img_inv, true_corrected_img_raw,
            diff_tensor,       pred_corrected_img_inv, pred_corrected_img_raw
        ]
        # nrow=3 で 2行3列 の配置
        vutils.save_image(comparison_list, compare_path, nrow=3, padding=5, normalize=False)

        #出力画像が何bitなのか
        print("確認事項：画像bit深度")
        try:
            #保存した画像を読み込みして確認
            loaded_img = iio.imread(pred_inv_path)
            print(f"  保存ファイル: {pred_inv_path.name}")
            print(f"  データ型(dtype): {loaded_img.dtype}")

            if loaded_img.dtype == np.uint8:
                print("  → 結果: 8-bit 画像")
            elif loaded_img.dtype == np.uint16:
                print("  → 結果: 16-bit 画像")
            else:
                print(f"  → 結果: その他 ({loaded_img.dtype}) です。")
        except Exception as e:
            print(f"  確認エラー: {e}")
        
        #画像補正が正常に行なわれているのか
        print("確認事項：補正実施の数値確認")

        # 画像全体の平均輝度を計算
        mean_orig = baseline_srgb_img.mean().item()
        mean_pred = pred_corrected_img_inv.mean().item()
        diff_val  = mean_pred - mean_orig

        print(f"  予測EV値: {pred_ev:.4f}")
        print(f"  平均輝度 (補正前): {mean_orig:.5f}")
        print(f"  平均輝度 (補正後): {mean_pred:.5f}")
        print(f"  輝度差分        : {diff_val:+.5f}")

        # 判定
        if abs(diff_val) < 0.00001:
            print(" 判定: 変化なし")
            print(" 可能性1: 入力画像が真っ白/真っ黒になっている (上のMax値を確認)")
            print(" 可能性2: 予測EVが 0.0 に近すぎる")
        else:
            print(" 判定: 数値上で明るさが変化確認")
            print(f"視覚確認用画像を作成: {path_diff.name}")
            print("(この画像に何かが写っていれば、補正処理は機能")


        print("DEBUG img:", torch.isnan(baseline_srgb_img).any(), baseline_srgb_img.min(), baseline_srgb_img.max())
        print("denorm_img:", denorm_img.min().item(), denorm_img.max().item())
        print("linear_img:", linear_img.min().item(), linear_img.max().item())
        print("baseline_srgb_img:", baseline_srgb_img.min().item(), baseline_srgb_img.max().item())

        print(f"補正前後の画像を {output_root} に保存しました")

        # ターミナルに分かりやすく表示 
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

if __name__ == "__main__":
    main()