import rawpy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# === 1. 入力フォルダ設定 ===
data_dir = Path("conf/dataset/HDR+burst/20171106/results_20171023")
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# === 2. DNGファイルを取得 ===
dng_files = sorted(list(data_dir.rglob("*.dng")))
if len(dng_files) == 0:
    raise FileNotFoundError(f"{data_dir} に .dng ファイルが見つかりません")

# === 3. ランダムに1枚選択 ===
file_path = random.choice(dng_files)
print(f"選ばれたファイル: {file_path}")

# === 4. RAW読み込み & 処理 ===
with rawpy.imread(str(file_path)) as raw:
    # --- (A) RAWセンサー生データ情報 ---
    raw_data = raw.raw_image.astype(np.uint16)
    raw_min = raw_data.min()
    raw_max = raw_data.max()
    raw_mean = raw_data.mean()

    # 推定ビット深度 (例: max=4095 → 12bit)
    bit_depth_est = int(np.ceil(np.log2(raw_max + 1)))

    print("\n--- RAW生データ情報 ---")
    print(f"min={raw_min}, max={raw_max}, mean={raw_mean:.2f}")
    print(f"推定ビット深度（RAW元データ）: {bit_depth_est}bit")

    # --- (B) RGBにデモザイク変換（16bit出力） ---
    rgb = raw.postprocess(
        output_bps=16,
        no_auto_bright=True,
        use_auto_wb=False,
        gamma=(1, 1)
    )

# === 5. RGB画像情報を確認 ===
print("\n--- 画像情報（postprocess後） ---")
print(f"shape: {rgb.shape}, dtype: {rgb.dtype}")
print(f"min={rgb.min()}, max={rgb.max()}, mean={rgb.mean():.2f}")

# 推定ビット深度（RGB出力側）
rgb_bit_depth = int(np.ceil(np.log2(rgb.max() + 1)))
print(f"推定ビット深度（出力RGB）: {rgb_bit_depth}bit\n")

# === 6. ヒストグラム作成（16bit範囲） ===
plt.figure(figsize=(8, 5))
plt.hist(rgb[..., 0].ravel(), bins=256, color='r', alpha=0.5, label='Red')
plt.hist(rgb[..., 1].ravel(), bins=256, color='g', alpha=0.5, label='Green')
plt.hist(rgb[..., 2].ravel(), bins=256, color='b', alpha=0.5, label='Blue')
plt.xlabel("Pixel value (0–65535)")
plt.ylabel("Pixel count")
plt.title("Histogram of RAW→RGB values (16bit)")
plt.legend()
plt.tight_layout()

save_path = output_dir / "rgb_histogram_16bit.png"
plt.savefig(save_path, dpi=200)
plt.close()
print(f"ヒストグラムを保存しました: {save_path}")

# === 7. 正規化(0–1)ヒストグラムも保存 ===
rgb_norm = np.float32(rgb) / 65535.0
plt.figure(figsize=(8, 5))
plt.hist(rgb_norm[..., 0].ravel(), bins=256, color='r', alpha=0.5, label='Red (norm)')
plt.hist(rgb_norm[..., 1].ravel(), bins=256, color='g', alpha=0.5, label='Green (norm)')
plt.hist(rgb_norm[..., 2].ravel(), bins=256, color='b', alpha=0.5, label='Blue (norm)')
plt.xlabel("Normalized value (0–1)")
plt.ylabel("Pixel count")
plt.title("Histogram of normalized RGB values")
plt.legend()
plt.tight_layout()

save_path_norm = output_dir / "rgb_histogram_normalized.png"
plt.savefig(save_path_norm, dpi=200)
plt.close()
print(f"正規化後のヒストグラムを保存しました: {save_path_norm}\n")

print("処理完了！")
