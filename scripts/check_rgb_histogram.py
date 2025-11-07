import rawpy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# === 設定 ===
INPUT_DIR = Path("conf/dataset/HDR+burst/20171106/results_20171023")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === 1. .dngファイルを探索 ===
dng_files = list(INPUT_DIR.rglob("*.dng"))
if not dng_files:
    raise FileNotFoundError(f"{INPUT_DIR} に .dng ファイルが見つかりません")

# === 2. ランダムに1枚選択 ===
file_path = random.choice(dng_files)
print(f"\n 選ばれたファイル: {file_path}")

# === 3. RAWを読み込み、RGB化 ===
with rawpy.imread(str(file_path)) as raw:
    rgb = raw.postprocess(
        output_bps=16,           # 16bit出力
        no_auto_bright=True,
        use_auto_wb=False,
        gamma=(1, 1)             # リニア出力
    )

# === 4. 情報を出力 ===
bit_depth = 16 if rgb.dtype == np.uint16 else 8
print(f"--- 画像情報 ---")
print(f"shape: {rgb.shape}, dtype: {rgb.dtype}")
print(f"min={rgb.min()}, max={rgb.max()}, mean={rgb.mean():.2f}")
print(f"推定ビット深度: {bit_depth}bit\n")

# === 5. ヒストグラム（16bit値） ===
plt.figure(figsize=(8, 5))
plt.hist(rgb[..., 0].ravel(), bins=256, color='r', alpha=0.5, label='Red')
plt.hist(rgb[..., 1].ravel(), bins=256, color='g', alpha=0.5, label='Green')
plt.hist(rgb[..., 2].ravel(), bins=256, color='b', alpha=0.5, label='Blue')
plt.xlabel("Pixel value (0–65535)")
plt.ylabel("Pixel count")
plt.title(f"Histogram (16bit) - {file_path.name}")
plt.legend()
plt.tight_layout()

save_path = OUTPUT_DIR / "rgb_histogram_16bit.png"
plt.savefig(save_path, dpi=200)
plt.close()
print(f"ヒストグラム(16bit)を保存しました: {save_path}")

# === 6. 正規化して再度ヒストグラム ===
rgb_norm = np.float32(rgb) / 65535.0
plt.figure(figsize=(8, 5))
plt.hist(rgb_norm[..., 0].ravel(), bins=256, color='r', alpha=0.5, label='Red (norm)')
plt.hist(rgb_norm[..., 1].ravel(), bins=256, color='g', alpha=0.5, label='Green (norm)')
plt.hist(rgb_norm[..., 2].ravel(), bins=256, color='b', alpha=0.5, label='Blue (norm)')
plt.xlabel("Normalized value (0–1)")
plt.ylabel("Pixel count")
plt.title(f"Normalized RGB Histogram - {file_path.name}")
plt.legend()
plt.tight_layout()

save_path_norm = OUTPUT_DIR / "rgb_histogram_normalized.png"
plt.savefig(save_path_norm, dpi=200)
plt.close()
print(f"正規化後ヒストグラムを保存しました: {save_path_norm}")
