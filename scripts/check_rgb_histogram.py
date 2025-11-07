import rawpy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# === 1. 入力フォルダ設定 ===
data_dir = Path("conf/dataset/HDR+burst/20171106/results_20171023")
dng_files = sorted(data_dir.glob("*.dng"))

if not dng_files:
    raise FileNotFoundError(f"{data_dir} に .dng ファイルが見つかりません")

# === 2. ランダムに1枚選択 ===
file_path = random.choice(dng_files)
print(f"選択されたファイル: {file_path.name}")

# === 3. 出力フォルダ設定 ===
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# === 4. RAW → RGB (16bit) 変換 ===
with rawpy.imread(str(file_path)) as raw:
    rgb = raw.postprocess(
        output_bps=16,           # 16bit出力
        no_auto_bright=True,     # 自動輝度補正なし
        use_auto_wb=False,       # 自動ホワイトバランスなし
        gamma=(1, 1)             # リニア出力（ガンマ補正なし）
    )

# === 5. 基本情報出力 ===
max_val = rgb.max()
min_val = rgb.min()
mean_val = rgb.mean()

# --- 実効ビット深度を推定 ---
def estimate_bit_depth(max_value):
    # max値に基づいて最も近いbit深度を推定
    possible_bits = [8, 10, 12, 14, 16]
    expected_max = [2**b - 1 for b in possible_bits]
    closest = min(expected_max, key=lambda x: abs(x - max_value))
    bit_depth = possible_bits[expected_max.index(closest)]
    return bit_depth

bit_depth = estimate_bit_depth(max_val)

print(f"\n--- {file_path.name} ---")
print(f"shape: {rgb.shape}, dtype: {rgb.dtype}")
print(f"min={min_val}, max={max_val}, mean={mean_val:.2f}")
print(f"→ 推定bit深度: {bit_depth}-bit\n")

# === 6. ヒストグラム（16bit値） ===
plt.figure(figsize=(8, 5))
plt.hist(rgb[..., 0].ravel(), bins=256, color='r', alpha=0.5, label='Red')
plt.hist(rgb[..., 1].ravel(), bins=256, color='g', alpha=0.5, label='Green')
plt.hist(rgb[..., 2].ravel(), bins=256, color='b', alpha=0.5, label='Blue')
plt.xlabel("Pixel value (0–65535)")
plt.ylabel("Pixel count")
plt.title(f"Histogram (Estimated {bit_depth}-bit)")
plt.legend()
plt.tight_layout()

save_path = output_dir / f"{file_path.stem}_histogram_16bit.png"
plt.savefig(save_path, dpi=200)
plt.close()
print(f"16bitヒストグラム保存: {save_path}")

# === 7. 正規化後のヒストグラム ===
rgb_norm = np.float32(rgb) / 65535.0
plt.figure(figsize=(8, 5))
plt.hist(rgb_norm[..., 0].ravel(), bins=256, color='r', alpha=0.5, label='Red (norm)')
plt.hist(rgb_norm[..., 1].ravel(), bins=256, color='g', alpha=0.5, label='Green (norm)')
plt.hist(rgb_norm[..., 2].ravel(), bins=256, color='b', alpha=0.5, label='Blue (norm)')
plt.xlabel("Normalized value (0–1)")
plt.ylabel("Pixel count")
plt.title(f"Histogram of normalized RGB values ({bit_depth}-bit source)")
plt.legend()
plt.tight_layout()

save_path_norm = output_dir / f"{file_path.stem}_histogram_normalized.png"
plt.savefig(save_path_norm, dpi=200)
plt.close()
print(f"正規化ヒストグラム保存: {save_path_norm}")

print("\n🎉 完了しました！")
