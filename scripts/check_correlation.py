import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np
import torch
from src.util import normalize_hdr

# ================= 設定 =================
csv_path = Path("conf/dataset/HDR+burst_split/train.csv")
processed_root = Path("conf/dataset/HDR+burst/processed_1024px_exr")
original_root = Path("conf/dataset/HDR+burst/20171106/results_20171023")
output_path = Path("outputs/visual_check_with_norm.png")
# ========================================

def get_dark_sample(df):
    # 暗い画像 (EV < -1.0) を選ぶ
    subset = df[df["Exposure"] < -1.0]
    if len(subset) > 0:
        return subset.sample(1).iloc[0]
    return df.sample(1).iloc[0]

def process_display(img_tensor):
    """表示用にトーンマップ(Clip)とガンマ補正"""
    clipped = np.clip(img_tensor, 0.0, 1.0)
    return np.power(clipped, 1.0/2.2)

def main():
    print("=== normalize_hdr を適用した正解比較 ===")
    df = pd.read_csv(csv_path)
    row = get_dark_sample(df)
    
    fname = row["Filename"]
    label_ev = float(row["Exposure"])
    
    exr_path = processed_root / row["filepath"]
    jpg_path = original_root / fname / "final.jpg"
    
    print(f"ID: {fname}")
    print(f"Label: {label_ev}")

    # 1. 画像読み込み
    try:
        # Input (EXR)
        img_raw = iio.imread(exr_path).astype(np.float32)
        if img_raw.max() > 1.0: img_raw /= 65535.0
        
        # Ground Truth (JPG)
        if jpg_path.exists():
            img_gt = iio.imread(jpg_path).astype(np.float32) / 255.0
        else:
            img_gt = np.zeros_like(img_raw) # ダミー
            
    except Exception as e:
        print(e)
        return

    # 2. normalize_hdr の適用 (ここが重要！)
    # HWCのまま渡す
    base_img, _, _ = normalize_hdr(img_raw, 0)
    
    # 3. 補正パターンの作成
    
    # A: そのまま (2^EV) - 反転なし
    factor_a = 2.0 ** label_ev
    img_a = base_img * factor_a
    
    # B: 反転 (2^-EV)
    factor_b = 2.0 ** (-label_ev)
    img_b = base_img * factor_b

    # --- プロット ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    # (1) Raw Input
    axes[0].imshow(process_display(img_raw))
    axes[0].set_title("Raw Input\n(Dark)", fontsize=12)
    
    # (2) Base Image (Normalized)
    axes[1].imshow(process_display(base_img))
    axes[1].set_title("Base Image\n(After normalize_hdr)", fontsize=12, color='blue')
    
    # (3) Corrected (反転なし)
    axes[2].imshow(process_display(img_a))
    axes[2].set_title(f"Current Code (2^EV)\nEV={label_ev}", fontsize=12, color='red')
    
    # (4) Answer
    axes[3].imshow(img_gt)
    axes[3].set_title("ANSWER (final.jpg)", fontsize=12, color='green', fontweight='bold')

    for ax in axes: ax.axis('off')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"✅ 画像を保存しました: {output_path}")
    print("→ 「Base Image」が明るくなっていることと、「Current Code」が「ANSWER」に似ていることを確認してください。")

if __name__ == "__main__":
    main()