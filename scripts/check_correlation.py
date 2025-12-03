import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np
import torch

# ================= 設定 =================
# 学習に使っているCSV
csv_path = Path("conf/dataset/HDR+burst_split/train.csv")

# 前処理後のEXRがある場所
processed_root = Path("conf/dataset/HDR+burst/processed_1024px_exr")

# ★重要: 元のデータセット（final.jpgが入っている場所）を指定してください
# 前処理コードの INPUT_DIR と同じ場所です
original_root = Path("conf/dataset/HDR+burst/20171106/results_20171023")

output_path = Path("outputs/ground_truth_check.png")
# ========================================

def get_sample_pair(df):
    # 矛盾が起きていた「暗い画像 (EV < -1.0)」を探す
    subset = df[df["Exposure"] < -1.0]
    if len(subset) > 0:
        return subset.sample(1).iloc[0]
    return df.sample(1).iloc[0]

def process_image(img, ev):
    """トーンマップとガンマ補正"""
    # 1. 露出補正 (Linear)
    factor = 2.0 ** ev
    corrected = img * factor
    # 2. クリップ (0-1)
    clipped = np.clip(corrected, 0.0, 1.0)
    # 3. ガンマ補正 (sRGB)
    gamma = np.power(clipped, 1.0/2.2)
    return gamma

def main():
    print("=== 正解画像(final.jpg)との答え合わせ ===")
    
    df = pd.read_csv(csv_path)
    row = get_sample_pair(df)
    
    dir_name = row["Filename"]     # ID (フォルダ名)
    label_ev = float(row["Exposure"])
    
    # パスの構築
    exr_path = processed_root / row["filepath"]
    
    # final.jpg を探す (original_root/ID/final.jpg)
    jpg_path = original_root / dir_name / "final.jpg"
    
    print(f"ID: {dir_name}")
    print(f"Label EV: {label_ev}")
    print(f"EXR Path: {exr_path}")
    print(f"JPG Path: {jpg_path}")
    
    if not jpg_path.exists():
        print(f"[Error] 正解画像(final.jpg)が見つかりません: {jpg_path}")
        print("original_root のパス設定を確認してください。")
        return

    # 画像読み込み
    try:
        # EXR (Input)
        img_input = iio.imread(exr_path).astype(np.float32)
        if img_input.max() > 1.0: img_input /= 65535.0
            
        # JPG (Ground Truth)
        img_gt = iio.imread(jpg_path).astype(np.float32) / 255.0
        
    except Exception as e:
        print(f"画像読み込みエラー: {e}")
        return

    # --- 比較画像の生成 ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    # 1. Input (補正なし)
    view_input = process_image(img_input, 0.0)
    axes[0].imshow(view_input)
    axes[0].set_title(f"Input (No shift)\nMean: {img_input.mean():.4f}")
    
    # 2. Current Code (2^EV)
    view_current = process_image(img_input, label_ev)
    axes[1].imshow(view_current)
    axes[1].set_title(f"A: Current (2^EV)\nEV={label_ev}", color='red', fontsize=14)

    # 3. Inverted Code (2^-EV)
    view_inverted = process_image(img_input, -label_ev)
    axes[2].imshow(view_inverted)
    axes[2].set_title(f"B: Inverted (2^-EV)\nEV={-label_ev}", color='blue', fontsize=14)
    
    # 4. Ground Truth (final.jpg)
    axes[3].imshow(img_gt)
    axes[3].set_title("★ ANSWER (final.jpg) ★", color='green', fontsize=14, fontweight='bold')
    axes[3].set_xlabel(f"Target Mean: {img_gt.mean():.4f}")

    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"\n✅ 確認画像保存完了: {output_path}")
    print("【判定方法】")
    print("一番右の『ANSWER (final.jpg)』に似ているのは、A(赤) と B(青) のどちらですか？")
    print("似ている方が、このデータセットにおける「正しい計算式」です。")

if __name__ == "__main__":
    main()