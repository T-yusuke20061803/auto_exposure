import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np

# ================= 設定 =================
csv_path = Path("conf/dataset/HDR+burst_split/test.csv")
image_root = Path("conf/dataset/HDR+burst/processed_1024px_exr")
output_path = Path("outputs/force_visualization.png")
num_samples = 5
# ========================================

def normalize_for_display(img):
    """画像を強制的に0-1の範囲に正規化して見やすくする"""
    img = img.astype(np.float32)
    
    # 異常値（NaN/Inf）の除去
    img = np.nan_to_num(img)
    
    # ピクセル情報の統計を表示（デバッグ用）
    print(f"  [Stats] Min: {img.min():.4f}, Max: {img.max():.4f}, Mean: {img.mean():.4f}")
    
    # 全くデータがない場合
    if img.max() == img.min():
        return np.zeros_like(img)

    # 上位1%を最大値として正規化（外れ値対策）
    v_min = img.min()
    v_max = np.percentile(img, 99) # 99パーセンタイルをMaxとする
    if v_max == v_min: v_max = img.max()
    
    # 0-1に正規化
    img_norm = (img - v_min) / (v_max - v_min + 1e-8)
    img_norm = np.clip(img_norm, 0, 1)
    
    # ガンマ補正（暗い部分を持ち上げる）
    img_gamma = np.power(img_norm, 1/2.2)
    
    return img_gamma

def main():
    if not csv_path.exists():
        print("CSVが見つかりません")
        return

    df = pd.read_csv(csv_path)
    
    # ランダムにサンプリング
    samples = df.sample(n=num_samples, random_state=42)
    
    print(f"=== 画像データの強制可視化 ({num_samples}枚) ===")
    
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
    if num_samples == 1: axes = [axes]
    
    for i, (_, row) in enumerate(samples.iterrows()):
        filename = row['Filename']
        ev_val = row['Exposure']
        
        print(f"\nProcessing: {filename}")
        
        # 画像検索
        found = list(image_root.rglob(f"{filename}*"))
        if not found:
            print(f"  画像なし")
            continue
            
        img_path = found[0]
        
        try:
            # 画像読み込み
            img = iio.imread(img_path)
            
            # ★強制正規化処理
            img_disp = normalize_for_display(img)
            
            # 表示
            ax = axes[i]
            ax.imshow(img_disp)
            
            # CSVの値と状態を表示
            status = "Bright(+)" if ev_val > 0 else "Dark(-)"
            color = "red" if ev_val > 0 else "blue"
            
            ax.set_title(f"CSV: {ev_val}\n{status}", color=color, fontsize=12, fontweight='bold')
            ax.axis('off')
            
        except Exception as e:
            print(f"  読み込みエラー: {e}")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"\n\n確認画像を保存しました: {output_path.resolve()}")
    print("生成された画像を確認してください。")

if __name__ == "__main__":
    main()