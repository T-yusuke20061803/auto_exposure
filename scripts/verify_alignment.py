import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np

# ================= 設定 =================
csv_path = Path("conf/dataset/HDR+burst_split/test.csv")
image_root = Path("conf/dataset/HDR+burst/processed_1024px_exr")
output_path = Path("outputs/dataset_verification_raw.png") # 保存ファイル名
num_samples = 5 # 確認する枚数
# ========================================

def normalize_simple(img):
    """
    画像ごとの自動調整を行わず、固定の計算式だけで表示する。
    これにより、画像の「本来の明るさの違い」が維持される。
    """
    img = img.astype(np.float32)
    img = np.nan_to_num(img)
    
    # 完全にデータがない場合
    if img.max() <= 0: return img

    # 単純なトーンマップ（Reinhard）のみ行う
    # x / (1 + x)
    img_mapped = img / (img + 1.0)
    
    # ガンマ補正 (sRGB表示用)
    img_gamma = np.power(img_mapped, 1.0/2.2)
    
    # 0-1クリップ
    img_final = np.clip(img_gamma, 0, 1)
    
    return img_final

def main():
    if not csv_path.exists():
        print(f"エラー: CSVが見つかりません {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # ランダムに5枚抽出
    samples = df.sample(n=num_samples)
    
    print(f"=== 画像データの可視化 ({num_samples}枚) ===")
    
    # 5枚を横に並べる設定
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
    if num_samples == 1: axes = [axes]
    
    for i, (_, row) in enumerate(samples.iterrows()):
        filename = row['Filename']
        ev_val = row['Exposure']
        ax = axes[i]
        
        print(f"Processing [{i+1}/{num_samples}]: {filename} (EV: {ev_val})")
        
        # フォルダ検索
        found_dirs = list(image_root.rglob(f"{filename}*"))
        target_file = None
        
        if found_dirs:
            base_path = found_dirs[0]
            if base_path.is_dir():
                # *.exr を優先検索
                exr = list(base_path.glob("*.exr"))
                if exr: target_file = exr[0]
                else: 
                    # なければ他のファイル
                    files = [p for p in base_path.iterdir() if p.is_file()]
                    if files: target_file = files[0]
            else:
                target_file = base_path
        
        if target_file:
            try:
                img = iio.imread(target_file)
                
                # 自動補正なしで表示（暗いものは暗く映る）
                img_disp = normalize_simple(img)
                
                ax.imshow(img_disp)
                
                # タイトルにEV値と判定を表示
                if ev_val < 0:
                    status = "Dark (-)"
                    color = "blue"
                else:
                    status = "Bright (+)"
                    color = "red"
                
                ax.set_title(f"EV: {ev_val}\n{status}", color=color, fontweight='bold', fontsize=12)
                
            except Exception as e:
                print(f"  Error: {e}")
        else:
            ax.text(0.5, 0.5, "Not Found", ha='center')
        
        ax.axis('off')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"\n確認画像を保存しました: {output_path.resolve()}")
    print("生成された画像を確認してください。")

if __name__ == "__main__":
    main()