import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import imageio.v3 as iio
from PIL import Image
import numpy as np

# ================= 設定 =================
csv_path = Path("conf/dataset/HDR+burst_split/test.csv")
image_root = Path("conf/dataset/HDR+burst/processed_1024px_exr")
output_path = Path("outputs/dataset_verification.png")
num_samples = 5  # 確認する枚数
# ========================================

def main():
    if not csv_path.exists():
        print("CSVが見つかりません")
        return

    df = pd.read_csv(csv_path)
    
    # ランダムに5枚サンプリング（偏りを防ぐため）
    # または、特定のファイルを指定してもOK
    samples = df.sample(n=num_samples, random_state=42)
    
    print(f"=== データセット整合性確認 ({num_samples}枚) ===")
    
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
    
    for i, (_, row) in enumerate(samples.iterrows()):
        filename = row['Filename']
        ev_val = row['Exposure']
        
        # 画像検索
        found = list(image_root.rglob(f"{filename}*"))
        if not found:
            print(f"画像なし: {filename}")
            continue
            
        img_path = found[0]
        
        # 画像読み込み (EXR対応)
        try:
            img = iio.imread(img_path)
            # トーンマップ（表示用）
            img = img / (img + 1.0)
            img = np.clip(img ** (1/2.2), 0, 1)
        except:
            img = np.zeros((100,100,3)) # エラー時は黒画像

        # 表示
        ax = axes[i] if num_samples > 1 else axes
        ax.imshow(img)
        
        # タイトルに「CSVの値」を表示
        status = "Bright(+)" if ev_val > 0 else "Dark(-)"
        color = "red" if ev_val > 0 else "blue"
        
        ax.set_title(f"CSV: {ev_val}\n({status})", color=color, fontsize=12, fontweight='bold')
        ax.axis('off')
        
        print(f"File: {filename} | CSV Value: {ev_val}")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"\n確認画像を保存しました: {output_path.resolve()}")
    print("この画像を見て、「CSVの値」と「画像の見た目」が一致していれば、ズレはありません。")

if __name__ == "__main__":
    main()