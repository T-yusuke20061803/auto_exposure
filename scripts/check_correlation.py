import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm # 進行状況表示用

# ================= 設定 =================
csv_path = Path("conf/dataset/HDR+burst_split/test.csv")
image_root = Path("conf/dataset/HDR+burst/processed_1024px_exr")
output_plot = Path("outputs/dataset_correlation_check.png")
# ========================================

def check_label_brightness_correlation(csv_path, root_dir):
    print(f"--- ラベルと画像輝度の相関チェック ---")
    
    # 1. CSV読み込み
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] {e}")
        return

    # カラム名特定
    fname_col = "Filename" if "Filename" in df.columns else "filename"
    label_col = "Exposure" if "Exposure" in df.columns else "ev"
    
    if label_col not in df.columns:
        # 他の候補を探す
        for c in df.columns:
            if c.lower() in ['ev', 'exposure', 'label', 'true_ev']:
                label_col = c
                break
    
    print(f"File Col: {fname_col}, Label Col: {label_col}")

    ev_values = []
    brightness_values = []
    
    print("画像をスキャンして輝度を計算中...")
    
    # 全件だと時間がかかる場合は一部サンプリングしてもOK
    # df = df.sample(n=min(len(df), 500), random_state=42) 

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        dir_name = str(row[fname_col]).strip()
        label = row[label_col]
        
        target_path = root_dir / dir_name
        
        # フォルダ内の画像を探す
        img_path = None
        if target_path.is_dir():
            for p in target_path.glob("*"):
                if p.suffix.lower() in ['.exr', '.jpg', '.png', '.dng', '.tif']:
                    img_path = p
                    break
        elif target_path.is_file():
            img_path = target_path
            
        if img_path:
            try:
                # 画像読み込み
                img = iio.imread(img_path)
                # 平均輝度を計算 (簡易的に全チャンネル平均)
                mean_val = np.mean(img)
                
                ev_values.append(label)
                brightness_values.append(mean_val)
            except:
                pass

    # --- グラフ描画 ---
    plt.figure(figsize=(8, 6))
    plt.scatter(ev_values, brightness_values, alpha=0.5, s=10)
    
    plt.title(f"Correlation: Label EV vs Image Brightness (N={len(ev_values)})")
    plt.xlabel("Label EV (Exposure)")
    plt.ylabel("Image Mean Brightness")
    plt.grid(True)
    
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot)
    plt.close()
    
    print(f"\n完了: 相関グラフを保存しました -> {output_plot}")
    print("【確認ポイント】")
    print("グラフが「右肩上がり」になっていれば正常です。")
    print("もし「水平（相関なし）」や「ランダム」であれば、ラベルと画像の中身が一致していません。")

if __name__ == "__main__":
    check_label_brightness_correlation(csv_path, image_root)