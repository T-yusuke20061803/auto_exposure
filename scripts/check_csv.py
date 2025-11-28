import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg') # サーバー環境用（GUIなし）
import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np
import random

# ================= 設定 =================
csv_path = Path("conf/dataset/HDR+burst_split/test.csv")
image_root = Path("conf/dataset/HDR+burst/processed_1024px_exr")
output_check_img = Path("outputs/dataset_check_sample.png") # 確認用画像の保存先
# ========================================

def check_dataset_integrity(csv_path, root_dir):
    csv_p = Path(csv_path)
    root_p = Path(root_dir)

    print(f"\n{'='*40}")
    print(f"   データ整合性・不整合チェック")
    print(f"{'='*40}")
    print(f"CSVファイル: {csv_p}")
    print(f"画像ルート : {root_p}")

    # --- 1. パス存在確認 ---
    if not csv_p.exists():
        print(f"\n[Fatal Error] CSVファイルが見つかりません: {csv_p}")
        return
    if not root_p.exists():
        print(f"\n[Fatal Error] 画像ルートディレクトリが見つかりません: {root_p}")
        return

    # --- 2. CSV読み込み ---
    try:
        df = pd.read_csv(csv_p)
        print(f"CSV読み込み成功: {len(df)} 行")
        print(f"カラム一覧: {df.columns.tolist()}")
    except Exception as e:
        print(f"\n[Fatal Error] CSV読み込み失敗: {e}")
        return

    # 必須カラムチェック
    required_cols = ["filename"]
    for col in required_cols:
        if col not in df.columns:
            print(f"\n[Fatal Error] CSVに必須カラム '{col}' がありません。")
            return

    # --- 3. ファイル存在チェック (全件) ---
    print("\n--- [Step 1] ファイルの実在確認 (全件走査) ---")
    missing_files = []
    
    # 拡張子の傾向確認用
    extensions = set()

    for idx, row in df.iterrows():
        fname = str(row['filename']).strip()
        full_path = root_p / fname
        
        # 拡張子記録
        extensions.add(full_path.suffix)

        if not full_path.exists():
            missing_files.append({
                "index": idx,
                "filename": fname,
                "path": str(full_path)
            })

    # 結果表示
    if len(missing_files) == 0:
        print("OK: CSV内のすべての画像ファイルがディスク上に存在します。")
    else:
        print(f" NG: {len(missing_files)} / {len(df)} 件のファイルが見つかりません！")
        print("   [見つからないファイルの例 (先頭5件)]")
        for item in missing_files[:5]:
            print(f"    - Row {item['index']}: {item['filename']}")
    
    print(f"   検出された拡張子: {extensions}")


    # --- 4. ラベルとファイル名の整合性チェック (ランダムサンプリング) ---
    print("\n--- [Step 2] ファイル名とラベルの整合性確認 (ランダム5件) ---")
    print("※ ここを目視確認してください。「ファイル名に含まれる情報」と「EV値」はずれていませんか？")
    
    if len(df) > 0:
        sample_indices = random.sample(range(len(df)), min(5, len(df)))
        samples = df.iloc[sample_indices]
        
        print(f"{'Index':<6} | {'EV (Label)':<10} | {'Filename'}")
        print("-" * 50)
        
        # 'true_ev' カラムがあるか確認（CSVのヘッダー名に合わせてください。例: ev, label, true_ev）
        label_col = None
        for cand in ['true_ev', 'ev', 'label', 'EV']:
            if cand in df.columns:
                label_col = cand
                break
        
        for idx, row in samples.iterrows():
            fname = row['filename']
            label_val = row[label_col] if label_col else "N/A"
            print(f"{idx:<6} | {str(label_val):<10} | {fname}")
            
    # --- 5. 画像読み込みテスト (1枚だけ) ---
    print("\n--- [Step 3] 画像読み込みテスト (最初の1枚) ---")
    try:
        first_row = df.iloc[0]
        fname = first_row['filename']
        full_path = root_p / fname
        
        if full_path.exists():
            # EXR対応のために imageio を使用
            img = iio.imread(full_path)
            print(f"読み込み成功: {fname}")
            print(f"  Shape: {img.shape}")
            print(f"  Dtype: {img.dtype}")
            print(f"  Min: {img.min():.4f}, Max: {img.max():.4f}")
            
            # 簡易可視化 (トーンマップなしの単純クリップ表示)
            plt.figure(figsize=(5,5))
            # EXRなどのfloat画像の場合、表示用にクリップ
            disp_img = np.clip(img, 0, 1) 
            # チャンネル処理 (CHW -> HWC 等の確認は省略、通常imageioはHWCで読む)
            if disp_img.ndim == 3 and disp_img.shape[0] == 3: # もしCHWなら
                 disp_img = disp_img.transpose(1, 2, 0)
            
            plt.imshow(disp_img)
            plt.title(f"Check: {fname}")
            output_check_img.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_check_img)
            print(f"  確認用画像を保存しました: {output_check_img}")
            print("  → この画像を開いて、真っ黒/ノイズでないか確認してください。")
        else:
            print("  (最初のファイルが存在しないためスキップ)")

    except Exception as e:
        print(f"⚠️ 画像読み込み中にエラーが発生しました: {e}")
        print("  拡張子が .exr の場合、imageioのプラグイン(freeimageなど)が必要な場合があります。")

if __name__ == "__main__":
    check_dataset_integrity(csv_path, image_root)