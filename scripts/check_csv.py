import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg') # サーバー環境用
import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np
import random

# ================= 設定 =================
csv_path = Path("conf/dataset/HDR+burst_split/test.csv")
image_root = Path("conf/dataset/HDR+burst/processed_1024px_exr")
output_check_img = Path("outputs/dataset_check_sample.png") 
# ========================================

def check_dataset_integrity(csv_path, root_dir):
    csv_p = Path(csv_path)
    root_p = Path(root_dir)

    print(f"\n{'='*40}")
    print(f"   データ整合性・不整合チェック (修正版)")
    print(f"{'='*40}")
    print(f"CSVファイル: {csv_p}")
    print(f"画像ルート : {root_p}")

    if not csv_p.exists():
        print(f"\n[Fatal Error] CSVファイルが見つかりません: {csv_p}")
        return
    if not root_p.exists():
        print(f"\n[Fatal Error] 画像ルートディレクトリが見つかりません: {root_p}")
        return

    # CSV読み込み
    try:
        df = pd.read_csv(csv_p)
        print(f"CSV読み込み成功: {len(df)} 行")
        print(f"カラム一覧: {df.columns.tolist()}")
    except Exception as e:
        print(f"\n[Fatal Error] CSV読み込み失敗: {e}")
        return

    # ★ 修正: カラム名の揺らぎを吸収
    fname_col = "Filename" if "Filename" in df.columns else "filename"
    if fname_col not in df.columns:
        # filepathしかない場合の対応
        if "filepath" in df.columns:
            fname_col = "filepath"
            print(f"※ 'Filename' カラムがないため 'filepath' を使用します。")
        else:
            print(f"\n[Fatal Error] 画像ファイル名を特定できるカラム(Filename, filename, filepath)がありません。")
            return

    # ★ 修正: EV値カラムの特定
    label_col = None
    # 優先順位: Exposure -> ev -> label -> true_ev
    for cand in ['Exposure', 'ev', 'label', 'true_ev']:
        if cand in df.columns:
            label_col = cand
            break
    
    print(f"使用カラム -> ファイル名: '{fname_col}', EV値: '{label_col}'")

    # --- ファイル存在チェック ---
    print("\n--- [Step 1] ファイルの実在確認 (全件走査) ---")
    missing_files = []
    
    for idx, row in df.iterrows():
        fname = str(row[fname_col]).strip()
        # ファイル名がパスを含む場合と含まない場合に対応
        # root_p と結合して確認
        full_path = root_p / fname
        
        if not full_path.exists():
            # もしファイル名だけ(Filename)でなく、filepathがフルパスに近い場合の救済措置確認
            # ここでは単純に結合のみチェック
            missing_files.append({
                "index": idx,
                "filename": fname,
                "path": str(full_path)
            })

    if len(missing_files) == 0:
        print("✅ OK: CSV内のすべての画像ファイルがディスク上に存在します。")
    else:
        print(f"❌ NG: {len(missing_files)} / {len(df)} 件のファイルが見つかりません！")
        print("   [見つからないファイルの例 (先頭5件)]")
        for item in missing_files[:5]:
            print(f"    - Row {item['index']}: {item['filename']}")
            print(f"      (探した場所: {item['path']})")

    # --- 整合性チェック (ここが本題) ---
    print("\n--- [Step 2] ファイル名とラベルの整合性確認 (ランダム5件) ---")
    print("※ 以下の表を見て、「Filename」に含まれるEV値と、「Exposure」の値が一致しているか確認してください。")
    
    if len(df) > 0:
        sample_indices = random.sample(range(len(df)), min(5, len(df)))
        samples = df.iloc[sample_indices]
        
        print(f"{'Index':<6} | {str(label_col):<12} | {fname_col}")
        print("-" * 60)
        
        for idx, row in samples.iterrows():
            fname = row[fname_col]
            label_val = row[label_col] if label_col else "N/A"
            print(f"{idx:<6} | {str(label_val):<12} | {fname}")
            
    # --- 画像読み込みテスト ---
    print("\n--- [Step 3] 画像読み込みテスト (最初の1枚) ---")
    try:
        first_row = df.iloc[0]
        fname = first_row[fname_col]
        full_path = root_p / fname
        
        if full_path.exists():
            img = iio.imread(full_path)
            print(f"読み込み成功: {fname}")
            print(f"  Shape: {img.shape}")
            print(f"  Min: {img.min():.4f}, Max: {img.max():.4f}")
            
            plt.figure(figsize=(5,5))
            disp_img = np.clip(img, 0, 1) 
            # チャンネル処理 (HWC想定)
            if disp_img.ndim == 3 and disp_img.shape[0] == 3: 
                 disp_img = disp_img.transpose(1, 2, 0)
            
            plt.imshow(disp_img)
            plt.title(f"Check: {fname}\nLabel: {first_row[label_col] if label_col else 'None'}")
            output_check_img.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_check_img)
            print(f"  確認用画像を保存しました: {output_check_img}")
        else:
            print("  (最初のファイルが存在しないためスキップ)")

    except Exception as e:
        print(f"⚠️ 画像読み込み中にエラーが発生しました: {e}")

if __name__ == "__main__":
    check_dataset_integrity(csv_path, image_root)