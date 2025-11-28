import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np
import os

# ================= 設定 =================
csv_path = Path("conf/dataset/HDR+burst_split/test.csv")
image_root = Path("conf/dataset/HDR+burst/processed_1024px_exr")
output_path = Path("outputs/dataset_verification.png")
num_samples = 5
# ========================================
def check_dataset_integrity(csv_path, root_dir):
    csv_p = Path(csv_path)
    root_p = Path(root_dir)

    print(f"--- データ整合性チェック開始 ---")
    print(f"CSVファイル: {csv_p}")
    print(f"画像ルート : {root_p}\n")

    if not csv_p.exists():
        print(f"[Error] CSVファイルが見つかりません: {csv_p}")
        return
    if not root_p.exists():
        print(f"[Error] 画像ルートディレクトリが見つかりません: {root_p}")
        return

    # CSV読み込み
    try:
        df = pd.read_csv(csv_p)
        print(f"CSV読み込み成功: {len(df)} 行")
    except Exception as e:
        print(f"[Error] CSV読み込み失敗: {e}")
        return

    if "filename" not in df.columns:
        print(f"[Error] CSVに 'filename' カラムがありません。現在のカラム: {df.columns.tolist()}")
        return

    # --- チェック1: CSVにあるファイルが実際に存在するか ---
    missing_files = []
    
    # プログレスバー代わりにカウント表示
    print("ファイルの存在確認を実行中...")
    
    for idx, row in df.iterrows():
        fname = str(row['filename']).strip() # 空白除去
        
        # パスの結合 (pathlibはOSに合わせて区切り文字を自動調整します)
        full_path = root_p / fname
        
        if not full_path.exists():
            missing_files.append({
                "index": idx,
                "csv_filename": fname,
                "expected_path": str(full_path)
            })

    # --- 結果報告 ---
    print("\n" + "="*30)
    print("       診断結果")
    print("="*30)

    if len(missing_files) == 0:
        print(" 正常: CSV内のすべての画像ファイルが存在します。")
    else:
        print(f"警告: {len(missing_files)} 件のファイルが見つかりません！")
        print("\n[見つからないファイルの例 (最初の5件)]:")
        for item in missing_files[:5]:
            print(f"  Row {item['index']}: {item['csv_filename']}")
            print(f"    -> 探したパス: {item['expected_path']}")
        
        print("\n考えられる原因:")
        print("  1. CSV内のパスが相対パス('folder/img.dng')ではなく、ファイル名のみ('img.dng')になっている")
        print("  2. ルートディレクトリ(root_dir)の指定が間違っている")
        print("  3. 拡張子の大文字・小文字が違う (.dng vs .DNG)")
        print("  4. パス区切り文字の違い (Windows '\\' vs Linux '/')")

    # --- おまけ: 拡張子の確認 ---
    extensions = df['filename'].apply(lambda x: Path(x).suffix).unique()
    print(f"\n[Info] CSVに含まれる拡張子一覧: {extensions}")

if __name__ == "__main__":
    check_dataset_integrity(csv_path, image_root)