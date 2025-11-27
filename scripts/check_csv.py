import pandas as pd
from pathlib import Path
import shutil
import os

# ================= 設定エリア =================
# 1. CSVのパス
csv_path = Path("conf/dataset/HDR+burst_split/test.csv")

# 2. 画像が格納されているルートディレクトリ
image_root_dir = Path("conf/dataset/HDR+burst/processed_1024px_exr")

# 3. 確認したいファイル名（拡張子なし、ログに出ていたもの）
target_filename = "0006_20160723_203435_899"

# 4. 見つけた画像を保存する場所 (outputs フォルダ)
output_dir = Path("outputs")
# =============================================

def main():
    print(f"=== 確認対象: {target_filename} ===")
    
    # --- 1. CSVデータの確認 ---
    try:
        if not csv_path.exists():
            print(f"エラー: CSVファイルが見つかりません -> {csv_path}")
            return

        df = pd.read_csv(csv_path)
        row = df[df['Filename'].astype(str).str.contains(target_filename)]
        
        if not row.empty:
            print("\n[CSVデータ確認]")
            ev_val = row['Exposure'].values[0]
            print(f"  CSV内の正解EV  : {ev_val}")
            
            if ev_val < 0:
                print("  判定: マイナス (暗い画像のはず)")
            else:
                print("  判定: プラス (明るい画像のはず)")
        else:
            print(f"\n[警告] ファイル名 {target_filename} がCSV内に見つかりませんでした。")

    except Exception as e:
        print(f"CSV読み込みエラー: {e}")
        return

    # --- 2. 画像ファイルの検索と保存 ---
    print("\n[画像ファイル検索と保存]")
    
    if not image_root_dir.exists():
        print(f"エラー: 画像ディレクトリが見つかりません -> {image_root_dir}")
        return

    found_files = list(image_root_dir.rglob(f"{target_filename}*"))
    
    if found_files:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for src_path in found_files:
            if src_path.is_file():
                dst_path = output_dir / src_path.name
                shutil.copy(src_path, dst_path)
                
                # ★修正: 絶対パスを取得して表示
                abs_path = dst_path.resolve()
                print(f"\n保存完了しました。以下のパスを確認してください:")
                print(f"{abs_path}")
                print("-" * 40)
    else:
        print(f"  エラー: 画像フォルダ内に '{target_filename}' を含むファイルが見つかりませんでした。")

if __name__ == "__main__":
    main()