import pandas as pd
from pathlib import Path
import shutil

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
        
        # ファイル名で検索
        row = df[df['Filename'].astype(str).str.contains(target_filename)]
        
        if not row.empty:
            print("\n[CSVデータ確認]")
            print(f"  CSV内のファイル名: {row['Filename'].values[0]}")
            ev_val = row['Exposure'].values[0]
            print(f"  CSV内の正解EV  : {ev_val}")
            
            print("-" * 30)
            if ev_val < 0:
                print(f"  判定: EVは {ev_val} (マイナス) です。")
                print("  意味: この画像は「暗い (Underexposed)」状態です。")
                print("  補正: 明るくするために「プラス」の補正が必要です。")
            else:
                print(f"  判定: EVは {ev_val} (プラス) です。")
                print("  意味: この画像は「明るい (Overexposed)」状態です。")
                print("  補正: 暗くするために「マイナス」の補正が必要です。")
            print("-" * 30)
                
        else:
            print(f"\n[警告] ファイル名 {target_filename} がCSV内に見つかりませんでした。")

    except Exception as e:
        print(f"CSV読み込みエラー: {e}")
        return

    # --- 2. 画像ファイルの検索と保存 ---
    print("\n[画像ファイル検索]")
    
    if not image_root_dir.exists():
        print(f"エラー: 画像ディレクトリが見つかりません -> {image_root_dir}")
        return

    # 拡張子が不明なため、ファイル名部分一致で検索します
    found_files = list(image_root_dir.rglob(f"{target_filename}*"))
    
    if found_files:
        # 保存先作成
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 見つかったファイルをコピー（複数あれば全て）
        for src_path in found_files:
            # フォルダではなくファイルのみ対象
            if src_path.is_file():
                dst_path = output_dir / src_path.name
                shutil.copy(src_path, dst_path)
                print(f"  発見・保存しました -> {dst_path}")
        
        print(f"\n完了: '{output_dir}' フォルダを確認してください。")
        print("保存された画像を開いて、CSVの判定（暗い/明るい）と一致するか確認してください。")
    else:
        print(f"  エラー: 画像フォルダ内に '{target_filename}' を含むファイルが見つかりませんでした。")

if __name__ == "__main__":
    main()