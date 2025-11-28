import pandas as pd
from pathlib import Path
import sys

# ================= 設定 =================
csv_path = Path("conf/dataset/HDR+burst_split/val.csv")
image_root = Path("conf/dataset/HDR+burst/processed_1024px_exr")
# ========================================

def check_filename_matching(csv_path, root_dir):
    csv_p = Path(csv_path)
    root_p = Path(root_dir)

    print(f"\n{'='*50}")
    print(f"   Filename 完全一致・過不足チェック")
    print(f"{'='*50}")

    # --- 1. CSV側の Filename を取得 (Set A) ---
    print("1. CSVファイルを読み込んでいます...")
    try:
        df = pd.read_csv(csv_p)
        # カラム名の揺らぎ吸収
        fname_col = "Filename" if "Filename" in df.columns else "filename"
        if fname_col not in df.columns:
            print(f"[Error] CSVにFilenameカラムがありません。")
            return
        
        # 空白除去してセットに格納
        csv_filenames = set(str(x).strip() for x in df[fname_col].unique())
        print(f"   -> CSV内のユニークなID数: {len(csv_filenames)}")
        
    except Exception as e:
        print(f"[Error] CSV読み込み失敗: {e}")
        return

    # --- 2. ディレクトリ側の Filename を取得 (Set B) ---
    print("2. 画像ディレクトリをスキャンしています...")
    if not root_p.exists():
        print(f"[Error] ディレクトリが存在しません: {root_p}")
        return

    # ルート直下の「ファイル名」または「フォルダ名」をすべて取得
    # 隠しファイル（.で始まるもの）は除外
    actual_filenames = set(p.name for p in root_p.iterdir() if not p.name.startswith('.'))
    print(f"   -> ディレクトリ内のアイテム数: {len(actual_filenames)}")

    # --- 3. 集合演算で比較 ---
    print("\n3. 突き合わせを実行中...")
    
    # A & B : 両方にある（OK）
    matched = csv_filenames & actual_filenames
    
    # A - B : CSVにあるが、フォルダにない（Missing）
    missing_in_dir = csv_filenames - actual_filenames
    
    # B - A : フォルダにあるが、CSVにない（Unused / Extra）
    extra_in_dir = actual_filenames - csv_filenames

    # --- 4. 結果レポート ---
    print(f"\n{'='*20} 結果レポート {'='*20}")
    
    print(f"✅ 一致 (OK)          : {len(matched)} 件")
    print(f"❌ 欠損 (Missing)     : {len(missing_in_dir)} 件 (CSVにあるがフォルダにない)")
    print(f"⚠️ 未使用 (Extra)     : {len(extra_in_dir)} 件 (フォルダにあるがCSVにない)")
    
    print("-" * 56)
    
    # 詳細表示
    if len(missing_in_dir) > 0:
        print("\n[❌ 欠損ファイル一覧 (最初の10件)]")
        print("これらは学習・評価時に「FileNotFoundError」を引き起こします。")
        for i, name in enumerate(sorted(list(missing_in_dir))):
            if i >= 10: break
            print(f"  - {name}")

    if len(extra_in_dir) > 0:
        print("\n[⚠️ 未使用ファイル一覧 (最初の10件)]")
        print("これらはCSVに含まれていないため、学習・評価には使われません。")
        for i, name in enumerate(sorted(list(extra_in_dir))):
            if i >= 10: break
            print(f"  - {name}")

    # --- 5. フォルダの中身空っぽチェック (重要) ---
    # Filenameが一致していても、その中身（画像）が空だとエラーになります。
    print(f"\n{'='*50}")
    print("   [追加チェック] 一致したフォルダの中身確認")
    print(f"{'='*50}")
    
    empty_folders = []
    check_count = 0
    
    # プログレス表示なしで全件チェックすると遅い場合があるので、ここでは全件チェック
    for name in matched:
        target_path = root_p / name
        
        if target_path.is_dir():
            # 中身に画像ファイルがあるか？
            # 拡張子は代表的なものを指定
            has_image = any(target_path.glob("*.exr")) or \
                        any(target_path.glob("*.jpg")) or \
                        any(target_path.glob("*.png")) or \
                        any(target_path.glob("*.dng"))
            
            if not has_image:
                empty_folders.append(name)
        
        # ファイルそのものの場合はチェック不要（存在確認済みなので）

    if len(empty_folders) == 0:
        print("✅ OK: 一致した全てのフォルダの中に、画像ファイル(.exr/.jpg/.png/.dng)が含まれています。")
    else:
        print(f"❌ NG: {len(empty_folders)} 件のフォルダが空（または画像なし）です！")
        for i, name in enumerate(sorted(empty_folders)):
            if i >= 5: break
            print(f"  - {name} (中身なし)")

if __name__ == "__main__":
    check_filename_matching(csv_path, image_root)