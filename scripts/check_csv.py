import pandas as pd
from pathlib import Path
import shutil
import os
import random

# ================= 設定エリア =================
# 1. CSVのパス
csv_path = Path("conf/dataset/HDR+burst_split/test.csv")

# 2. 画像が格納されているルートディレクトリ
image_root_dir = Path("conf/dataset/HDR+burst/processed_1024px_exr")

# 3. 確認したい枚数
num_samples = 5

# 4. 保存先 (outputs/random_check フォルダを作ります)
output_dir = Path("outputs/random_check")
# =============================================

def main():
    print(f"=== ランダムに {num_samples} 枚の画像を確認します ===")
    
    # 保存先の初期化（フォルダ作成）
    if output_dir.exists():
        shutil.rmtree(output_dir) # 前回の結果を削除
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. CSV読み込みとサンプリング ---
    try:
        if not csv_path.exists():
            print(f"エラー: CSVファイルが見つかりません -> {csv_path}")
            return

        df = pd.read_csv(csv_path)
        
        # ランダムに抽出
        # (データ数より要求数が多い場合のエラー回避)
        n = min(num_samples, len(df))
        samples = df.sample(n=n, random_state=random.randint(0, 10000))
        
        print(f"データセット総数: {len(df)} から {n} 枚を抽出しました。\n")

    except Exception as e:
        print(f"CSV読み込みエラー: {e}")
        return

    # --- 2. 各サンプルの処理 ---
    success_count = 0

    for i, (_, row) in enumerate(samples.iterrows()):
        filename = row['Filename']
        ev_val = row['Exposure']
        
        print(f"[{i+1}/{n}] Target: {filename}")
        print(f"  CSV EV値: {ev_val} ", end="")
        
        # 判定メッセージ
        if ev_val < 0:
            status_str = "Dark(Minus)"
            print("-> マイナス (暗い画像のはず)")
        else:
            status_str = "Bright(Plus)"
            print("-> プラス (明るい画像のはず)")

        # --- 画像検索 ---
        # ファイル名で検索 (ディレクトリかもしれないし、ファイルかもしれない)
        found_paths = list(image_root_dir.rglob(f"{filename}*"))
        
        target_image_path = None
        
        # 見つかったパスの中から「実体のある画像ファイル」を探す
        for p in found_paths:
            if p.is_file() and p.suffix.lower() in ['.exr', '.jpg', '.png', '.dng', '.tif']:
                target_image_path = p
                break
            elif p.is_dir():
                # ディレクトリなら中身を探す (例: merged.exr)
                files_in_dir = list(p.glob("*.exr")) # exr優先
                if not files_in_dir:
                    files_in_dir = list(p.glob("*.*")) # なければ何でも
                
                if files_in_dir:
                    target_image_path = files_in_dir[0]
                    break
        
        # --- 保存 ---
        if target_image_path:
            # ファイル名を分かりやすく変更して保存
            # 例: "[Dark_EV-3.0]_0006_....exr"
            new_filename = f"[{status_str}_EV{ev_val}]_{filename}{target_image_path.suffix}"
            dst_path = output_dir / new_filename
            
            shutil.copy(target_image_path, dst_path)
            print(f"  保存成功: {dst_path.name}")
            success_count += 1
        else:
            print("  警告: 画像ファイルが見つかりませんでした。")
        
        print("-" * 40)

    # --- 終了報告 ---
    print(f"\n確認完了: {success_count}/{n} 枚を保存しました。")
    print(f"保存先フォルダ: {output_dir.resolve()}")
    print("\n【確認方法】")
    print("1. このフォルダ内の画像を見てください。")
    print("2. ファイル名に [Dark_EV-3.0] とある画像が『暗い』か？")
    print("3. ファイル名に [Bright_EV+1.5] とある画像が『明るい』か？")
    print("これを確認すれば、符号の関係が確実に分かります。")

if __name__ == "__main__":
    main()