import pandas as pd
from pathlib import Path
import shutil
import os
import random
import imageio.v3 as iio
import numpy as np

# ================= 設定エリア =================
# 1. CSVのパス
csv_path = Path("conf/dataset/HDR+burst_split/test.csv")
# 2. 画像が格納されているルートディレクトリ
image_root_dir = Path("conf/dataset/HDR+burst/processed_1024px_exr")
# 3. 確認したい枚数
num_samples = 5
# 4. 保存先
output_dir = Path("outputs/random_check")
# =============================================

def save_as_png(src_path, dst_path):
    """
    EXRなどのHDR画像を、普通のビューアーで見れるPNGに変換して保存する関数
    (白飛び・黒つぶれ防止処理付き)
    """
    try:
        # 画像を読み込む
        img = iio.imread(src_path)
        img = img.astype(np.float32)
        img = np.nan_to_num(img) # エラー値除去

        # --- 白飛び防止 (トーンマッピング) ---
        # Reinhard法: x / (1 + x)
        # どんなに明るい光(100.0など)も、必ず 0.0〜1.0 の間に収めます。
        img_mapped = img / (img + 1.0)

        # --- 黒つぶれ防止 (ガンマ補正) ---
        # 暗い部分を人間が見やすい明るさに持ち上げます。
        img_gamma = np.power(img_mapped, 1.0/2.2)

        # 0-255 の整数に変換して保存
        img_8bit = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)
        iio.imwrite(dst_path, img_8bit)
        return True
    except Exception as e:
        print(f"  [Error] PNG変換失敗: {e}")
        return False

def main():
    print(f"=== ランダムに {num_samples} 枚の画像をPNG変換して確認します ===")
    
    # 保存先の初期化
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. CSV読み込みとランダムサンプリング ---
    try:
        if not csv_path.exists():
            print(f"エラー: CSVファイルが見つかりません -> {csv_path}")
            return

        df = pd.read_csv(csv_path)
        
        # ランダムに抽出 (毎回違う画像が選ばれます)
        n = min(num_samples, len(df))
        samples = df.sample(n=n) # random_stateを指定しないことで完全ランダムになります
        
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
        found_paths = list(image_root_dir.rglob(f"{filename}*"))
        target_image_path = None
        
        # 優先順位をつけて画像を探す (.exr優先)
        for p in found_paths:
            if p.is_file() and p.suffix.lower() in ['.exr', '.jpg', '.png', '.dng']:
                target_image_path = p
                break
            elif p.is_dir():
                # フォルダなら中身を探す
                exr_files = list(p.glob("*.exr"))
                if exr_files:
                    target_image_path = exr_files[0]
                    break
                else:
                    any_files = list(p.glob("*.*"))
                    if any_files:
                        target_image_path = any_files[0]
                        break
        
        # --- 保存 (PNG変換) ---
        if target_image_path:
            # ファイル名を変更し、拡張子を .png にする
            new_filename = f"[{status_str}_EV{ev_val}]_{filename}.png"
            dst_path = output_dir / new_filename
            
            # 変換して保存
            if save_as_png(target_image_path, dst_path):
                print(f"  保存成功(PNG): {dst_path.name}")
                success_count += 1
        else:
            print("  警告: 画像ファイルが見つかりませんでした。")
        
        print("-" * 40)

    # --- 終了報告 ---
    print(f"\n確認完了: {success_count}/{n} 枚をPNGで保存しました。")
    print(f"保存先フォルダ: {output_dir.resolve()}")
    print("\n【確認方法】")
    print("生成された PNG画像 を開いてください（ブラウザやスマホで見られます）。")
    print("  [Dark(Minus)...] が暗く、[Bright(Plus)...] が明るければ OK です。")

if __name__ == "__main__":
    main()