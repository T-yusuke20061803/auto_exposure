import imageio.v3 as iio
import numpy as np
from pathlib import Path

# 画像があるディレクトリ（どこでも良いので1枚選ぶ）
# さきほど作成したフォルダを指定してください
image_root = Path("conf/dataset/HDR+burst/processed_1024px_exr")

# フォルダ内の .exr ファイルを1つ探す
exr_files = list(image_root.rglob("*.exr"))

if exr_files:
    test_file = exr_files[0]
    print(f"検証ファイル: {test_file}")
    
    # 読み込み
    img = iio.imread(test_file)
    
    print(f"データ型 : {img.dtype}")
    print(f"最小値 (Min): {img.min():.4f}")
    print(f"最大値 (Max): {img.max():.4f}")
    print(f"平均値 (Mean): {img.mean():.4f}")
    
    # --- 判定 ---
    if img.max() > 1.0:
        print("\n👉 判定: 値は [0-65535] (またはそれ以上) です。")
        print("   対策: LogTransform で '65535倍' は【不要】です。")
    else:
        print("\n👉 判定: 値は [0-1] に正規化されています。")
        print("   対策: LogTransform で '65535倍' が【必要】です。")
else:
    print("EXRファイルが見つかりません。パスを確認してください。")