import pandas as pd
from pathlib import Path

# 元のCSVファイル
input_csv = "conf/dataset/annotations.csv"

# 出力先
output_csv = "conf/dataset/annotations_clean.csv"

# データセットのrootフォルダ
root_dir = Path("conf/dataset/HDR_subdataset")

# CSVを読み込み
df = pd.read_csv(input_csv)

# 新しいCSVを作成
df_clean = pd.DataFrame({
    "Filename": [str(root_dir / f"{name}.jpg") for name in df["Filename"]],
    "Exposure": df["Exposure"]
})

# 保存
df_clean.to_csv(output_csv, index=False)
print(f"✅ Cleaned annotations saved to {output_csv}")
