import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd


def split_dataset(
    dataset_name="HDR+burst",
    subset_name="20171106/results_20171023",
    annotations_csv="conf/dataset/annotations.csv",
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    seed=42
):
    # ===== パス設定 =====
    input_dir = Path(f"conf/dataset/{dataset_name}/{subset_name}")
    output_dir = Path(f"conf/dataset/{dataset_name}_split")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"入力ディレクトリが存在しません: {input_dir}")
    if not Path(annotations_csv).exists():
        raise FileNotFoundError(f"アノテーションCSVが存在しません: {annotations_csv}")

    # ===== 画像一覧取得 =====
    image_exts = [".jpg", ".jpeg", ".png"]
    image_paths = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in image_exts])
    if not image_paths:
        raise RuntimeError(f"画像が見つかりません: {input_dir}")

    print(f"総画像数: {len(image_paths)} 枚検出 ({input_dir})")

    # ===== アノテーション読み込み =====
    df_ann = pd.read_csv(annotations_csv)
    if "Filename" not in df_ann.columns or "Exposure" not in df_ann.columns:
        raise ValueError("annotations.csv に 'Filename' および 'Exposure' 列が必要です。")

    # ---- 正規化処理 ----
    df_ann["Filename"] = (
        df_ann["Filename"]
        .astype(str)
        .str.strip()
        .str.replace(".jpg", "", case=False)
        .str.replace(".jpeg", "", case=False)
        .str.replace(".png", "", case=False)
    )
    df_ann["Exposure"] = pd.to_numeric(df_ann["Exposure"], errors="coerce")

    df_imgs = pd.DataFrame({
        "filepath": [str(p) for p in image_paths],
        "Filename_ext": [p.parent.name for p in image_paths]
    })

    # ---- マージ ----
    df_merged = pd.merge(
        df_imgs,
        df_ann,
        left_on="Filename_noext",
        right_on=df_ann["Filename"].str.lower().str.strip(),
        how="inner"
    )

    if len(df_merged) == 0:
        raise RuntimeError("一致する画像がありません→至急：Filename列と拡張子を確認")

    print(f"一致した画像数: {len(df_merged)}件検出")

    # ===== データ分割 =====
    df_train_val, df_test = train_test_split(df_merged, test_size=test_size, random_state=seed, shuffle=True)
    val_ratio_adj = val_size / (1 - test_size)
    df_train, df_val = train_test_split(df_train_val, test_size=val_ratio_adj, random_state=seed, shuffle=True)

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    cols_to_save = ["filepath", "Filename", "Exposure", "split"]
    df_train[cols_to_save].to_csv(output_dir / "train.csv", index=False)
    df_val[cols_to_save].to_csv(output_dir / "val.csv", index=False)
    df_test[cols_to_save].to_csv(output_dir / "test.csv", index=False)

    # ===== 結果 =====
    print("\n=== データ分割結果 ===")
    print(f"train: {len(df_train)} 枚")
    print(f"val  : {len(df_val)} 枚")
    print(f"test : {len(df_test)} 枚")
    print(f"出力先: {output_dir}")

if __name__ == "__main__":
    split_dataset()
