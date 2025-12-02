import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm


def split_dataset(
    dataset_name="HDR+burst",
    subset_name="processed_1024px_exr",  #変更前：20171106/results_20171023 ← 使用フォルダをここで変更
    annotations_csv="conf/dataset/annotations.csv",
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    seed=42
):
    """
    HDR+burst データセット (merged.dng を使用) を train/val/test に分割し、
    annotations.csv に記録された露出値（Exposure）を統合。
    拡張子自動判定 & 欠損チェック対応。
    """

    # === ディレクトリ設定 ===
    input_dir = Path(f"conf/dataset/{dataset_name}/{subset_name}")
    output_dir = Path(f"conf/dataset/{dataset_name}_split")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 入力確認
    if not input_dir.exists():
        raise FileNotFoundError(f"入力ディレクトリが存在しません: {input_dir}")
    if not Path(annotations_csv).exists():
        raise FileNotFoundError(f"アノテーションCSVが存在しません: {annotations_csv}")

    # 画像ファイル検索
    print(f"{input_dir} 内の merged.exr ファイルを検索中")
    image_paths = sorted(list(input_dir.rglob("merged.exr"))) #11/28 merged.dng -> mergeged.exr
    if not image_paths:
        raise RuntimeError(f"merged.exr 無し: {input_dir}")

    print(f"総画像数: {len(image_paths)} 枚検出 ({input_dir})")

    # annotations.csv 読み込み
    df_ann = pd.read_csv(annotations_csv)
    # 重複行がある場合は最初の1つを採用
    df_ann = df_ann.drop_duplicates(subset="Filename", keep="last")#keep="first"で最初に記録されたExposureを優先、keep="last"で最後に記録されたExposureを優先
    if "Filename" not in df_ann.columns or "Exposure" not in df_ann.columns: #columns:列の項目名を指定
        raise ValueError("annotations.csv に 'Filename' + 'Exposure' 列が必要です。")

    # Exposure欠損チェック
    missing_expo = df_ann["Exposure"].isna().sum() #isna():データの欠損値を判定する関数
    if missing_expo > 0:
        print(f"警告: Exposure が欠損している行有り： {missing_expo} 件")

    # 画像ファイル情報をDataFrame化
    # 各画像の親フォルダ名（例：0006_20160721_163256_525）をキーとして利用
    df_imgs = pd.DataFrame({
        # input_dir からの相対パスを保存する
        "filepath": [str(p.relative_to(input_dir)) for p in image_paths],
        "Filename": [p.parent.name for p in image_paths], # 親フォルダ名
        "extension": [p.suffix.lower() for p in image_paths]
    })

    #マージ（Filenameベース）
    df_merged = pd.merge(df_imgs, df_ann, on="Filename", how="inner")

    # 一致・不一致数を報告
    print(f"一致した画像数: {len(df_merged)}枚 / 総画像数: {len(df_imgs)}枚")
    dupes = df_ann[df_ann.duplicated("Filename", keep=False)]
    print(f"重複しているFilename数: {dupes['Filename'].nunique()}件")
    if not dupes.empty:
        print(dupes.head(10))

    unmatched = set(df_imgs["Filename"]) - set(df_merged["Filename"])
    if unmatched:
        print(f"一致しなかった画像フォルダ数: {len(unmatched)}枚")
        print(f"例: {list(unmatched)[:5]} ")

    if len(df_merged) == 0:
        raise RuntimeError("一致する画像(merged.dng)が無し→確認事項：Filename列")

    # データ分割
    df_train_val, df_test = train_test_split(
        df_merged, test_size=test_size, random_state=seed, shuffle=True
    )
    val_ratio_adj = val_size / (1 - test_size) if (1 - test_size) > 0 else 0
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_ratio_adj, random_state=seed, shuffle=True
    )

    # split列を追加
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    # CSV出力
    cols_to_save = ["filepath", "Filename", "Exposure", "split"]
    df_train[cols_to_save].to_csv(output_dir / "train.csv", index=False)
    df_val[cols_to_save].to_csv(output_dir / "val.csv", index=False)
    df_test[cols_to_save].to_csv(output_dir / "test.csv", index=False)

    # 結果出力
    print("\n=== データ分割結果 ===")
    print(f"train: {len(df_train)} 枚")
    print(f"val  : {len(df_val)} 枚")
    print(f"test : {len(df_test)} 枚")
    print(f"出力先: {output_dir}")

    print("\n出力されたCSV:")
    print(f"  train.csv -> {output_dir / 'train.csv'}")
    print(f"  val.csv   -> {output_dir / 'val.csv'}")
    print(f"  test.csv  -> {output_dir / 'test.csv'}")


if __name__ == "__main__":
    split_dataset()
