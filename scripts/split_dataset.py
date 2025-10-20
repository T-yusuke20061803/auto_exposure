import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm


def split_dataset(
    dataset_name="HDR+burst", #データセットを変更する場合は、ここを変更する
    annotations_csv="conf/dataset/annotations.csv",
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    seed=42
):
    """
    画像データセットを train/val/test に分割し、
    対応するラベルCSV (train_labels.csv / val_labels.csv / test_labels.csv) を生成する。
    """

    # ディレクトリ設定
    input_dir = Path(f"conf/dataset/{dataset_name}")
    output_dir = Path(f"conf/dataset/{dataset_name}_split")
    train_dir, val_dir, test_dir = [output_dir / x for x in ("train", "val", "test")]

    # 存在確認
    if not input_dir.exists():
        raise FileNotFoundError(f"入力ディレクトリ：無し: {input_dir}")
    if not Path(annotations_csv).exists():
        raise FileNotFoundError(f"CSVファイル：無し: {annotations_csv}")

    # 出力フォルダ作成
    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 画像一覧を取得
    image_paths = sorted([p for p in input_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if not image_paths:
        raise FileNotFoundError(f"該当画像ファイル無し: {input_dir}")

    print(f"総画像数: {len(image_paths)} 枚検出")

    # データ分割
    train_val_imgs, test_imgs = train_test_split(image_paths, test_size=test_size, random_state=seed, shuffle=True)
    val_ratio_adj = val_size / (1 - test_size)
    train_imgs, val_imgs = train_test_split(train_val_imgs, test_size=val_ratio_adj, random_state=seed, shuffle=True)

    # 分割結果をコピー
    def copy_images(img_list, dest_dir, label):
        print(f"\n[{label}] 画像をコピー中 ({len(img_list)} 枚)")
        for p in tqdm(img_list, desc=label):
            shutil.copy(p, dest_dir / p.name)

    copy_images(train_imgs, train_dir, "train")
    copy_images(val_imgs, val_dir, "val")
    copy_images(test_imgs, test_dir, "test")

    # === CSV処理 ===
    df = pd.read_csv(annotations_csv)
    df["Filename"] = df["Filename"].astype(str) + ".jpg"  # 拡張子を統一

    # ファイル名でフィルタリング
    def filter_and_save(df, img_list, path):
        names = [p.name for p in img_list]
        subset = df[df["Filename"].isin(names)]
        subset.to_csv(path, index=False)
        print(f"  {path.name}: {len(subset)} 件保存")

    filter_and_save(df, train_imgs, output_dir / "train_labels.csv")
    filter_and_save(df, val_imgs, output_dir / "val_labels.csv")
    filter_and_save(df, test_imgs, output_dir / "test_labels.csv")

    print("\nデータセット分割および対応CSV作成:完了。")
    print(f"  訓練: {len(train_imgs)}枚 / 検証: {len(val_imgs)}枚 / テスト: {len(test_imgs)}枚")
    print(f"  出力先: {output_dir}")


if __name__ == "__main__":
    split_dataset()
