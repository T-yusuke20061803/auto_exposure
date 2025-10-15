import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm


def split_dataset(
    dataset_name = "HDR_subdataset"  # データセット名
    input_dir = f"conf/dataset/{dataset_name}"
    output_dir = f"conf/dataset/{dataset_name}_split"
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    seed=42
):
    """
    画像データセットを train/test に分割してコピーする。

    Args:
        dataset_name (str): データセット名（例: "HDR_subdataset"）
        input_dir (str): 元の画像ディレクトリ
        output_dir (str): 分割後の出力ルートディレクトリ
        train_size (float): 訓練データの割合
        val_size (float): 検証データの割合
        test_size (float): テストデータの割合
        seed (int): 再現性確保のための乱数シード
    """

    # パス設定
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # 出力先ディレクトリ構造を作成
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    # 入力画像の存在確認
    if not input_dir.exists():
        raise FileNotFoundError(f"入力ディレクトリが存在しません: {input_dir}")

    # 出力フォルダ作成 
    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 対象画像を取得（jpg, png, jpeg対応）
    image_paths = sorted([
        p for p in input_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if len(image_paths) == 0:
        raise FileNotFoundError(f"該当する画像ファイル無し: {input_dir}")

    print(f"総画像数: {len(image_paths)}枚検出")

    # データを分割
    train_val_imgs, test_imgs = train_test_split(
        image_paths, 
        test_size=test_size, 
        random_state=seed, 
        shuffle=True
    )
    val_ratio_adjusted = val_size / (1 - test_size)
    train_imgs, val_imgs = train_test_split(
        train_val_imgs,
        test_size=val_ratio_adjusted,
        random_state=seed,
        shuffle=True
    )

    # ファイルをコピー(用量制約がある場合には、'shutil.move` に変更可能）
    def copy_images(img_list, dest_dir, label):
        print(f"\n[{label}] 画像をコピー中 ({len(img_list)} 枚)")
        for p in tqdm(img_list, desc=f"{label}"):
            shutil.copy(p, dest_dir / p.name)


    copy_images(train_imgs, train_dir, "train")
    copy_images(val_imgs, val_dir, "val")
    copy_images(test_imgs, test_dir, "test")

    print("\nデータ分割完了")
    print(f"  訓練画像: {len(train_imgs)} → {train_dir}")
    print(f"  テスト画像: {len(val_imgs)} → {val_dir}")
    print(f"  テスト画像: {len(test_imgs)} → {test_dir}")

if __name__ == "__main__":
    if __name__ == "__main__":
    split_dataset()