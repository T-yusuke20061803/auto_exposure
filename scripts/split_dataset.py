import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm


def split_dataset(
    input_dir="conf/dataset/HDR_subdataset",
    output_dir="conf/dataset/HDR_subdataset_split",
    test_size=0.2,
    random_state=42,
):
    """
    画像データセットを train/test に分割してコピーする。

    Args:
        input_dir (str): 元の画像ディレクトリ
        output_dir (str): train/test を出力するルートディレクトリ
        test_size (float): テストデータの割合 (0~1)
        random_state (int): 再現性のためのシード
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    # 入力画像の存在確認
    if not input_dir.exists():
        raise FileNotFoundError(f"入力ディレクトリが存在しません: {input_dir}")

    # 出力ディレクトリを作成
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # 対象画像を取得（jpg, png, jpeg対応）
    image_paths = sorted([
        p for p in input_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if len(image_paths) == 0:
        raise FileNotFoundError(f"該当する画像無し: {input_dir}")

    print(f"総画像数: {len(image_paths)}枚検出")

    # データを分割
    train_imgs, test_imgs = train_test_split(
        image_paths,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # ファイルをコピー
    print(f"\n訓練用画像をコピー中 ({len(train_imgs)}枚)")
    for p in tqdm(train_imgs):
        shutil.copy(p, train_dir / p.name)#コピーではなく移動（容量制約）にする場合は　shutil.move(p, train_dir / p.name)

    print(f"\nテスト用画像をコピー中 ({len(test_imgs)}枚)")
    for p in tqdm(test_imgs):
        shutil.copy(p, test_dir / p.name)

    print("\nデータ分割完了")
    print(f"  訓練画像: {len(train_imgs)} → {train_dir}")
    print(f"  テスト画像: {len(test_imgs)} → {test_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="画像フォルダを train/test に分割するスクリプト")
    parser.add_argument("--input_dir", type=str, default="conf/dataset/HDR_subdataset", help="元の画像フォルダパス")
    parser.add_argument("--output_dir", type=str, default="conf/dataset/HDR_subdataset_split", help="分割後の保存フォルダ")
    parser.add_argument("--test_size", type=float, default=0.2, help="テストデータの割合 (例: 0.2)")
    parser.add_argument("--random_state", type=int, default=42, help="乱数シード")

    args = parser.parse_args()
    split_dataset(args.input_dir, args.output_dir, args.test_size, args.random_state)
