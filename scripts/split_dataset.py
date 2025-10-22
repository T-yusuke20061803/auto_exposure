import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm





def split_dataset(
    dataset_name="HDR+burst",
    subset_name="20171106/results_20171023", #データセットを変更する場合は、ここを変更する
    annotations_csv="conf/dataset/annotations.csv",
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    seed=42

):

    # ディレクトリ設定
    input_dir = Path(f"conf/dataset/{dataset_name}/{subset_name}")
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

    image_exts = [".jpg", ".jpeg", ".png"]  # 必要に応じて .tiff, .dng を追加
    image_paths = sorted([
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in image_exts
    ])

    if not image_paths:
            raise FileNotFoundError(f"該当画像ファイルが見つかりません: {input_dir}")
    print(f"総画像数: {len(image_paths)} 枚検出({input_dir})")

    # データ分割
    train_val_imgs, test_imgs = train_test_split(
        image_paths, test_size=test_size, random_state=seed, shuffle=True
    )
    val_ratio_adj = val_size / (1 - test_size)
    train_imgs, val_imgs = train_test_split(
        train_val_imgs, test_size=val_ratio_adj, random_state=seed, shuffle=True
    )

    def make_df(img_list, split_name):
        return pd.DataFrame({
            "filepath": [str(p) for p in img_list],
            "split": [split_name] * len(img_list)
        })



    df_train = make_df(train_imgs, "train")
    df_val = make_df(val_imgs, "val")
    df_test = make_df(test_imgs, "test")

    df_train.to_csv(output_dir / "train.csv", index=False)
    df_val.to_csv(output_dir / "val.csv", index=False)
    df_test.to_csv(output_dir / "test.csv", index=False)



    print(f"\n=== データ分割結果 ===")
    print(f"train: {len(df_train)} 枚")
    print(f"val  : {len(df_val)} 枚")
    print(f"test : {len(df_test)} 枚")
    print(f"出力先: {output_dir}")


if __name__ == "__main__":

    split_dataset()



#学習用のデータを記録したパスを作る

#results_20171023の中にある画像を採用する（連射した画像を除いた綺麗な画像のみのフォルダ）

#ytomita@oit:~/auto_exposure$ ls conf/dataset/HDR+burst/20171106/bursts/ -l | wc -l でファイルの総数を調べるコマンド