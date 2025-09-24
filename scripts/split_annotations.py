import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse


def split_annotations(
        input_csv="conf/dataset/annotations.csv", 
        output_dir="conf/dataset", 
        test_size=0.25,  #← デフォルトは 0.2 (＝8:2 に分割
        random_state=42,
        filename_col="Filename", 
        target_col="Exposure"
    ):
    """
    CSVを train/test に分割して保存する。
    - input_csv: 元のCSVファイル
    - output_dir: 保存先ディレクトリ
    - test_size: テストデータの割合 (例: 0.2 → 20%)
    - random_state: 再現性のための乱数シード
    - filename_col, target_col: CSVの列名
    """

    # CSVを読み込む
    df = pd.read_csv(input_csv)
    print(f"読み込み：{len(df)}件")

    # 列が存在するかチェック
    if filename_col not in df.columns or target_col not in df.columns:
        raise KeyError(f"CSVに {filename_col}, {target_col} カラムが必要です。")

    # 拡張子 .jpg を追加 
    # 文字列型に変換してから結合することで、予期せぬエラーを防ぎます
    df[filename_col] = df[filename_col].astype(str) + ".jpg"

    # 学習用・テスト用に分割
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    #保存する列のみ残す
    train_df = train_df[[filename_col, target_col]]
    test_df = test_df[[filename_col, target_col]]
    # 保存先ディレクトリを作成
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    # index=False を指定しないと、余計な列が保存されてしまうので注意
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[INFO] データ分割完了")
    print(f"  訓練データ: {len(train_df)} サンプル → {train_path}")
    print(f"  テストデータ: {len(test_df)} サンプル → {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSVを train/test に分割するスクリプト")
    parser.add_argument("--input_csv", type=str, required=True, help="元のアノテーションCSVのパス")
    parser.add_argument("--output_dir", type=str, default="conf/dataset", help="保存先ディレクトリ")
    parser.add_argument("--test_size", type=float, default=0.2, help="テストデータの割合 (0~1)")
    parser.add_argument("--random_state", type=int, default=42, help="乱数シード")
    args = parser.parse_args()

    split_annotations(args.input_csv, args.output_dir, args.test_size, args.random_state)
