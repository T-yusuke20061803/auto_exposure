import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np
import os

csv_path = Path("conf/dataset/HDR+burst_split/test.csv")
image_root = Path("conf/dataset/HDR+burst/processed_1024px_exr")
output_path = Path("outputs/dataset_verification.png")
num_samples = 5


def normalize_for_display(img):
    """画像を強制的に0-1の範囲に正規化して見やすくする"""
    img = img.astype(np.float32)
    img = np.nan_to_num(img) # NaN除去

    # 完全にデータがない(真っ黒)場合
    if img.max() <= 0:
        return img
    # ★修正ポイント: 画像ごとの最大値(v_max)を使わず、単純なトーンマップのみ行う
    # Reinhard Tone Mapping: x / (1 + x)
    # 明るい場所は1.0に近づき、暗い場所は0.0に近いままになる
    img_mapped = img / (img + 1.0)
    
    # ガンマ補正 (sRGB表示用)
    img_gamma = np.power(img_mapped, 1.0/2.2)
    
    # 0-1クリップ
    img_final = np.clip(img_gamma, 0, 1)
    
    return img_final

def main():
    if not csv_path.exists():
        print(f"エラー: CSVが見つかりません {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    samples = df.sample(n=num_samples)
    print(f"=== 画像データの強制可視化 ({num_samples}枚) ===")

    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
    if num_samples == 1: 
        axes = [axes]

    for i, (_, row) in enumerate(samples.iterrows()):
        filename = row['Filename']
        ev_val = row['Exposure']
        print(f"\nProcessing: {filename}")

        # フォルダ検索
        found_dirs = list(image_root.rglob(f"{filename}*"))
        target_file = None
        if found_dirs:
            # 見つかったパスがディレクトリなら、その中の画像ファイルを探す
            base_path = found_dirs[0]
            if base_path.is_dir():
                # ディレクトリ内のファイルを検索 (merged.exr を優先、なければ最初のファイル)
                files_in_dir = list(base_path.iterdir())
                exr_files = list(base_path.glob("*.exr"))

                if exr_files:
                    target_file = exr_files[0]
                elif files_in_dir:
                    target_file = files_in_dir[0]
            else:
                target_file = base_path

        if target_file is None:
            print("  -> 画像ファイルが見つかりませんでした (スキップ)")
            continue
        print(f"  -> 読み込み対象: {target_file.name}")
        try:
            # 画像読み込み
            img = iio.imread(target_file)

            # 強制可視化処理
            img_disp = normalize_for_display(img)

            # 表示
            ax = axes[i]
            ax.imshow(img_disp)

            # ラベル表示
            status = "Bright(+)" if ev_val > 0 else "Dark(-)"
            color = "red" if ev_val > 0 else "blue"
            ax.set_title(f"CSV: {ev_val}\n{status}", color=color, fontsize=12, fontweight='bold')
            ax.axis('off')
        except Exception as e:
            print(f"  読み込みエラー: {e}")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"\n確認画像を保存しました: {output_path.resolve()}")
    print("生成された画像を確認してください。青字(Dark)が暗く、赤字(Bright)が明るければ、データセットは正常です。")



if __name__ == "__main__":

    main()