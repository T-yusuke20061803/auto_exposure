import torch
import imageio.v3 as iio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import rawpy
import pandas as pd
from src.dataset import AnnotatedDatasetFolder, LogTransform, collate_fn_skip_none, imageio_loader

# 設定 
DATASET_ROOT = Path("conf/dataset/HDR+burst/processed_1024px_exr")  # EXR画像フォルダ
#processed_2048px_linear_exr→20171106/results_20171023
# 訓練データのファイルリストが書かれたCSV
TRAIN_CSV_PATH = Path("conf/dataset/HDR+burst_split/train.csv")
BATCH_SIZE = 32
NUM_WORKERS = 4


def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# 平均・標準偏差を計算
def calculate_mean_std(csv_path, root_dir, batch_size, num_workers):
    transform = LogTransform()
    dataset = AnnotatedDatasetFolder(
        root = root_dir,
        csv_file = csv_path, 
        loader = imageio_loader,
        transform = transform
    )
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        num_workers=num_workers, 
                        shuffle=False,
                        collate_fn=collate_fn_skip_none
                        )

    n_pixels = 0
    mean = torch.zeros(3)
    sum_sq = torch.zeros(3)

    print(f"\n{len(dataset)}枚の画像から (0-65535 -> Log変換後)平均・標準偏差を計算中...\n")

    for linear_images in tqdm(loader):
        if linear_images is None: continue

        # images は Log変換済み (約 0.0 〜 16.0 の範囲)
        images = linear_images[0]

        n_pixels += images.shape[0] * images.shape[2] * images.shape[3]
        mean += images.sum(dim=(0, 2, 3))
        sum_sq += (images ** 2).sum(dim=(0, 2, 3))

    if n_pixels == 0:
        print("エラー: 有効なピクセルがありませんでした。")
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    mean /= n_pixels
    var = (sum_sq / n_pixels) - (mean ** 2)
    var = torch.clamp(var, min=1e-6)# 非常に稀な計算誤差（負の値）を防ぐ
    std = torch.sqrt(var)

    mean_list = [round(m.item(), 4) for m in mean]
    std_list = [round(s.item(), 4) for s in std]

    print("\n=== 結果 ===")
    print(f"mean: {mean_list}")
    print(f" std: {std_list}")

    return mean_list, std_list

def main():
    if not TRAIN_CSV_PATH.exists(): # ★ 修正3: DATASET_ROOT -> TRAIN_CSV_PATH
        raise FileNotFoundError(f"訓練CSVが見つかりません: {TRAIN_CSV_PATH}")
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"指定されたフォルダが存在しません: {DATASET_ROOT}")

    calculate_mean_std(TRAIN_CSV_PATH, DATASET_ROOT, BATCH_SIZE, NUM_WORKERS)

if __name__ == "__main__":
    main()
