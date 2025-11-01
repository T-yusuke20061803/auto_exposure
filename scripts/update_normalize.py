import torch
import imageio.v3 as iio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 設定 
DATASET_ROOT = Path("conf/dataset/HDR+burst/processed_2048px_linear_exr")  # EXR画像フォルダ
BATCH_SIZE = 8
NUM_WORKERS = 0

# データセットクラス (.exr対応)
class EXRDataset(Dataset):
    """
    HDR (.exr) 用データセットクラス
    - imageioでEXRをfloat32として読み込み
    - CHWに変換しtorch.Tensorとして返す
    """
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.image_paths = sorted(self.root_dir.rglob("*.exr"))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"有効な画像(.exr)無し: {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = iio.imread(path).astype(np.float32)
            # shapeが(H, W, C)前提、C=3の場合のみ処理
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            elif img.shape[2] > 3:
                img = img[:, :, :3]
            img = torch.from_numpy(img).permute(2, 0, 1)
        except Exception as e:
            print(f"読み込み失敗: {path} ({e})")
            img = torch.zeros((3, 512, 512), dtype=torch.float32)
        return img

# 平均・標準偏差を計算
def calculate_mean_std(root_dir, batch_size, num_workers):
    dataset = EXRDataset(root_dir)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    n_pixels = 0
    mean = torch.zeros(3)
    sum_sq = torch.zeros(3)

    print(f"\n{len(dataset)}枚の画像から平均・標準偏差を計算中...\n")

    for images in tqdm(loader):
        n_pixels += images.shape[0] * images.shape[2] * images.shape[3]
        mean += images.sum(dim=(0, 2, 3))
        sum_sq += (images ** 2).sum(dim=(0, 2, 3))

    mean /= n_pixels
    var = (sum_sq / n_pixels) - (mean ** 2)
    std = torch.sqrt(var)

    mean_list = [round(m.item(), 4) for m in mean]
    std_list = [round(s.item(), 4) for s in std]

    print("\n=== 結果 ===")
    print(f"mean: {mean_list}")
    print(f" std: {std_list}")

    return mean_list, std_list

def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"指定されたフォルダが存在しません: {DATASET_ROOT}")

    calculate_mean_std(DATASET_ROOT, BATCH_SIZE, NUM_WORKERS)

if __name__ == "__main__":
    main()
