import yaml
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
import argparse
from PIL import Image
import os

class FlatImageDataset(Dataset):
    """
    クラス別のサブフォルダがない、フラットな画像フォルダ用のデータセットクラス。
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_exts = [".jpg", ".jpeg", ".png"]
        
        self.image_paths = sorted([
            p for p in self.root_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in self.image_exts
        ])

        if len(self.image_paths) == 0:
            print(f"有効な画像が見つかりません: {self.root_dir} (拡張子={self.image_exts})")
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f" 画像読み込み失敗: {img_path} ({e})")
            # 破損画像対策: ダミー画像で代替
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, 0

# 設定
dataset_root = Path("conf/dataset/HDR+burst/20171106/results_20171023")  # データセットを変更する場合ここ
config_path = Path("conf/config.yaml")               # 更新対象のconfig.yamlパス
batch_size=16
num_workers=0

def calculate_mean_std(dataset_root, batch_size, num_workers):
     # 画像をTensor化（正規化前）
    transform = transforms.Compose([
        # 1. 画像の短辺を256ピクセルにリサイズ
        transforms.Resize(256),
        # 2. 画像の中央部分を224x224ピクセルで切り抜く
        transforms.CenterCrop(224),
        transforms.ToTensor(), #0~1にスケール

    ])
    dataset = FlatImageDataset(dataset_root, transform=transform)
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, shuffle= False)

    #mean: 各チャンネル（R,G,B）ごとの合計値を保持。
    #sum_sq: 各チャンネルごとのピクセル値の2乗の合計を保持。
    # n_pixels: 全ピクセルの数をカウント。

    n_pixels = 0
    mean = torch.zeros(3)
    sum_sq = torch.zeros(3)

    print(f"\n{len(dataset)}枚の画像から平均・標準偏差を計算中\n")

    for images, _ in tqdm(loader):
        # images の形状: (batch, channels, height, width) 
        # ピクセル数を加算
        n_pixels += images.shape[0] * images.shape[2] * images.shape[3]
        # 全ピクセル値の合計をチャンネルごとに加算 (平均計算用)
        mean += images.sum(dim=(0, 2, 3))
        # 全ピクセル値の2乗の合計を加算 (標準偏差計算用)
        sum_sq += (images ** 2).sum(dim=(0, 2, 3))

    # --- 最終的な計算 ---
    mean /= n_pixels
    # Var = E[X^2] - (E[X])^2
    var = (sum_sq / n_pixels) - (mean ** 2)
    std = torch.sqrt(var)

    mean_list = [round(m.item(),4) for m in mean]
    std_list = [round(s.item(),4) for s in std]

    print(f"計算結果\n")
    print(f"mean:{mean_list}\n")
    print(f" std:{std_list}\n")

    return mean_list, std_list

def main():
    if not dataset_root.exists():
        raise FileNotFoundError(f" 指定されたデータセットフォルダが見つかりません: {dataset_root}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml が見つかりません: {config_path}")

    calculate_mean_std(dataset_root, batch_size, num_workers)


if __name__ == "__main__":
    main()


