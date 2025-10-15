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
        self.image_paths = sorted([
            p for p in self.root_dir.iterdir() 
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0 # ラベルは使わないのでダミーで0を返す

# 設定
dataset_root = Path("conf/dataset/HDR_subdataset/")  # データセットを変更する場合ここ
config_path = Path("conf/config.yaml")               # 更新対象のconfig.yamlパス
batch_size=32 
num_workers=2

def calculate_mean_std(dataset_root, batch_size, num_workrs):
     # 画像をTensor化（正規化前）
    transform = transforms.Compose([
        transforms.ToTensor(), #0~1にスケール

    ])
    dataset = FlatImageDataset(dataset_root, transform=transform)
    loader = DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, shuffle= False)

    mean = torch.zeros(3)
    std = torch.zeros(3)

    print(f"\n{len(dataset)}枚の画像から平均・標準偏差を計算中・・・\n")

    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, 3, -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(loader)
    std /= len(loader)

    mean_list = [round(m.item(),4) for m in mean]
    std_list = [round(m.item(),4) for m in std]

    print(f"計算結果\n")
    print(f"mean:{mean_list}\n")
    print(f"std:{std_list}\n")

    return mean_list, std_list

def update_config(config_path, mean, std):
    "config.yaml の normalize を自動更新"
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # normalize部分を安全に更新
    #train側
    if "dataset" in cfg and "train" in cfg["dataset"]:
        if "transform" in cfg["dataset"]["train"]:
            cfg["dataset"]["train"]["transform"]["normalize"] = {
                "mean": mean,
                "std": std
            }
    #test側
    if "dataset" in cfg and "test" in cfg["dataset"]:
        if "transform" in cfg["dataset"]["test"]:
            cfg["dataset"]["test"]["transform"]["normalize"] = {
                "mean": mean,
                "std": std
            }

    #上書き保存
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"{config_path}を更新\n")
    print(f"mean:{mean}\n")
    print(f"std :{std}\n")

def main():
    if not dataset_root.exists():
        raise FileNotFoundError(f" 指定されたデータセットフォルダが見つかりません: {dataset_root}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml が見つかりません: {config_path}")

    mean, std = calculate_mean_std(dataset_root, batch_size, num_workers)
    update_config(config_path, mean, std)


if __name__ == "__main__":
    main()


