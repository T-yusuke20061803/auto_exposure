import yaml
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
# import argparse # 未使用
from PIL import Image
import rawpy
import numpy as np
import os
import imageio # .exr 読み込みに必要 (このファイルでは使われないが、他で使っている)

# .dng を読み込むためのデータセットクラス 
class FlatDNGDataset(Dataset): 
    def __init__(self, root_dir, transform=None): # ★ transform を受け取る ★
        self.root_dir = Path(root_dir)
        self.transform = transform # ★ transform を保存 ★
        self.image_exts = [".dng"] 

        print(f"検索中: {self.root_dir} (拡張子={self.image_exts})")
        #  split_dataset.py と同じ merged.dng のみ対象に ★
        self.image_paths = sorted([
            p for p in self.root_dir.rglob("merged.dng") 
            if p.is_file()
        ])
        if len(self.image_paths) == 0:
            print(f"有効な画像 (merged.dng) が見つかりません: {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # dng_loader と同じロジック 
            with rawpy.imread(str(img_path)) as raw:
                rgb_16bit = raw.postprocess(
                    use_camera_wb=True, output_bps=16, no_auto_bright=True, gamma=(1, 1)
                )
            rgb_linear_float = rgb_16bit.astype(np.float32) / 65535.0
            rgb_linear_float = np.clip(rgb_linear_float, 0.0, 1.0) 
            tensor = torch.from_numpy(rgb_linear_float).permute(2, 0, 1)
            
            #  統計計算のためにリサイズ/クロップを適用 
            if self.transform:
                tensor = self.transform(tensor) 
            
            return tensor, 0
            
        except Exception as e:
            print(f" 画像読み込み失敗: {img_path} ({e})")
            return None 

def collate_fn_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None, None
    return torch.utils.data.dataloader.default_collate(batch)

# ★元の .dng フォルダを指す 
dataset_root = Path("conf/dataset/HDR+burst/20171106/results_20171023") 
config_path = Path("conf/config.yaml")
batch_size = 16 #  32 -> 16 に変更 (フル解像度読み込みはメモリを食うため)
num_workers = 4 


# mean/std 計算関数
def calculate_mean_std(dataset_root, batch_size, num_workers):
    
    # 統計計算用にリサイズ/クロップ処理を定義
    # (学習時の val_transforms と合わせるのが一般的)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    
    # transform を渡し、collate_fn を指定
    dataset = FlatDNGDataset(dataset_root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn_skip_none)

    n_pixels = 0
    mean = torch.zeros(3)
    sum_sq = torch.zeros(3)

    print(f"\n{len(dataset)}枚の .dng 画像 (クロップ後) から平均・標準偏差を計算中...\n")

    for images, _ in tqdm(loader): 
        if images is None: continue 
        
        n_pixels += images.shape[0] * images.shape[2] * images.shape[3]
        mean += images.sum(dim=(0, 2, 3))
        sum_sq += (images ** 2).sum(dim=(0, 2, 3))

    mean /= n_pixels
    var = (sum_sq / n_pixels) - (mean ** 2)
    var = torch.clamp(var, min=1e-6)
    std = torch.sqrt(var)

    mean_list = [round(m.item(), 4) for m in mean]
    std_list = [round(s.item(), 4) for s in std]

    print("\n=== 結果 ===")
    print(f"mean: {mean_list}")
    print(f" std: {std_list}")

    return mean_list, std_list

def main():
    if not dataset_root.exists():
        raise FileNotFoundError(f"指定されたフォルダが存在しません: {dataset_root}")
    calculate_mean_std(dataset_root, batch_size, num_workers)

if __name__ == "__main__":
    main()