import torch
import imageio.v3 as iio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import rawpy
import pandas as pd

# 設定 
DATASET_ROOT = Path("conf/dataset/HDR+burst/processed_1024px_exr")  # EXR画像フォルダ
#processed_2048px_linear_exr→20171106/results_20171023
# 訓練データのファイルリストが書かれたCSV
TRAIN_CSV_PATH = Path("conf/dataset/HDR+burst_split/train.csv")
BATCH_SIZE = 32
NUM_WORKERS = 4

# データセットクラス (.exr対応)
class EXRDataset(Dataset):
    """
    HDR (.exr) 用データセットクラス
    - imageioでEXRをfloat32として読み込み
    - CHWに変換しtorch.Tensorとして返す
    """
    def __init__(self, csv_file, root_dir):
        self.root_dir = Path(root_dir)
        try:
            # train.csv を読み込む
            dataframe = pd.read_csv(csv_file)
        except FileNotFoundError:
            raise RuntimeError(f"アノテーションファイルが見つかりません: {csv_file}")
        
        self.image_paths = []
        print(f"{csv_file} から訓練サンプルを読み込み中 (基準パス: {root_dir})...")
        
        # CSVをループし、.dng パスを .exr パスに変換してリストに追加
        for _, row in dataframe.iterrows():
            relative_path_dng_str = row['filepath']
            # .dng を .exr に置換
            relative_path_exr_str = str(Path(relative_path_dng_str).with_suffix('.exr'))
            # ルートパスと結合
            path = self.root_dir / relative_path_exr_str
            
            if path.exists():
                self.image_paths.append(path)
            else:
                print(f"[WARN] ファイルが見つかりません: {path}")
    
        #self.image_paths = sorted(self.root_dir.rglob("*.exr"))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"有効な画像(.exr)無し: {self.root_dir}")
        
        print(f"--- {self.root_dir} 以下で {len(self.image_paths)}件の画像パスを検出 ---")
        # 最初の5件だけ表示
        for path in self.image_paths[:5]:
            print(f"  {path}")
        if len(self.image_paths) > 5:
            print(f"  ...他 {len(self.image_paths) - 5} 件")
        print("---------------------------------")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = iio.v3.imread(path).astype(np.float32)
            # shapeが(H, W, C)前提、C=3の場合のみ処理
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            elif img.shape[2] > 3:
                img = img[:, :, :3]
            img = torch.from_numpy(img).permute(2, 0, 1)
        except Exception as e:
            print(f"読み込み失敗: {path} ({e})")
            return None
            #img = torch.zeros((3, 512, 512), dtype=torch.float32)
        return img

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# 平均・標準偏差を計算
def calculate_mean_std(csv_path, root_dir, batch_size, num_workers):
    dataset = EXRDataset(csv_path, root_dir)
    loader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        num_workers=num_workers, 
                        shuffle=False,
                        collate_fn=collate_fn_skip_none
                        )

    n_pixels = 0
    mean = torch.zeros(3)
    sum_sq = torch.zeros(3)

    print(f"\n{len(dataset)}枚の画像から平均・標準偏差を計算中...\n")

    for linear_images in tqdm(loader):
        if linear_images is None: continue

        images = torch.log2(linear_images + 1.0) #計算の前にLogを入れた
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
