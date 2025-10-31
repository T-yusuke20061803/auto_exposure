import os
import pandas as pd
import torch
from PIL import Image
import torch.utils.data as torchdata
from tqdm import tqdm
import imageio 
import numpy as np 
from pathlib import Path

class AnnotatedDatasetFolder(torchdata.Dataset):
    """
    画像フォルダパスと、アノテーション情報(CSVパス or データフレーム)を受け取りデータセットを作成するクラス
    """
    def __init__(self, root, loader, transform=None, csv_file=None, dataframe=None):
        """
            root (str): 画像フォルダのルートパス
            dataframe (pd.DataFrame): "Filename", "Exposure" を含む DataFrame
            loader (callable, optional): 画像を読み込む関数 (デフォルト: PIL)
            transform (callable, optional): 前処理 (torchvision.transforms など)
            extensions (list, optional): 許可する拡張子 (例: [".jpg", ".png", ".jpeg"])
        """
        self.root = root
        self.loader = loader
        self.transform = transform
        
        if dataframe is None:
            if csv_file is None:
                raise ValueError("annotation_fileまたはdataframeのどちらかが必要です。")
            try:
                #annotation_fileは実行場所からの相対パス、または絶対パスです
                dataframe = pd.read_csv(csv_file)
            except FileNotFoundError:
                raise RuntimeError(f"アノテーションファイルが見つかりません: {csv_file}")
            
        if "filepath" not in dataframe.columns:
            raise ValueError("CSVには 'filepath' 列が必要です。")
        if "Exposure" not in dataframe.columns:
            raise ValueError("CSVには 'Exposure' 列が必要です。")
        
        self.samples = []
        missing_files = 0
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Loading samples"):
            relative_path_dng_str = row['filepath'] # .csv に書かれている .dng のパス
            target = row['Exposure']
            filename = row.get('Filename', relative_path_dng_str) # Filename があれば使う

            relative_path_exr_str = str(Path(relative_path_dng_str).with_suffix('.exr'))
            # .exr のフルパスを構築
            path = os.path.join(self.root, relative_path_exr_str)
            
            if os.path.exists(path):
                try:
                    self.samples.append((path, float(target), filename))
                except (ValueError, TypeError):
                     print(f"警告: {path} の Exposure '{target}' を float に変換不可。スキップします。")
                     missing_files += 1
            else:
                missing_files += 1
                if missing_files < 10: # ログが溢れないように
                    print(f"警告: パスが見つかりません ( .exr に変換): {path}") 
        
        if missing_files > 0:
             print(f"警告: {missing_files} 件のファイルが見つかりませんでした。")
        if not self.samples:
             raise RuntimeError("有効なサンプルが見つかりませんでした。")
        print(f"{len(self.samples)} 件の有効なサンプルを読み込みました。")
    def __getitem__(self, index):
        path, target, filename = self.samples[index]
        try:
            sample = self.loader(path)
        except Exception as e:
            print(f"エラー: 画像の読み込み失敗{path}, {e}")
            return None
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        target = torch.tensor([target], dtype=torch.float32)
        return sample, target, filename

    def __len__(self):
        return len(self.samples)
    
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
# .exr を読み込むローダーを新設
def imageio_loader(path):
    """ .exr や .tiff などのHDR形式を float32 テンソルで読み込むローダー """
    try:
        # .exr を float32 の numpy 配列 (H, W, C) として読み込む
        img_float_numpy = imageio.v3.imread(path) 
        
        # NumPy (H, W, C) -> PyTorch Tensor (C, H, W) に変換
        # [0, 1] 範囲の float32 を想定
        tensor = torch.from_numpy(img_float_numpy.astype(np.float32)).permute(2, 0, 1) 
        return tensor
    except Exception as e:
        print(f"imageioでの画像読み込みエラー: {path}, エラー: {e}")
        raise

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None, None
    return torch.utils.data.dataloader.default_collate(batch)