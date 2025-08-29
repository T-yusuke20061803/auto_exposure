import os
import pandas as pd
import torch
from PIL import Image
import torch.utils.data as torchdata

class AnnotatedDatasetFolder(torchdata.Dataset):
    """
    画像フォルダパスと、アノテーション情報(CSVパス or データフレーム)を受け取りデータセットを作成するクラス
    """
    def __init__(self, root, loader, transform=None, csv_file=None, dataframe=None):
        """
        Args:
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
        
        self.samples = []
        for _, row in dataframe.iterrows():
            filename = row['Filename']
            target = row['Exposure']
            path = os.path.join(self.root, filename)
            
            if os.path.exists(path):
                self.samples.append((path, float(target), filename))
                
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
    
def pil_loader(path): # <-- タイプミスを修正 (loafer -> loader)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None, None
    return torch.utils.data.dataloader.default_collate(batch)