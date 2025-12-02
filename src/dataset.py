import os
import pandas as pd
import torch
from PIL import Image
import torch.utils.data as torchdata
from tqdm import tqdm
import imageio 
import numpy as np 
from pathlib import Path
import rawpy
import torch

_loader_print_count = 0

class LogTransform(object):
    """
    各入力の画素値に対してlog2(T+1.0)を適応する 
    理由：画素値が0から10,000に集中しているため対数を用いることで、小さいな（0～10000）に集中している部分を強調するため
         対数処理を取ることがで、0～16程度の扱いやすい範囲に返還する
    """
    def __call__(self, tensor):

        # もし入力が 0-1 なら 65535倍に戻す
        if tensor.max() <= 1.0:
             return torch.log2(tensor * 65535.0 + 1.0)
        # 0～65535ならそのまま
        return torch.log2(tensor + 1.0)


class AnnotatedDatasetFolder(torchdata.Dataset):
    #画像フォルダパスと、アノテーション情報(CSVパス or データフレーム)を受け取りデータセットを作成するクラス
    def __init__(self, root, loader, transform=None, csv_file=None, dataframe=None):
        """
            root (str): 画像フォルダのルートパス
            dataframe (pd.DataFrame): "Filename", "Exposure" を含む DataFrame
            loader (callable, optional): 画像を読み込む関数 (デフォルト: PIL)
            transform (callable, optional): 前処理 (torchvision.transforms など)
        """
        self.root = root
        self.loader = loader
        self.transform = transform
        
        if dataframe is None:
            if csv_file is None:
                raise ValueError("annotation_fileまたはdataframeのどちらかが必要です。")
            try:
                #annotation_fileは実行場所からの相対パス、または絶対パス
                dataframe = pd.read_csv(csv_file)
            except FileNotFoundError:
                raise RuntimeError(f"アノテーションファイルが見つかりません: {csv_file}")
            
        if "filepath" not in dataframe.columns:#画像の場所
            raise ValueError("CSVには 'filepath' 列が必要です。")
        if "Exposure" not in dataframe.columns:#正解ラベルの場所
            raise ValueError("CSVには 'Exposure' 列が必要です。")
        
        self.samples = []
        print(f"{csv_file or 'DataFrame'} からサンプルを読み込み中 (基準パス: {root})...")
        missing_files = 0
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Loading samples"):
            relative_path_dng_str = row['filepath'] # .csv に書かれている .dng のパス
            target = row['Exposure']
            filename = row.get('Filename', relative_path_dng_str) # Filename があれば使う

            relative_path_exr_str = str(Path(relative_path_dng_str).with_suffix('.exr')) #リサイズしない場合は、ここをコメントアウト
            # .exr のフルパスを構築
            path = os.path.join(self.root, relative_path_exr_str)#リサイズしない場合：relative_path_exr_str -> relative_path_dng_str
            
            if os.path.exists(path):
                try:
                    # リストに追加 (場所, 正解ラベル, ファイル名ID)
                    self.samples.append((path, float(target), filename))
                except (ValueError, TypeError):
                     print(f"警告: {path} の Exposure '{target}' を float に変換不可。スキップします。")
                     missing_files += 1
            else:
                missing_files += 1
                if 0 < missing_files < 10: 
                    print(f"警告: パスが無し( .exr に変換): {path}:{missing_files} 件") # .exr に変換 -> .dng

        if not self.samples:
             raise RuntimeError("有効なサンプル無し")
        print(f"{len(self.samples)} 件の有効なサンプルを読み込み")
    def __getitem__(self, index):
        path, target, filename = self.samples[index]
        try:
            sample = self.loader(path)
        except Exception as e:
            print(f"エラー: 画像の読み込み失敗{path}, {e}")
            return None
        #前処理適応
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
    try:
        # .exr を float32 の numpy 配列 (H, W, C) として読み込む　値は 0〜65535 のまま
        img_float_numpy = imageio.v3.imread(path) 
        # NumPy (縦, 横, 色) -> PyTorch Tensor (色, 縦, 横) に変換　この順でないと処理できない
        tensor = torch.from_numpy(img_float_numpy.astype(np.float32)).permute(2, 0, 1) 
        return tensor 
    
    except Exception as e:
        print(f"imageioでの画像読み込みエラー: {path}, エラー: {e}")
        raise

def dng_loader(path):
    # .dng を rawpy で読み込み、線形 float32 テンソルで返す 
    try:
        with rawpy.imread(str(path)) as raw:
        # 線形16bitデータを取得
            rgb_16bit = raw.postprocess(
                use_camera_wb=True, output_bps=16, no_auto_bright=True, gamma=(1, 1) 
            )
        # float32 [0, 1] (clipなし) に変換
        rgb_linear_float = rgb_16bit.astype(np.float32) / 65535.0
        # rgb_linear_float = np.clip(rgb_linear_float, 0.0, 1.0) # クリップはNormalize前にはしない方が良いかも
        
        # NumPy (H, W, C) -> PyTorch Tensor (C, H, W)
        tensor = torch.from_numpy(rgb_linear_float).permute(2, 0, 1) 
        return tensor
    except Exception as e:
        print(f"rawpyでの画像読み込みエラー: {path}, エラー: {e}")
        raise

def collate_fn_skip_none(batch):
    # 読み込み失敗(None)のデータがあったら、リストから除外
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None, None, None
    return torch.utils.data.dataloader.default_collate(batch)