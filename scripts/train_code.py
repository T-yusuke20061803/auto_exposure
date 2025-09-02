# 訓練用コード
from pathlib import Path
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from PIL import Image
import torchinfo
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
import pandas as pd


from src.model import SimpleCNN, ResNet
from src.trainer import Trainer, LossEvaluator
from src.train_id import print_config, generate_train_id, is_same_config
from src.extension import ModelSaver, HistorySaver, HistoryLogger, IntervalTrigger, LearningCurvePlotter, MinValueTrigger
from src.util import set_random_seed
from src.dataset import AnnotatedDatasetFolder, pil_loader, collate_fn_skip_none


# === CSVから画像パスと補正量(EV)を読み込むデータセット ===
class EVRegressionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.samples = []
        with open(csv_file, newline='') as f:
            reader = csv.reader(f)
            next(reader)# ヘッダーをスキップ
            for row in reader:
                img_path, ev = row
                self.samples.append((img_path, float(ev)))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, ev = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([ev], dtype=torch.float32)

# ---- メイン処理 ----
@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    set_random_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 保存先を conf/dataset/results に変更
    output_dir = Path("outputs/train_reg")
    train_id = generate_train_id(cfg)
    history_path = output_dir / "history" / train_id
    history_path.mkdir(parents=True, exist_ok=True)

    print_config(cfg)
    cfg_path = history_path / 'config.yaml'
    if cfg_path.exists() and not is_same_config(cfg, OmegaConf.load(str(cfg_path))):
        raise ValueError("Train ID conflict with different config.")
    OmegaConf.save(cfg, str(cfg_path))

    # データ変換 (データ拡張＋正規化)
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(**cfg.dataset.train.transform.random_resized_crop),
        v2.RandomHorizontalFlip(**cfg.dataset.train.transform.random_horizontal_flip),
        v2.RandomVerticalFlip(**cfg.dataset.train.transform.random_vertical_flip),
        v2.RandomRotation(**cfg.dataset.train.transform.random_rotation),
        v2.ColorJitter(**cfg.dataset.train.transform.color_jitter), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(**cfg.dataset.train.transform.normalize),
    ])

    val_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(**cfg.dataset.train.transform.normalize),
    ])

     # --- データ分割 ---
    master_df = pd.read_csv(cfg.dataset.train.csv_file)
    train_df, val_df = train_test_split(
        master_df, 
        test_size=cfg.dataset.split.val_size,
        random_state=cfg.dataset.split.random_state
    )
    print(f"データ分割：訓練 {len(train_df)}, 検証 {len(val_df)} 件")

    # --- データセット ---
    train_set = AnnotatedDatasetFolder(cfg.dataset.train.root, dataframe=train_df, loader=pil_loader, transform=train_transforms)
    val_set   = AnnotatedDatasetFolder(cfg.dataset.train.root, dataframe=val_df, loader=pil_loader, transform=val_transforms)

    # データセット読み込み
    train_loader = DataLoader(train_set, shuffle=True, **cfg.dataloader)
    val_loader = DataLoader(val_set, shuffle=False, **cfg.dataloader)

    # モデル
    if cfg.model.name.lower() == "simplecnn":
        net = SimpleCNN(**cfg.model.params).to(device)
    elif cfg.model.name.lower() == "resnet":
        net = ResNet(**cfg.model.params).to(device)
    else:
        raise ValueError(f"未対応のモデルです: {cfg.model.name}")
    
    # サマリー表示
    sample_batch = next(iter(train_loader))
    if sample_batch[0] is not None:
        input_size = sample_batch[0][0].shape
        torchinfo.summary(net, input_size=(cfg.dataloader.batch_size, *input_size))


 # 損失関数、最適化手法、スケジューラ
    criterion = nn.MSELoss()
    # Optimizer 選択 
    if cfg.optimizer.name.lower() == "sgd":
        optimizer = optim.SGD(net.parameters(), **cfg.optimizer.params)
    elif cfg.optimizer.name.lower() == "adam":
        optimizer = optim.Adam(net.parameters(), **cfg.optimizer.params)
    else:
        raise ValueError(f"未対応のoptimizerです: {cfg.optimizer.name}")
    if cfg.lr_scheduler.name == "multi_step":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            [int(i * cfg.epoch) for i in cfg.lr_scheduler.params.milestones],
            gamma=cfg.lr_scheduler.params.gamma
        )
     # ReduceLROnPlateau（性能が停滞したらLR変更）Trainer側で毎エポックの検証lossを渡す必要がある
    elif cfg.lr_scheduler.name == "plateau":
         scheduler = optim.lr_scheduler.ReduceLROnPlateau(
             optimizer, **cfg.lr_scheduler.params)
    else:
        # スケジューラを使わない場合
        scheduler = None

    # 評価指標と拡張
    evaluators = [LossEvaluator(criterion, criterion_name="MSE")]
    extensions = [
            ModelSaver(directory=history_path, name=lambda x: "best_model.pth", trigger=MinValueTrigger(mode="validation", key="loss")),
            HistorySaver(directory=history_path, name=lambda x: "history.pth", trigger=IntervalTrigger(period=1)),
            HistoryLogger(trigger=IntervalTrigger(period=1), print_func=print),
            LearningCurvePlotter(directory=history_path, trigger=IntervalTrigger(period=1)),
        ]

    # 学習
    trainer = Trainer(
            net, 
            optimizer, 
            criterion,
            train_loader, 
            scheduler=scheduler,
            extensions=extensions,
            evaluators=evaluators,
            device=device
        )
    trainer.train(cfg.epoch, val_loader)

    model_output_dir = Path("./outputs/model")
    model_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), model_output_dir / "final_model.pth")


if __name__ == "__main__":
    main()
