# train_reg.py
from pathlib import Path
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
import torchvision
from torchvision.transforms import v2
from PIL import Image

import hydra
from omegaconf import DictConfig, OmegaConf

from src.dataset import AnnotatedDatasetFolder, collate_fn_skip_none
from src.model import SimpleCNN
from src.trainer import Trainer, LossEvaluator
from src.train_id import print_config, generate_train_id, is_same_config
from src.extension import ModelSaver, HistorySaver, HistoryLogger, IntervalTrigger, LearningCurvePlotter, MinValueTrigger
from src.util import set_random_seed


class EVLabelWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        ev = torch.tensor([label - 4.5], dtype=torch.float32)  # EV値に変換 CIFAR-10のカテゴリラベル（0〜9）をEV値（-4.5〜+4.5）に線形変換
        return img, ev

    def __len__(self):
        return len(self.dataset)


@hydra.main(version_base=None, config_path=f"/{os.environ['PROJECT_NAME']}/conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    set_random_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(f"/{os.environ['PROJECT_NAME']}/outputs/train")
    train_id = generate_train_id(cfg)
    p = output_dir / "history" / train_id
    p.mkdir(parents=True, exist_ok=True)

    print_config(cfg)
    cfg_path = p / 'config.yaml'
    if cfg_path.exists():
        existing_cfg = OmegaConf.load(str(cfg_path))
        if not is_same_config(cfg, existing_cfg):
            raise ValueError("Train ID {} already exists, but config is different".format(train_id))
    OmegaConf.save(cfg, str(cfg_path))

    transforms = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(**cfg.dataset.train.transform.random_resized_crop),
        #データ拡張（左右反転、回転+α上下反転）
        v2.RandomHorizontalFlip(p=0.5),  #左右反転
        v2.RandomRotation(degrees=15), # 画像を±15度以内で回転
        v2.RandomVerticalFlip(p=0.5),        # 上下反転
        v2.RandomHorizontalFlip(**cfg.dataset.train.transform.random_horizontal_flip),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(**cfg.dataset.train.transform.normalize),
    ])

    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms)
    dataset = EVLabelWrapper(dataset)
    train_set, val_set = torch.utils.data.random_split(dataset, cfg.dataset.random_split.lengths)

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **cfg.dataloader)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **cfg.dataloader)

    net = SimpleCNN(num_classes=1).to(device)
    input_size = train_set[0][0].shape
    torchinfo.summary(net, input_size=(cfg.dataloader.batch_size, *input_size))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), **cfg.optimizer.params)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        [i * cfg.epoch for i in cfg.lr_scheduler.params.milestones],
        gamma=cfg.lr_scheduler.params.gamma
    )

    evaluators = [LossEvaluator(criterion, criterion_name="MSE")]

    extensions = [
        ModelSaver(directory=p, name=lambda x: "best_model.pth", trigger=MinValueTrigger(mode="validation", key="loss")),
        HistorySaver(directory=p, name=lambda x: "history.pth", trigger=IntervalTrigger(period=1)),
        HistoryLogger(trigger=IntervalTrigger(period=1), print_func=print),
        LearningCurvePlotter(directory=p, trigger=IntervalTrigger(period=1)),
    ]

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