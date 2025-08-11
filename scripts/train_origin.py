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

@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    set_random_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path("../outputs/train")
    train_id = generate_train_id(cfg)
    p = output_dir / "history" / train_id
    p.mkdir(parents=True, exist_ok=True)

    print_config(cfg)
    cfg_path = p / 'config.yaml'
    if cfg_path.exists():
        existing_cfg = OmegaConf.load(str(cfg_path))
        if not is_same_config(cfg, existing_cfg):
            raise ValueError(f"Train ID {train_id} already exists, but config is different")
    OmegaConf.save(cfg, str(cfg_path))
    
    # --- 主要な変更点 ---
    # 変更点: transformsの定義を削除し、Hydraがconfig.yamlから直接生成するようにする
    # 変更点: config.yamlの定義に基づき、AnnotatedDatasetFolderを自動で読み込む

    dataset = hydra.utils.instantiate(cfg.dataset.train)
     # 読み込んだデータセットを学習用と検証用に分割
    train_set, val_set = torch.utils.data.random_split(dataset, cfg.dataset.random_split.lengths)
    # 変更点: collate_fnを追加してエラー耐性を向上
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, collate_fn=collate_fn_skip_none, **cfg.dataloader)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, collate_fn=collate_fn_skip_none, **cfg.dataloader)
    
    net = SimpleCNN(num_classes=1).to(device)

    # 最初の有効なデータでモデルのサマリーを表示
    sample_batch = next(iter(train_loader))
    if sample_batch[0] is not None:
        input_size = sample_batch[0][0].shape
        torchinfo.summary(net, input_size=(cfg.dataloader.batch_size, *input_size))
     # 損失関数、最適化手法、スケジューラ
    criterion = nn.MSELoss()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=net.parameters())
    scheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer=optimizer)

    evaluators = [LossEvaluator(criterion, criterion_name="MSE")]
    extensions = [
        ModelSaver(directory=p, name=lambda x: "best_model.pth", trigger=MinValueTrigger(mode="validation", key="loss")),
        HistorySaver(directory=p, name=lambda x: "history.pth", trigger=IntervalTrigger(period=1)),
        HistoryLogger(trigger=IntervalTrigger(period=1), print_func=print),
        LearningCurvePlotter(directory=p, trigger=IntervalTrigger(period=1)),
    ]

    # トレーナーの初期化と学習の開始
    trainer = Trainer(
        net, optimizer, criterion, train_loader, 
        scheduler=scheduler, extensions=extensions, evaluators=evaluators, device=device
    )
    trainer.train(cfg.epoch, val_loader)

     # 最終モデルの保存 
    model_output_dir = Path("../outputs/model")
    model_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), model_output_dir / "final_model.pth")

if __name__ == "__main__":
    main()