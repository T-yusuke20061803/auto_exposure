# 訓練用コード
from pathlib import Path
import os
import csv
import torch
import datetime
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


from src.model import SimpleCNN, ResNet,ResNetRegression, RegressionEfficientNet, RegressionMobileNet
from src.trainer import Trainer, LossEvaluator
from src.train_id import print_config, generate_train_id, is_same_config
from src.extension import ModelSaver, HistorySaver, HistoryLogger, IntervalTrigger, LearningCurvePlotter, MinValueTrigger
from src.util import set_random_seed
from src.dataset import AnnotatedDatasetFolder, pil_loader,imageio_loader, dng_loader, collate_fn_skip_none


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

    # 保存先
    train_id = generate_train_id(cfg)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/train_reg/history") / cfg.model.name
    history_path = output_dir / f"{train_id}_{timestamp}"
    history_path.mkdir(parents=True, exist_ok=True)

    print_config(cfg)
    cfg_path = history_path / 'config.yaml'
    if cfg_path.exists() and not is_same_config(cfg, OmegaConf.load(str(cfg_path))):
        raise ValueError("Train ID conflict with different config.")
    OmegaConf.save(cfg, str(cfg_path))

    # データ変換 (データ拡張＋正規化)
    train_transforms = v2.Compose([
        #v2.ToImage(),(入力がPILではないため)
        v2.RandomResizedCrop(**cfg.dataset.train.transform.random_resized_crop),
        v2.RandomHorizontalFlip(**cfg.dataset.train.transform.random_horizontal_flip),
        v2.RandomRotation(**cfg.dataset.train.transform.random_rotation),
        #v2.ToDtype(torch.float32, scale=True),(入力がすでにfloat32のため)
        v2.Normalize(**cfg.dataset.train.transform.normalize),
        v2.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    ])
    #採用しなかったデータ拡張及び正規化
        #v2.RandomVerticalFlip(**cfg.dataset.train.transform.random_vertical_flip),
        #v2.ColorJitter(**cfg.dataset.train.transform.color_jitter), 
        #v2.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),

    val_transforms = v2.Compose([
        #v2.ToImage(),
        v2.Resize(cfg.dataset.val.transform.resize),
        v2.CenterCrop(cfg.dataset.val.transform.center_crop),
        #v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(**cfg.dataset.val.transform.normalize),
    ])
     # データ分割 
    train_df = pd.read_csv(cfg.dataset.train.csv_file)
    val_df = pd.read_csv(cfg.dataset.val.csv_file)
    # データセット

    train_set = AnnotatedDatasetFolder(
        root=cfg.dataset.train.root,
        csv_file=cfg.dataset.train.csv_file, # dataframe= ではなく csv_file=
        loader=imageio_loader, # ★ imageio_loader から dng_loader
        transform=train_transforms
    )
    val_set = AnnotatedDatasetFolder(
        root=cfg.dataset.val.root,
        csv_file=cfg.dataset.val.csv_file, # dataframe= ではなく csv_file=
        loader=imageio_loader, # ★ imageio_loader から dng_loader
        transform=val_transforms
    )

    print(f"[INFO] 訓練CSV: {cfg.dataset.train.csv_file}:{len(train_set)} 件")
    print(f"[INFO] 検証CSV: {cfg.dataset.val.csv_file}:{len(val_set)} 件")

    # データセット読み込み
    train_loader = DataLoader(train_set, shuffle=True, **cfg.dataloader)
    val_loader = DataLoader(val_set, shuffle=False, **cfg.dataloader)

    # モデル
    if cfg.model.name.lower() == "simplecnn":
        net = SimpleCNN(**cfg.model.params).to(device)
    elif cfg.model.name.lower() == "resnet":
        net = ResNet(**cfg.model.params).to(device) #ResNet -> ResNetRegression 事前学習済み
    elif cfg.model.name.lower() == "efficientnet":
        net = RegressionEfficientNet(**cfg.model.params).to(device)
    elif cfg.model.name.lower() == "mobilenet":
        net = RegressionMobileNet(**cfg.model.params).to(device)
    else:
        raise ValueError(f"未対応のモデルです: {cfg.model.name}")
    
    #  重み読み込み 
    if "load_weights_from" in cfg and cfg.load_weights_from is not None:
        model_path_str = cfg.load_weights_from
        # 'latest'が指定された場合、自動で最新のモデルを探す
        if model_path_str.lower() == 'latest':
            # Hydraの出力は outputs/YYYY-MM-DD/HH-MM-SS 形式なので、その親ディレクトリを探す
            base_history_dir = Path("./outputs/train_reg/history")
            if not base_history_dir.exists():
                raise FileNotFoundError("学習履歴ディレクトリ：無し")
            
             # 現在の実行ディレクトリを取得
            current_run_dir = Path(os.getcwd())
            
            # 自分自身のディレクトリを除外して、更新日時が最新のディレクトリを探す
            other_run_dirs = [d for d in base_history_dir.iterdir() if d.is_dir() and d.resolve() != current_run_dir.resolve()]
            if not other_run_dirs:
                raise FileExistsError("学習履歴：無し")
            else:
                latest_train_dir = max(other_run_dirs, key=lambda p: p.stat().st_mtime)
                model_path = latest_train_dir / "best_model.pth"
                print(f"[INFO] 'latest'が指定されたため、最新のモデルを読み込みます: {model_path}")
        else:
            # 具体的なパスが指定された場合は、そのまま使う
            model_path = Path(model_path_str)

        if model_path.exists():
            net.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"警告: 指定された重みファイルが見つかりません: {model_path}")  
    # サマリー表示
    sample_batch = next(iter(train_loader))
    if sample_batch[0] is not None:
        input_size = sample_batch[0][0].shape
        torchinfo.summary(net, input_size=(cfg.dataloader.batch_size, *input_size))


 # 損失関数、最適化手法、スケジューラ
    criterion = nn.MSELoss()
    # Optimizer 選択 
    opt_name = cfg.optimizer.name.lower()
    if opt_name == "sgd":
        optimizer = optim.SGD(net.parameters(), **cfg.optimizer.params)
    elif opt_name == "adam":
        optimizer = optim.Adam(net.parameters(), **cfg.optimizer.params)
    elif opt_name == "adamw":
        optimizer = optim.AdamW(net.parameters(), **cfg.optimizer.params)
    else:
        raise ValueError(f"未対応のoptimizer: {opt_name}")
    
    # Scheduler 設定
    sched_name = cfg.lr_scheduler.name.lower()
    if sched_name == "multi_step":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(i * cfg.epoch) for i in cfg.lr_scheduler.params.milestones],
            gamma=cfg.lr_scheduler.params.gamma
        )
        scheduler_type = "epoch"

     # ReduceLROnPlateau（性能が停滞したらLR変更）Trainer側で毎エポックの検証lossを渡す必要がある
    elif sched_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **cfg.lr_scheduler.params
        )
        scheduler_type = "plateau"

    elif sched_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.lr_scheduler.params.T_max,
            eta_min=cfg.lr_scheduler.params.eta_min
        )
        scheduler_type = "epoch"

    else:
        # スケジューラを使わない場合
        scheduler = None
        scheduler_type = None

    # 評価指標と拡張
    evaluators = [LossEvaluator(criterion, criterion_name="loss")]
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
            cfg = cfg,
            scheduler=scheduler,
            extensions=extensions,
            evaluators=evaluators,
            device=device
        )
    #学習開始
    trainer.train(cfg.epoch, val_loader)
    # 最終モデル保存
    torch.save(net.state_dict(), history_path / "final_model.pth")


if __name__ == "__main__":
    main()
