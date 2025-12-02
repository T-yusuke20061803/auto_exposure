# 訓練用コード
from pathlib import Path
import os
import csv
import torch
import datetime
import matplotlib.pyplot as plt
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
from src.dataset import AnnotatedDatasetFolder, pil_loader,imageio_loader, dng_loader, collate_fn_skip_none, LogTransform


class DebugPrintTransform(object):
    def __init__(self, name=""):
        self.name = name

    def __call__(self, tensor):
        print("\n" + "="*20)
        print(f"DebugPoint: {self.name}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Min:   {tensor.min().item():.6f}")
        print(f"  Max:   {tensor.max().item():.6f}")
        print(f"  Mean:  {tensor.mean().item():.6f}")
        print("="*20 + "\n")
        return tensor

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
        LogTransform(),
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
        LogTransform(),
        v2.Normalize(**cfg.dataset.val.transform.normalize),
    ])
     # データ分割 
    train_df = pd.read_csv(cfg.dataset.train.csv_file)
    val_df = pd.read_csv(cfg.dataset.val.csv_file)
    # データセット

    train_set = AnnotatedDatasetFolder(
        root=cfg.dataset.train.root,
        csv_file=cfg.dataset.train.csv_file, # dataframe= ではなく csv_file=
        loader=imageio_loader, #imageio_loader から dng_loader
        transform=train_transforms
    )
    val_set = AnnotatedDatasetFolder(
        root=cfg.dataset.val.root,
        csv_file=cfg.dataset.val.csv_file, # dataframe= ではなく csv_file=
        loader=imageio_loader, # imageio_loader から dng_loader
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
        net = ResNetRegression(**cfg.model.params).to(device) #ResNet -> ResNetRegression 事前学習済み
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
                print(f"[INFO] 最新のモデルを読み込み: {model_path}")
        else:
            # 具体的なパスが指定された場合は、そのまま使う
            model_path = Path(model_path_str)

        if model_path.exists():
            net.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"警告: 指定された重みファイル:無し  {model_path}")  

    print("\n--- 事前学習（凍結）設定の確認 ---")
    if cfg.model.params.get("freeze_base", False):
        unfreeze_count = cfg.model.params.get("unfreeze_layers", 0)
        if unfreeze_count == 0:
            print(f" [INFO] 凍結: ON (freeze_base: true, unfreeze_layers: 0)")
            print(f" [INFO] 学習対象: 最後のFC層のみ")
        else:
            print(f" [INFO] 凍結: 部分的 (freeze_base: true, unfreeze_layers: {unfreeze_count})")
            print(f" [INFO] 学習対象: 最後のFC層 + Layer {5 - unfreeze_count} 以降")
    else:
        print(f" [INFO] 凍結: OFF (freeze_base: false)")
        print(f" [INFO] 学習対象: 全てのパラメータ (11.18M)")
    print("------------------------------------\n")

    # === [デバッグ] LogTransform の動作確認 === LogTransformありなしで二回実行しそれぞれの値を比較する
    print("\n" + "="*30)
    print("LogTransform 動作確認 (1バッチ目)")
    print("="*30)
    try:
        # 1バッチ取得
        # (アンパックしない)
        sample_batch_debug = next(iter(train_loader)) 
        
        # 最初の要素（画像バッチ）を取得
        image_batch = sample_batch_debug[0] 
        
        if image_batch is not None and len(image_batch) > 0:
            print(f"  バッチ形状 (画像): {image_batch.shape}")
            print(f"  最小値 (Min): {image_batch.min().item():.4f}")
            print(f"  最大値 (Max): {image_batch.max().item():.4f}")
            print(f"  平均値 (Mean): {image_batch.mean().item():.4f}")
            print(f"  標準偏差 (Std): {image_batch.std().item():.4f}")
        else:
            print("  [WARN] デバッグ用の画像バッチが取得できませんでした。")

    except Exception as e:
        print(f"  [ERROR] デバッグ中にエラー: {e}")
    print("="*30 + "\n")

    # サマリー表示
    sample_batch = next(iter(train_loader))
    if sample_batch[0] is not None:
        input_size = sample_batch[0][0].shape
        torchinfo.summary(net, input_size=(cfg.dataloader.batch_size, *input_size))


 # 損失関数、最適化手法、スケジューラ
    #MSEからSmoothL１に変更すること、外れ値の影響を軽減
    #学習用
    train_criterion = nn.MSELoss().to(device)
    # 監視・評価用損失 (ログ記録・グラフ化用)
    # 公平な比較のために MSE(RMSE) を、実用誤差確認のために MAE を、
    # 学習の進み具合確認のために SmoothL1 をそれぞれ用意
    val_criterion_mse = nn.MSELoss().to(device)
    val_criterion_mae = nn.L1Loss().to(device)
    val_criterion_smooth = nn.SmoothL1Loss(beta=1.0).to(device)
    #評価用
    criterion = nn.MSELoss().to(device) #損失関数変更：nn.SmoothL1Loss(beta=1.0) or nn.MAELoss(beta=1.0) 

    # 差動学習率 (DLR) の設定 
    
    base_lr = cfg.optimizer.params.lr 

    
    # モデルの「ヘッド」（最後の層）の名前を特定
    model_name_lower = cfg.model.name.lower()
    head_name = None
    
    # モデルの「本体」の名前 (net.resnet, net.effnet, net.mobilenet)
    base_model_name = None 

    if model_name_lower == "resnet":
        head_name = "fc"
        base_model_name = "resnet" # ResNetRegression は self.resnet を持つ
    elif model_name_lower == "efficientnet":
        head_name = "classifier"
        base_model_name = "effnet" # RegressionEfficientNet は self.effnet を持つ
    elif model_name_lower == "mobilenet":
        head_name = "classifier"
        base_model_name = "mobilenet" # RegressionMobileNet は self.mobilenet を持つ

    # パラメータをグループ分け
    if head_name and base_model_name and hasattr(net, base_model_name) and hasattr(getattr(net, base_model_name), head_name):
        
        # getattr を使い、動的に層を取得
        # 例: net.effnet.classifier
        head_layer = getattr(getattr(net, base_model_name), head_name)

        # 新しい層（head_name）のパラメータ
        # 例: net.effnet.classifier.parameters()
        new_layer_params = [p for p in head_layer.parameters() if p.requires_grad]
        
        # 凍結解除した層（Base）のパラメータ
        # (head_name で始まらない *全ての* パラメータ)
        base_params = [
            p for n, p in net.named_parameters() 
            if not n.startswith(f"{base_model_name}.{head_name}") and p.requires_grad
        ]
        
        param_groups = [
            {'params': base_params, 'lr': base_lr * 0.01}, # ← Base層は 1/100 のLR
            {'params': new_layer_params, 'lr': base_lr}    # ← Head層は通常のLR
        ]
        
        print(f"[INFO] Base層 (凍結解除) のLR: {base_lr * 0.01:.1e}")
        print(f"[INFO] Head層 (新規) のLR: {base_lr:.1e}:net.{base_model_name}.{head_name}")

    else:
        # DLR非対応モデル (SimpleCNNなど) または head_name が見つからない場合
        if model_name_lower not in ["resnet", "efficientnet", "mobilenet"]:
            print(f"[INFO] 差動学習率(DLR)は {cfg.model.name} では未サポートです。単一のLRを使用します。")
        else:
            print(f"[WARN] DLR設定エラー: 'net.{base_model_name}.{head_name}' が見つかりません。単一のLRを使用します。")
        param_groups = net.parameters()

    opt_name = cfg.optimizer.name.lower()
    
    # DLR設定時は cfg.optimizer.params.lr を上書きするため、
    # オプティマイザのデフォルト 'lr' を削除する
    opt_params = OmegaConf.to_container(cfg.optimizer.params, resolve=True)
    if 'lr' in opt_params:
        del opt_params['lr'] # 'lr' は param_groups で指定済み
        
    if opt_name == "adam":
        #  (net.parameters() -> param_groups)
        optimizer = optim.Adam(param_groups, lr=base_lr, **opt_params) 
    elif opt_name == "adamw":
        optimizer = optim.AdamW(param_groups, lr=base_lr, **opt_params)
    elif opt_name == "sgd":
        optimizer = optim.SGD(param_groups, lr=base_lr, **opt_params)
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
        scheduler_type = "batch"

    else:
        # スケジューラを使わない場合
        scheduler = None
        scheduler_type = None

    # 評価指標と拡張
    # 訓練用の損失 (SmoothL1Loss) + 監視用の損失 (MSE)
    #training_criterion = criterion #学習ではSmoothL1Lossを用いて学習を安定させ、別途でMSEを計算するがそれを区別するため
    
    evaluators = [LossEvaluator(val_criterion_mse, criterion_name="mse"), #MSE
                  LossEvaluator(val_criterion_mae, criterion_name="mae"), #MAE
                  LossEvaluator(val_criterion_smooth, criterion_name="Smooth")]#,LossEvaluator(mse_eval_criterion, criterion_name="mse")]
    extensions = [
            ModelSaver(directory=history_path, name=lambda x: "best_model.pth", trigger=MinValueTrigger(mode="validation", key="mse")),
            HistorySaver(directory=history_path, name=lambda x: "history.pth", trigger=IntervalTrigger(period=1)),
            HistoryLogger(trigger=IntervalTrigger(period=1), print_func=print),
            LearningCurvePlotter(directory=history_path, trigger=IntervalTrigger(period=1)),
        ]

    # 学習
    trainer = Trainer(
            net, 
            optimizer, 
            train_criterion, # (MSEではなくSmoothL1Lossを渡す) criterion -> training_criterion
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