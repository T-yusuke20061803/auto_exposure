from pathlib import Path
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.model_selection import train_test_split
import torchinfo

from src.dataset import AnnotatedDatasetFolder, collate_fn_skip_none, pil_loader
from src.model import SimpleCNN
from src.trainer import Trainer, LossEvaluator
from src.train_id import print_config, generate_train_id
from src.extension import ModelSaver, HistorySaver, HistoryLogger, IntervalTrigger, LearningCurvePlotter, MinValueTrigger
from src.util import set_random_seed

@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    set_random_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path("./outputs/train")
    train_id = generate_train_id(cfg)
    p = output_dir / "history" / train_id
    p.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, p/'config.yaml')

    # データ拡張パイプラインをPythonコード内で定義
    # YAMLファイルを変更するだけでデータ拡張の内容を自由に変更可能になるように修正
    train_transforms = hydra.utils.instantiate(cfg.dataset.train.transform)
    val_transforms = hydra.utils.instantiate(cfg.dataset.test.transform)

     # 2. マスターアノテーションファイルを読み込んで分割
    master_df = pd.read_csv(cfg.dataset.train.annotation_file)
    train_df, val_df = train_test_split(
        master_df,
        test_size = cfg.dataset.split.val_size,
        random_state = cfg.dataset.split.random_state
    )
    print(f'データ分割：訓練{len(train_df)}, 検証{len(val_df)}件')

    #データセットを作成
    train_set = AnnotatedDatasetFolder(
        root=cfg.dataset.train.root, 
        dataframe=train_df,
        loader=pil_loader, 
        transform=train_transforms # YAMLから作成したtransformオブジェクトを渡す
    )
    #検証用データにはval_transformsを適用
    val_set = AnnotatedDatasetFolder(
        root=cfg.dataset.train.root, 
        dataframe=val_df,
        loader=pil_loader, 
        transform=val_transforms # YAMLから作成したtransformオブジェクトを渡す
    )

    # データローダーを作成 (cfg.dataloaderの内容を**で展開して渡す)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, collate_fn=collate_fn_skip_none, **cfg.dataloader)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, collate_fn=collate_fn_skip_none, **cfg.dataloader)
    
    # --- モデル、損失関数、最適化手法の準備 ---
    #YAMLから直接インスタンス化 model=resnet`のようにコマンドラインで指定するだけでモデルを切り替え可能になる
    net = hydra.utils.instantiate(cfg.model).to(device)

    # モデルの構造とパラメータ数を表示
    sample_batch = next(iter(train_loader))
    if sample_batch[0] is not None:
        input_size = sample_batch[0][0].shape
        torchinfo.summary(net, input_size=(cfg.dataloader.batch_size, *input_size))

     # 損失関数、最適化手法、スケジューラ
    criterion = nn.MSELoss()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=net.parameters())
    scheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer=optimizer)

    evaluators = [LossEvaluator(criterion, criterion_name="MSE")]
    # 訓練中のイベント（モデル保存、ログ記録、グラフ描画など）を定義
    extensions = [
        ModelSaver(directory=p, name=lambda x: "best_model.pth", trigger=MinValueTrigger(mode="validation", key="loss")),
        HistorySaver(directory=p, name=lambda x: "history.pth", trigger=IntervalTrigger(period=1)),
        HistoryLogger(trigger=IntervalTrigger(period=1), print_func=print),
        LearningCurvePlotter(directory=p, trigger=IntervalTrigger(period=1)),
    ]

    # --- 訓練の実行 ---
    trainer = Trainer(
        net, optimizer, criterion, train_loader, 
        scheduler=scheduler, extensions=extensions, evaluators=evaluators, device=device
    )
    trainer.train(cfg.epoch, val_loader)

     # 最終モデルの保存 
    model_output_dir = Path("./outputs/model")
    model_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), model_output_dir / "final_model.pth")

if __name__ == "__main__":
    main()