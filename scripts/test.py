from pathlib import Path
import os
import shutil
import argparse

import torch
import torchvision
from torchvision.transforms import v2 

from omegaconf import OmegaConf

from src.data_pipeline import DataPipeline
from src.model import ResNet, SimpleCNN #変更点simpleCNN→ResNet
from src.trainer import AccuracyEvaluator
from src.trainer import LossEvaluator
from src.train_id import print_config
from src.util import set_random_seed


def main(
        cfg,
        train_id,
        seed
    ):
    set_random_seed(seed)#乱数値の固定（再現性確保のため）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#使用デバイスの設定
    print_config(cfg)

#モデルの再構築と重みの読み込み
    model_name = cfg.model.name.lower()#.lower()→モデル名（ResNet→resnet）のように全て小文字変換させることで統一させ、バグorエラーを防止する
    if model_name == "resnet":
        net = ResNet(**cfg.model.params).to(device)
    elif model_name == "simplecnn":
        net = SimpleCNN(**cfg.model.params).to(device)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model.name}")
    
    model_path = Path("outputs/train/history") / train_id / "best_model.pth"
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device)
  #テスト用の前処理
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(**cfg.dataset.train.transform.normalize),
    ])
    #テストデータセットの用意
    dataset = torchvision.datasets.CIFAR10(#MNIST→CIFAR-10に変更することで統一
        root="./data/",
        train=False,
        download=True,
    )
    #transforms,Normalize等が一致しないと精度に影響
    #classes = dataset.classes 06/07
    datapipe = DataPipeline(dataset, static_transforms=transforms, dynamic_transforms=None, max_cache_size=0)
    test_loader = torch.utils.data.DataLoader(
        datapipe,
        shuffle=False,
        batch_size=64,
        num_workers=0
    )

    net.eval()
    criterion = torch.nn.MSELoss() #06/07　追加
    #精度に関する部分はここが考えられる 
    evaluator = LossEvaluator(criterion, criterion_name="MSE")
    #evaluator = AccuracyEvaluator(classes)
    evaluator.initialize()#評価の内部状態を初期化
    #評価実行
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device)
        targets = targets.to(device).float().unsqueeze(1) 
        outputs = net(inputs)
        evaluator.eval_batch(outputs, targets)
    result = evaluator.finalize()
    #精度表示
    print(f"\n[INFO]MSE result for Train ID {train_id} Model: {cfg.model.name})")
    for key, value in result.items():
        print(f"{key}: {value:.4f}")

    #結果をファイルに保存
    log_path = Path("outputs/test_results") / f"{train_id}_result.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path,"w") as f:
        f.write(f"MES result for Train ID {train_id}\n")
        f.write(f"Model: {cfg.model.name}\n")
        for key, value in result.items():
            f.write(f"{key}:{value:.4f}\n")

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--history_dir', type=str,
                        default="./outputs/train/history",
                        help='Directory path for searching trained models')
    parser.add_argument('--seed', type=int,
                        default=0,
                        help='Sets random seed')
    parser.add_argument('--dataset_dir', type=str,
                        default=None,
                        help='Sets random seed')
    parser.add_argument('--remove_untrained_id', action="store_true",
                        help='Delete folders that do not contain model checkpoints')
    parser.add_argument('--skip_tested', action="store_true",
                        help='Skip train IDs that already have test results')

    args = parser.parse_args()
    p = Path.cwd() / args.history_dir

    #フォルダを順に確認
    for q in sorted(p.glob('**/config.yaml')):
        cfg = OmegaConf.load(str(q))
        train_id = q.parent.name
        #今回（5月15日）に相談した所
        if not (q.parent / "best_model.pth").exists():
            print(f"A model for Train ID {train_id} does not exist")
            if args.remove_untrained_id:
                shutil.rmtree(q.parent)
            continue
        #既にテスト済みの場合、スキップ
        result_dir = Path("outputs/test_results") 
        result_file = result_dir / f"{train_id}_result.txt"
        if result_dir.exists():
            print(f"Train ID {train_id} is already tested")
            if args.skip_tested:
                continue
        else:
            result_dir.mkdir(parents=True, exist_ok=True)

        try:
            main(cfg=cfg,
                 train_id=train_id,
                 seed=args.seed)
        except Exception as e:
            print(f"Train ID {train_id} is skipped due to an exception {e}")
