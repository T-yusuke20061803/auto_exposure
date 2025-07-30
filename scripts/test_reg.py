# test.py
from pathlib import Path
import os
import shutil
import argparse
import torch
import torchvision
from torchvision.transforms import v2
from PIL import Image
from omegaconf import OmegaConf
import csv

from src.model import ResNet, SimpleCNN
from src.trainer import LossEvaluator
from src.train_id import print_config
from src.util import set_random_seed

class EVLabelWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.dataset = base_dataset

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        ev = torch.tensor([label - 4.5], dtype=torch.float32)
        return img, ev

    def __len__(self):
        return len(self.dataset)

def main(cfg, train_id, seed):
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_config(cfg)

    model_name = cfg.model.name.lower()
    if model_name == "resnet":
        net = ResNet(**cfg.model.params).to(device)
    elif model_name == "simplecnn":
        net = SimpleCNN(**cfg.model.params).to(device)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model.name}")

    model_path = Path("outputs/train/history") / train_id / "best_model.pth"
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device)

    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(**cfg.dataset.train.transform.normalize),
    ])

    dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms)#ここを変更する
    dataset = EVLabelWrapper(dataset)
    test_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=64, num_workers=0)

    net.eval()
    criterion = torch.nn.MSELoss()
    evaluator = LossEvaluator(criterion, criterion_name="MSE")
    evaluator.initialize()

    predictions = []
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        evaluator.eval_batch(outputs, targets)

        for img, target, output in zip(inputs, targets, outputs):
            predictions.append([target.item(), output.item()])
    
    result = evaluator.finalize()
    mse_value = result['loss/MSE']
    rmse_value = torch.sqrt(torch.tensor(mse_value)).item()
    result['loss/RMSE'] = rmse_value # 結果にRMSEを追加

    print(f"\n[INFO] RMSE result for Train ID {train_id} Model: {cfg.model.name}")
    for key, mse_value ,rmse_value in result.items():
        print(f"{key}: {rmse_value:.4f}")
    

    log_path = Path("outputs/test_results") / f"{train_id}_result.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"RMSE result for Train ID {train_id}\n")
        f.write(f"Model: {cfg.model.name}\n")
        for key, value in result.items():
            f.write(f"{key}:{rmse_value:.4f}[{mse_value:.4f}]\n")

    csv_path = Path(f"outputs/predictions/{train_id}_predictions.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path","true_ev", "pred_ev"])
        writer.writerows(predictions)
    print(f"予測結果を {csv_path} に保存しました")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--history_dir', type=str, default="./outputs/train/history")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--remove_untrained_id', action="store_true")
    parser.add_argument('--skip_tested', action="store_true")
    args = parser.parse_args()

    p = Path.cwd() / args.history_dir

    for q in sorted(p.glob('**/config.yaml')):
        cfg = OmegaConf.load(str(q))
        train_id = q.parent.name
        if not (q.parent / "best_model.pth").exists():
            print(f"A model for Train ID {train_id} does not exist")
            if args.remove_untrained_id:
                shutil.rmtree(q.parent)
            continue
        result_dir = Path("outputs/test_results")
        result_file = result_dir / f"{train_id}_result.txt"
        if result_file.exists() and args.skip_tested:
            print(f"Train ID {train_id} is already tested")
            continue
        try:
            main(cfg=cfg, 
                 train_id=train_id, 
                 seed=args.seed)
        except Exception as e:
            print(f"Train ID {train_id} is skipped due to an exception {e}")
