# 評価用コード
from pathlib import Path
import torch, csv, datetime, shutil
import torchvision.utils as vutils
from torchvision.transforms import v2
import hydra
from omegaconf import DictConfig

from src.dataset import AnnotatedDatasetFolder, pil_loader, collate_fn_skip_none
from src.model import SimpleCNN, ResNet, RegressionEfficientNet, RegressionMobileNet
from src.trainer import LossEvaluator
from src.util import set_random_seed

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def adjust_exposure(image_tensor, ev_value):
    correction_factor = 2.0 ** ev_value
    corrected_image = image_tensor * correction_factor
    return torch.clamp(corrected_image, 0, 1)

@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    set_random_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train ID を config.yaml から取得 
    if "train_id" in cfg and cfg.train_id != "your_train_id_here":
        train_id = cfg.train_id
    else:
        history_dir = Path("outputs/train_reg/history")
        if not history_dir.exists():
            raise FileNotFoundError("学習済みモデルが保存されていません")
        latest = max(history_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        train_id = latest.name
        print(f"[INFO] train_id が指定されていないため最新を使用します: {train_id}")

    #モデルとパス
    model_path = Path("./outputs/train_reg/history") / train_id / "best_model.pth"
    config_path = Path("./outputs/train_reg/history")/ train_id / "config.yaml"

    if not model_path.exists():
        raise FileNotFoundError(f"最良モデルが見つかりません: {model_path}")
    if not config_path.exists():
        print("[WARN] config.yaml が見つかりません。train_id のみで識別します。")
    
    print(f"[INFO] 使用モデル: {model_path}")

    #モデル読込
    if cfg.model.name.lower() == "simplecnn":
        net = SimpleCNN(**cfg.model.params).to(device)
    elif cfg.model.name.lower() == "resnet":
        net = ResNet(**cfg.model.params).to(device)
    elif cfg.model.name.lower() == "efficientnet":
        net = RegressionEfficientNet(**cfg.model.params).to(device)
    elif cfg.model.name.lower() == "mobilenet":
        net = RegressionMobileNet(**cfg.model.params).to(device)
    else:
        raise ValueError(f"未対応のモデルです: {cfg.model.name}")
    
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

#評価時においてはデータ拡張を行わない
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=cfg.dataset.test.transform.normalize.mean,
                     std=cfg.dataset.test.transform.normalize.std),
    ])

# test.csv を読み込む
    dataset = AnnotatedDatasetFolder(
        root=cfg.dataset.test.root,
        csv_file=cfg.dataset.test.csv_file,
        loader=pil_loader,
        transform=transform
    )
    
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        collate_fn=collate_fn_skip_none
    )
    #評価処理
    criterion = torch.nn.MSELoss()
    evaluator = LossEvaluator(criterion, "MSE")
    evaluator.initialize()

    predictions = []
    best_image_info = {"min_error": float("inf")}

    with torch.no_grad():
        for batch_idx, (inputs, targets, filenames) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            evaluator.eval_batch(outputs, targets)
            errors = torch.abs(outputs - targets).squeeze()

            for i, (filename, target, output) in enumerate(zip(filenames, targets, outputs)):
                predictions.append([filename, target.item(), output.item()])
                if errors[i] < best_image_info["min_error"]:
                    best_image_info.update({
                        "min_error": errors[i].item(),
                        "original": inputs[i].cpu(),
                        "pred_ev": output.item(),
                        "true_ev": target.item(),
                        "filename": filename
                    })

    #  評価結果計算
    result = evaluator.finalize()
    print("DEBUG: evaluator result =", result)#確認用
    #MSEのキーを柔軟に取得
    # MSE 取得（キーが "loss")
    mse_value = result.get("loss/MSE",result.get("loss", None))
    if mse_value is None:
        raise KeyError(f"MSEが見つかりません: {result}")

    rmse_value = float(torch.sqrt(torch.tensor(mse_value)))
    result["loss/MSE"] = mse_value
    result["loss/RMSE"] = rmse_value

     # ターミナルに分かりやすく表示 
    print("\n=== 最良モデルの検証結果 ===")
    print(f"Train ID: {train_id}")
    print(f"Model:{cfg.model.name}")
    print(f"Validation MSE:  {result['loss/MSE']:.4f}")
    print(f"Validation RMSE: {result['loss/RMSE']:.4f}")


    #保存処理
    timestamp = datetime.datatime.now().strftime("%Y%m%d_%H%M%S")
    # モデル別フォルダ構造に整理
    model_name = cfg.model.name
    output_root = Path(f"outputs/train_reg/outputs/ {model_name} / {train_id}_{timestamp}")
    result_dir = output_root / "result"
    csv_dir = output_root/ "csv_result"
    bestpred_dir = output_root/ "best_predictions"

    result_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    bestpred_dir.mkdir(parents=True, exist_ok=True)
 
    shutil.copy(model_path, output_root / "best_model.pth")
    if config_path.exists():
        shutil.copy(config_path, output_root / "config.yaml")
    
    # ログ保存 (GitHubで追跡されるディレクトリへ)
    #log_dir = Path("outputs/train_reg/history") / train_id / "result"
    #log_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{cfg.argtrain_id}_result.txt"
    with open(result_path, "w") as f:
        f.write(f"=== 最良モデルの検証結果 ===\n")
        f.write(f"Train ID: {train_id}\n")
        f.write(f"Model:{cfg.model.name}\n")
        f.write(f"Validation MSE:  {result['loss/MSE']:.5f}\n")
        f.write(f"Validation RMSE: {result['loss/RMSE']:.5f}\n")

    # 予測結果保存
    #pred_dir = Path("outputs/train_reg/history") / train_id / "csv_result"
    #pred_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{cfg.args.train_id}_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_ev", "pred_ev"])
        writer.writerows(predictions)
    print(f"\n予測結果を {csv_path} に保存しました")

    # 補正画像保存
    if "original" in best_image_info:
        mean, std = cfg.dataset.test.transform.normalize.mean, cfg.dataset.test.transform.normalize.std
        denorm_img = denormalize(best_image_info["original"], mean, std)
        corrected_img = adjust_exposure(denorm_img, best_image_info["pred_ev"])

        #img_dir = Path("outputs/train_reg/history") / train_id / "best_predictions"
        #img_dir.mkdir(parents=True, exist_ok=True)
        # 元のファイル名から拡張子 (.jpgなど) を取り除く
        base_filename = Path(best_image_info['filename']).stem

        original_path = bestpred_dir / "best_predictions"/ f"{base_filename}_補正前.png"
        corrected_path = bestpred_dir / "best_predictions"/ f"{base_filename}_補正後.png"

        vutils.save_image(denorm_img, original_path)
        vutils.save_image(corrected_img, corrected_path)

        print(f"補正前後の画像を {output_root} に保存しました")

if __name__ == "__main__":
    main()
