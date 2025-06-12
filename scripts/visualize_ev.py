import csv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import argparse

def plot_ev_predictions(csv_file, output_dir):
    df = pd.read_csv(csv_file)

    # 散布図（予測EV vs 正解EV）
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="true_ev", y="pred_ev", s=50, alpha=0.7)
    plt.plot([df.true_ev.min(), df.true_ev.max()],
             [df.true_ev.min(), df.true_ev.max()],
             'r--', label="Ideal")
    plt.xlabel("True EV")
    plt.ylabel("Predicted EV")
    plt.title("True vs Predicted EV")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "scatter_ev.png")
    plt.close()

    # 誤差分布のヒストグラム
    df["error"] = df["pred_ev"] - df["true_ev"]
    plt.figure(figsize=(6, 4))
    sns.histplot(df["error"], kde=True, bins=30)
    plt.xlabel("Prediction Error (EV)")
    plt.title("Distribution of Prediction Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "error_histogram.png")
    plt.close()

    print(f"可視化完了: {output_dir}/scatter_ev.png, error_histogram.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to predictions CSV file")
    parser.add_argument("--output_dir", type=str, default="./outputs/visualizations", help="Directory to save visualizations")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    plot_ev_predictions(args.csv, args.output_dir)
