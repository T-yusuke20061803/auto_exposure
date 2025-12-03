import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np
import torch
import random
# ================= 設定 =================
csv_path = Path("conf/dataset/HDR+burst_split/train.csv")
image_root = Path("conf/dataset/HDR+burst/processed_1024px_exr")
output_path = Path("outputs/画像確認.png")
# ========================================

def get_sample_with_ev(df):
    subset = df[df["Exposure"] < -1.0]
    if len(subset) > 0:
        return subset.sample(1).iloc[0] #iloc:行番号や列番号を使用して、PandasのDataFrameからデータを取得または操作するためのメソッド
    return df.sample(1).iloc[0]

def process_image_for_vis(img, ev):
    """表示用にトーンマップとガンマ補正をかける関数"""
    # 1. 露出補正 (Linear)
    factor = 2.0 ** ev
    corrected = img * factor
    # 2. クリップ (0-1)
    clipped = np.clip(corrected, 0.0, 1.0)
    # 3. ガンマ補正 (sRGB)
    gamma = np.power(clipped, 1.0/2.2)
    return gamma


def debug_pixel_values():
    print("原因特定のための数値追跡")
    #データ追跡
    df = pd.read_csv(csv_path)
    row = get_sample_with_ev(df)

    fname = row["Filename"]
    label_ev = float(row["Exposure"])
    filepath = image_root / row["filepath"]

    print(f"検証ファイル:{fname}")
    print(f"正解ラベル:{label_ev}")
    print("-"*40)

    #画像の読み込み
    try:
        img_raw = iio.imread(filepath).astype(np.float32) #astype:NumPy配列のデータ型を変換するためのメソッド
        #0～1に正規化←学習時と同じ条件にする
        img_01 = img_raw / 65535.0

        # 代表値（画像の真ん中あたりの画素）を取得
        h, w, _ = img_01.shape
        center_pixel = img_01[h//2, w//2, :] # RGB
        mean_brightness = img_01.mean()

        print(f"【1. 画像自体のチェック】")
        print(f"  画素値の平均: {mean_brightness:.6f}")
        print(f"  中心の画素値: {center_pixel}")

        if mean_brightness < 0.05:
            print("画像状態：暗い")
        elif mean_brightness > 0.5:
            print("画像状態：明るい")
        else:
            print("画像状態；普通")

    except Exception as e:
        print("error:画像読み込み失敗")

    #補正計算について
    print("補正計算の確認")

    #補正係数
    gain = 2 ** label_ev
    print(f"{label_ev}：倍率{gain:.4f}倍")

    #補正後の値
    corrected_img = mean_brightness *gain
    print(f"補正後の平均輝度→{mean_brightness}*{gain}={corrected_img}")

    #ディスプレイ表示を確認
    print("ディスプレイ表示の確認")

    # トーンマッピング
    tone_mapped = np.clip(corrected_img, 0.0, 1.0)
    # ガンマ補正
    ganma_val = np.power(tone_mapped, 1.0/2.2)

    print(f"0～1のクリッピング後{tone_mapped:.6f}")
    print(f"ガンマ補正後(最終出力){ganma_val:.6f}")


    # 最終判定
    print("\n" + "="*40)
    print("結果まとめ")
    print("="*40)

    #判定ロジック
    if mean_brightness > 0.3 and label_ev < 0:
        print("【原因候補: 正解値の意味】")
        print("元画像：ある程度明るい＋正解ラベル：暗くする")
        print("補正後画像が暗くなっていたらOK")

    elif mean_brightness < 0.01 and label_ev < 0:
        # 元が暗いのに、マイナスEVでさらに暗くしている場合
        print("【原因候補: 正解値と画像のミスマッチ】")
        print("  元画像:暗い+正解ラベル：暗くする")
        print("  考えられる原因:")
        print("    1. データセット自体が「暗い画像にマイナスのラベル」をつけている（仕様）")
        print("    2. 実は「マイナスラベル＝明るくする」という定義だった（符号の解釈）")

    elif ganma_val > 0.05:
        print("【原因候補: ディスプレイ/感覚】")
        print(f"  数値上は {ganma_val:.2f} (0-1) の明るさが出ています。")
        print("  これは真っ黒ではありません。モニタの輝度や、生成画像の確認方法に問題があるかもしれません。")
    
    else:
        print("  数値が極端に小さいです。画像データとラベルの組み合わせを再確認する必要があります。")

    
    print(f"\n--- 視覚確認画像の生成を開始 ---")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # (左上) 元画像
    # トーンマップなしで表示 (ただし1.0でクリップしないとエラーになる場合があるため簡易表示)
    orig_view = process_image_for_vis(img_01, 0.0)
    axes[0, 0].imshow(orig_view)
    axes[0, 0].set_title(f"Original (EV=0)\nMean: {mean_brightness:.4f}")
    axes[0, 0].axis('off')

    # (右上) ヒストグラム
    axes[0, 1].hist(img_01.ravel(), bins=50, range=(0.0, 0.5), color='gray')
    axes[0, 1].set_title("Histogram (Linear, 0.0-0.5)")
    axes[0, 1].set_xlabel("Pixel Value")
    axes[0, 1].set_ylabel("Count")

    # (左下) 現在の補正 (Code 1: 2^EV)
    # そのままの符号で補正
    current_view = process_image_for_vis(img_01, label_ev)
    axes[1, 0].imshow(current_view)
    axes[1, 0].set_title(f"Current Code (2^EV)\nEV={label_ev} (x{2**label_ev:.2f})", color='red')
    axes[1, 0].axis('off')

    # (右下) 反転補正 (Code 2改: 2^-EV)
    # 符号を反転して補正
    inverted_view = process_image_for_vis(img_01, -label_ev)
    axes[1, 1].imshow(inverted_view)
    axes[1, 1].set_title(f"Inverted (2^-EV)\nEV={-label_ev} (x{2**-label_ev:.2f})", color='blue')
    axes[1, 1].axis('off')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"画像を保存しました: {output_path}")
    print(" 左下(赤): 現在のコードの結果")
    print(" 右下(青): 符号を反転した場合の結果")
    print(" -> どちらが自然な明るさに見えるか確認してください。")

if __name__ == "__main__":
    debug_pixel_values()






