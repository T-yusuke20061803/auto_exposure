import rawpy
import numpy as np
import imageio.v3 as iio
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# === 設定 ===
INPUT_DIR = Path("conf/dataset/HDR+burst/20171106/results_20171023")# 処理対象の元画像(.dng)が入っている親フォルダ
OUTPUT_DIR = Path("conf/dataset/HDR+burst/processed_1024px_exr")# 処理後の画像(.exr)を保存する親フォルダ
TARGET_SIZE = (1024, 1024)  # リサイズ後サイズ

def process_dng(file_path: Path):
    try:
        relative_path = file_path.relative_to(INPUT_DIR)
        output_path = OUTPUT_DIR / relative_path.with_suffix(".exr")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # すでに処理済みならスキップ
        if output_path.exists():
            return "Skipped"

        # DNG読み込み → RGB化
        with rawpy.imread(str(file_path)) as raw:
            rgb = raw.postprocess(
                output_bps=16, # 16bit (0-65535) の精度で出力する
                no_auto_bright=True,# rawpyによる自動明るさ調整を「無効」にする
                use_auto_wb=False,# 自動ホワイトバランスを「無効」にする
                # ガンマ補正を「無効」にする (gamma=(1, 1))
                # これにより、生の「リニアな」輝度情報（HDR）が得られる
                gamma=(1, 1)
            )
            # HDR情報を保持したままfloat化
            # ここでは、[0, 65535] のリニアな輝度情報を保持したまま、データ型だけを「float32(浮動小数点数)」に変換する
            rgb = np.float32(rgb)
            #rgb は [0.0, 65535.0] の範囲の「float32」のNumpy配列になる

        # RGBチャンネル保証
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        elif rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]

        # リサイズ（HDRレンジ保持）
        rgb_resized = resize(rgb, 
                             TARGET_SIZE, 
                             anti_aliasing=True, 
                             preserve_range=True
                             ).astype(np.float32)

        # EXRで保存
        iio.imwrite(str(output_path), rgb_resized, extension=".exr")

        return "Success"

    except Exception as e:
        return f"Failed: {e}"

def main():
    print(f"入力元: {INPUT_DIR}")
    print(f"出力先: {OUTPUT_DIR}")
    print(f"リサイズサイズ: {TARGET_SIZE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(list(INPUT_DIR.rglob("merged.dng")))
    total = len(image_paths)

    if total == 0:
        print("merged.dng ファイル無し")
        return

    print(f"total: {total} 枚の 'merged.dng' 画像を処理中")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_dng, image_paths), total=total))

    success = results.count("Success")
    skipped = results.count("Skipped")
    failed = total - success - skipped

    print("\n前処理完了")
    print(f"  成功: {success} 枚")
    print(f"  スキップ: {skipped} 枚")
    print(f"  失敗: {failed} 枚")

if __name__ == "__main__":
    main()
