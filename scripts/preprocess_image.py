import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import concurrent.futures

#設定
#HDR+burstが入っているディレクトリ
INPUT_DIR = Path("conf/dataset/HDR+burst/20171106/results_20171023")
# リサイズした画像を保存する新しいディレクトリ
OUTPUT_DIR = Path("conf/dataset/HDR+burst/processed_512px")

#リサイズするサイズ（学習サイズ（224）より大きめにすること）
TARGET_SIZE = (512,512)

def resize_image(file_path):
    try:
        #出力先のパスを決める
        relative_path = file_path.relative_to(INPUT_DIR)
        output_path =   OUTPUT_DIR / relative_path

        #出力先の親ディレクトリがなければ作成する
        output_path.parent.mkdir(parents=True, exist_ok=True)

        #すでに処理済みの場合スキップ
        if output_path.exists():
            return(str(file_path),"Skipped")
        
        with Image.open(file_path) as img:
            img_resized = img.resize(TARGET_SIZE, Image.LANCZOS)# 高品質リサイズ
            img_resized.save(output_path)
        return (str(file_path), "Success")
    
    except Exception as e:
        return (str(file_path), f"Failed: {e}")
    

def main():
    print(f"入力元{INPUT_DIR}")
    print(f"出力先{OUTPUT_DIR}")
    print(f"リサイズ先{TARGET_SIZE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    #画像一覧を取得
    image_exts = [".jpg", ".jpeg", ".png"]
    image_paths = sorted([
        p for p in INPUT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in image_exts
    ])

    if not image_paths:
        print(f"error:画像なし")
        return
    
    print(f"total: {len(image_paths)} 枚の画像の前処理中")

    # CPUコアをフルに使って並列処理
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(resize_image, image_paths), total=len(image_paths)))

    print("前処理が完了しました。")

if __name__ == "__main__":
    main()
