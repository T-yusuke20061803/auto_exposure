#守田君が用いていたコード
import gradio as gr
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import csv
from datetime import datetime
import numpy as np
from pathlib import Path

from src.dataset import HDRBurstDataset
from src.util import normalize_hdr



class AnnotationTool:
    def __init__(self, image_dir, output_file):
        self.dataset = HDRBurstDataset(image_dir)
        self.current_index = 0
        self.output_file = Path(output_file)
        self.init_csv()
        self.current_image = None

    def init_csv(self):
        if not self.output_file.parent.exists():
            self.output_file.parent.mkdir(parents=True)
        if not self.output_file.exists():
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Index', 'Filename', 'Exposure', 'Comment', 'Timestamp'])

    def save_annotation(self, exposure, comment):
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_index,
                self.dataset.samples[self.current_index].parent.stem,
                exposure,
                comment,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

    def get_image(self, index=None):
        if index is not None:
            self.current_index = index
        image = self.dataset[self.current_index]
        hdr, _, _ = normalize_hdr(image, 0)
        self.current_image = hdr
        filename = self.dataset.samples[self.current_index].parent.stem
        return self.tone_map(self.current_image), filename, f"{self.current_index + 1}/{len(self.dataset)}"

    def next_image(self, exposure, comment):
        self.save_annotation(exposure, comment)
        self.current_index = (self.current_index + 1) % len(self.dataset)
        return self.get_image()

    def adjust_exposure(self, image, exposure):
        factor = 2 ** exposure  # Convert slider value to brightness factor
        return self.tone_map(np.clip(image * factor, 0, 1))
    
    def tone_map(self, image):
        return np.clip(image ** (1/2.2), 0, 1)


def create_ui(tool):
    with gr.Blocks() as app:
        gr.Markdown("# 画像露出アノテーションツール")
        
        with gr.Row():
            progress_output = gr.Textbox(label="進捗")
            index_input = gr.Number(label="インデックス指定", precision=0)
            filename_output = gr.Textbox(label="ファイル名")

        image_output = gr.Image(label="画像")
        
        with gr.Row():
            exposure_slider = gr.Slider(-4, 4, value=0, step=0.1, label="露出評価")
            comment_input = gr.Textbox(label="コメント", max_lines=3)

        save_button = gr.Button("保存して次へ")
        

        def update_image(index=None):
            image, filename, progress = tool.get_image(index)
            return image, filename, progress, 0, ""
        
        def next_image(exposure, comment):
            image, filename, progress = tool.next_image(exposure, comment)
            return image, filename, progress, 0, ""

        def adjust_image_exposure(exposure):
            if tool.current_image is not None:
                adjusted_image = tool.adjust_exposure(tool.current_image, exposure)
                return adjusted_image
            return None

        save_button.click(
            next_image,
            inputs=[exposure_slider, comment_input],
            outputs=[image_output, filename_output, progress_output, exposure_slider, comment_input]
        )

        index_input.submit(
            update_image,
            inputs=[index_input],
            outputs=[image_output, filename_output, progress_output, exposure_slider, comment_input]
        )

        exposure_slider.release(
            adjust_image_exposure,
            inputs=[exposure_slider],
            outputs=[image_output]
        )

        app.load(
            update_image,
            inputs=None,
            outputs=[image_output, filename_output, progress_output, exposure_slider, comment_input]
        )

    return app


if __name__ == "__main__":
    image_dir = "/labstorage/datasets/image/HDR+burst/20171106/results_20171023"
    output_file = "/exposure-value-annotation-tool/outputs/annotations.csv"
    
    tool = AnnotationTool(image_dir, output_file)
    app = create_ui(tool)
    app.launch(server_port=7860, server_name="0.0.0.0")