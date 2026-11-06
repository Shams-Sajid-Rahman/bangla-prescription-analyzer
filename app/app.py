"""Gradio web demo.

Colab:   !python app/app.py --run_dir /content/drive/MyDrive/rx_runs/effb0 --share
Local:   python app/app.py --run_dir runs/effb0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402

from src.analyzer import PrescriptionAnalyzer  # noqa: E402


def build_ui(analyzer: PrescriptionAnalyzer):
    def run(img):
        if img is None:
            return "একটি ছবি আপলোড করুন।", {}
        res = analyzer.analyze(img, top_k=5)
        scores = {res["prediction"]["medicine"]: res["prediction"]["confidence"]}
        scores |= {a["medicine"]: a["confidence"] for a in res["alternatives"]}
        return analyzer.to_bangla_report(res), scores

    with gr.Blocks(title="বাংলা প্রেসক্রিপশন বিশ্লেষক") as demo:
        gr.Markdown("# 🩺 বাংলা প্রেসক্রিপশন বিশ্লেষক\n"
                    "ডাক্তারের হাতে লেখা **একটি ওষুধের নামের** ক্রপ করা ছবি আপলোড করুন।")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="হাতে লেখা ওষুধের নাম")
                btn = gr.Button("বিশ্লেষণ করুন", variant="primary")
            with gr.Column():
                out_md = gr.Markdown()
                out_lbl = gr.Label(num_top_classes=5, label="সম্ভাব্য নাম")
        btn.click(run, inp, [out_md, out_lbl])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--share", action="store_true", help="Public link (needed in Colab)")
    a = ap.parse_args()
    build_ui(PrescriptionAnalyzer(a.run_dir)).launch(share=a.share)
