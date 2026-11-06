# 🩺 Bangla Prescription Analyzer

Recognise **handwritten medicine names** from Bangladeshi doctors' prescriptions and explain each medicine **in Bangla** — its generic name, what it is used for, common side effects and cautions.

```
handwritten word image ──► CNN classifier (78 medicines) ──► generic name ──► Bangla knowledge base ──► Bangla report
      (Napa)                  EfficientNet / ConvNeXt          (Paracetamol)     data/bangla_generic_info.json    "ব্যবহার: জ্বর, ব্যথা…"
```

> ⚠️ **Educational / research project only.** Model output can be wrong. It never gives dosages. Always confirm with a doctor or pharmacist.

## Dataset
[Doctor's Handwritten Prescription BD dataset](https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset) — 4,680 cropped word images, 78 medicine brands (60 images each), 15 generic names, split 60 / 20 / 20 into Training / Validation / Testing. The dataset is **not** included in this repo; the notebook downloads it.

## Quick start (Google Colab)
1. Open `notebooks/train_colab.ipynb` in Colab (GPU runtime).
2. Run all cells: download data → EDA → train → evaluate → Bangla analysis → Gradio demo.

## Quick start (local / command line)
```bash
pip install -r requirements.txt
python -m src.train     --data_root <dataset_folder> --out_dir runs/effb0 --model efficientnet_b0
python -m src.evaluate  --data_root <dataset_folder> --run_dir runs/effb0
python -m src.knowledge_base check --label_map runs/effb0/label_map.json
python -m src.analyzer  --run_dir runs/effb0 --image some_word.png
python app/app.py       --run_dir runs/effb0
```

## Project structure
```
bangla-prescription-analyzer/
├── notebooks/train_colab.ipynb     # end-to-end Colab notebook
├── src/
│   ├── data.py                     # dataset discovery, loading, augmentation
│   ├── model.py                    # timm backbone factory + checkpoint I/O
│   ├── train.py                    # training loop (AMP, cosine LR, early stopping)
│   ├── evaluate.py                 # test metrics, confusion matrix, error list
│   ├── knowledge_base.py           # generic -> Bangla description (+ builder from big medicine DB)
│   └── analyzer.py                 # image -> medicine -> Bangla report
├── app/app.py                      # Gradio web demo
├── data/bangla_generic_info.json   # curated Bangla medicine descriptions
├── docs/GUIDE.md                   # full step-by-step project guide
├── results/                        # (you add) metrics & plots from your runs
├── requirements.txt
└── .gitignore
```

## Results
| Model | Img size | Test top-1 | Test top-5 | Macro-F1 | Generic acc |
|---|---|---|---|---|---|
| efficientnet_b0 | 224 | _fill in_ | | | |
| convnext_tiny | 224 | _fill in_ | | | |

## Citation
If you use the dataset, cite: A. R. Mia et al., *"A Deep Neural Network Approach with Pioneering Local Dataset to Recognize Doctor's Handwritten Prescription in Bangladesh,"* iCACCESS 2024.

See **[docs/GUIDE.md](docs/GUIDE.md)** for the complete guideline.
