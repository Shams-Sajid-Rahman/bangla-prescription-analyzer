# 🩺 Bangla Prescription Analyzer

Recognise **handwritten medicine names** from Bangladeshi doctors' prescriptions and explain each medicine **in Bangla**: its generic name, what it is used for, common side effects and cautions.

```
handwritten word image ──► CNN classifier (78 medicines) ──► generic name ──► Bangla knowledge base ──► Bangla report
      (Renova)             ConvNeXt-Small, 2-model ensemble     (Paracetamol)    data/bangla_generic_info.json     "ব্যবহার: জ্বর, ব্যথা…"
```


> ⚠️ **Educational / research project only.** Model output can be wrong. The app never gives dosages. Always confirm with a doctor or pharmacist.

## Highlights

- **99.16 % top-1 / 99.86 % top-5** on the cleaned test set (708 of 714 images correct) with an ensemble of two ConvNeXt-Small models.
- **99.58 % generic-name accuracy**: when the model picks the wrong brand, it usually still lands on the right active ingredient.
- A **label audit** (5-fold cross-validation) found mislabelled images in the public dataset; they were corrected or removed before the final training.
- A **Gradio web app** returns the medicine, its generic name, the top-5 alternatives with confidence, and a Bangla description with a safety warning.

## Dataset

[Doctor's Handwritten Prescription BD dataset](https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset): 4,680 cropped word images of **78 medicine brand names** (60 images per brand, 15 generic names). The original split is 40 / 10 / 10 images per brand for Training / Validation / Testing.

Note that the words are **English-script brand names** (for example *Napa, Fexo, Montair*) handwritten by Bangladeshi doctors. They are not Bangla-script handwriting. The Bangla part of this project is the explanation shown to the user.

The dataset is **not** included in this repository.

**Label cleaning.** Suspected label errors were found with an out-of-fold check (5-fold cross-validation with EfficientNet-B0 on the training set). The fixes are stored in [`data/label_fixes.csv`](data/label_fixes.csv) and applied automatically when the data is loaded: 12 images relabelled and 102 dropped (36 validation, 66 test).

| Split | Images after cleaning |
|---|---|
| Train | 3,120 |
| Validation | 744 |
| Test | 714 |

## Results

All numbers are on the cleaned **test set (714 images)**.

| Model | Trained on | Top-1 | Top-5 | Macro-F1 | Generic acc. |
|---|---|---|---|---|---|
| EfficientNetV2-S | train | 96.08 % | 99.30 % | 0.958 | 97.34 % |
| ConvNeXt-Small | train | 97.90 % | 99.72 % | 0.979 | 98.74 % |
| ConvNeXt-Small + TTA | train | 98.32 % | – | – | – |
| ConvNeXt-Small, seed 42 + TTA | train + val | 98.88 % | – | – | – |
| ConvNeXt-Small, seed 7 + TTA | train + val | 98.74 % | – | – | – |
| **Ensemble (seed 42 + seed 7) + TTA** | train + val | **99.16 %** | **99.86 %** | – | **99.58 %** |

*TTA = test-time augmentation: the image is shifted by ±6 pixels horizontally and vertically and the predictions are averaged (5 views per model).*

What the experiments showed:

- **ConvNeXt-Small beat EfficientNetV2-S** by about 2 points. Averaging the two architectures was not better than ConvNeXt alone, so EfficientNetV2 was left out of the final model.
- **Training on train + validation** and **averaging two seeds** gave the best result.
- **Be careful with small differences.** With 714 test images, the 99.16 % ensemble has an approximate 95 % confidence interval of 98.2 % to 99.6 %. The gap between one model and the ensemble is only 2 images.

## How it works

**Model.** A ConvNeXt-Small backbone (ImageNet-pretrained, via [timm](https://github.com/huggingface/pytorch-image-models)) with a new 78-class head.

**Input.** Each word is converted to grayscale, resized to fit **128 × 384** while keeping its aspect ratio, and padded with white. Wide words are not stretched.

**Training recipe.**

- AdamW (learning rate 1e-4, weight decay 0.05), 2 warm-up epochs, then cosine decay
- Label smoothing 0.1, Mixup / CutMix, exponential moving average of the weights (0.999)
- Strong augmentation: ink thickness, elastic distortion, perspective, small rotations and shifts, brightness and contrast, blur, random erasing. **No flips**, because a mirrored word is a different word.
- Mixed precision, batch size 32, 40 epochs, one Tesla T4 (about 55 seconds per epoch)

**Inference.** Softmax probabilities are averaged over the TTA views and over the two models. The generic name of the top prediction is looked up in the Bangla knowledge base. If the top confidence is below 60 %, the app shows a warning and asks the user to check the alternatives with a pharmacist.

## Quick start

```bash
git clone https://github.com/Shams-Sajid-Rahman/bangla-prescription-analyzer.git
cd bangla-prescription-analyzer
pip install -r requirements.txt
```

**Train** the two final models (same settings as in the results table):

```bash
python -m src.train --data_root <dataset_folder> --out_dir runs/FINAL_convnext_s42 \
    --model convnext_small --input_hw 128 384 --aug strong --mixup --ema 0.999 \
    --trainval --epochs 40 --batch_size 32 --lr 1e-4 --seed 42

python -m src.train --data_root <dataset_folder> --out_dir runs/FINAL_convnext_s7 \
    --model convnext_small --input_hw 128 384 --aug strong --mixup --ema 0.999 \
    --trainval --epochs 40 --batch_size 32 --lr 1e-4 --seed 7
```

**Evaluate** a single run on the test set (no TTA):

```bash
python -m src.evaluate --data_root <dataset_folder> --run_dir runs/FINAL_convnext_s42
```

**Analyze one image** with the ensemble:

```bash
python -m src.analyzer --run_dir runs/FINAL_convnext_s42 runs/FINAL_convnext_s7 --image some_word.png
```

**Run the web app** (pass one run folder, or several to average them; add `--share` for a public link in Colab or Kaggle; add `--no_tta` for faster, slightly less accurate answers):

```bash
python app/app.py --run_dir runs/FINAL_convnext_s42 runs/FINAL_convnext_s7
```

Model weights are not stored in this repository, because each file is about 190 MB.

## Project structure

```
bangla-prescription-analyzer/
├── app/app.py                      # Gradio web demo (one or more models)
├── src/
│   ├── data.py                     # dataset discovery, label fixes, augmentation
│   ├── model.py                    # timm backbone factory + checkpoint I/O
│   ├── train.py                    # training loop (AMP, EMA, Mixup, warm-up + cosine LR)
│   ├── evaluate.py                 # test metrics, confusion matrix, error list
│   ├── knowledge_base.py           # generic name -> Bangla description
│   └── analyzer.py                 # image -> medicine -> Bangla report (ensemble + TTA)
├── data/
│   ├── bangla_generic_info.json    # curated Bangla medicine descriptions
│   └── label_fixes.csv             # relabelled / dropped images from the label audit
├── notebooks/train_colab.ipynb     # end-to-end notebook
├── docs/GUIDE.md                   # step-by-step project guide
├── results/                        # screenshots and metrics
└── requirements.txt
```

## Limitations

- **Closed vocabulary.** The model knows only 78 brand names. Any other word, or a non-medicine word, is still forced into one of those 78 classes.
- **Single words only.** The input must be one tightly cropped word. A full prescription page is not supported yet.
- **One dataset.** The model is trained and tested on a single public dataset. Accuracy on new writers, pens, paper and camera conditions has not been measured and is likely lower.
- **Cleaned test set.** Removing 66 test images and fixing labels means the numbers above are not directly comparable with results reported on the original split.
- **Slightly optimistic test score.** The test set was also used to compare architectures and ensembles, so 99.16 % is likely a little high.
- **Confidence is soft.** Label smoothing and Mixup make the model less sure of itself, so a correct answer may show only about 70 % confidence.
- **Medical content.** The Bangla descriptions are general information and should be reviewed by a qualified pharmacist before any real-world use. The app never gives dosages.

## Citation

If you use the dataset, please cite: A. R. Mia et al., *"A Deep Neural Network Approach with Pioneering Local Dataset to Recognize Doctor's Handwritten Prescription in Bangladesh,"* iCACCESS 2024.

See **[docs/GUIDE.md](docs/GUIDE.md)** for the complete project guide.
