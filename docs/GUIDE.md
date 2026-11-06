# Complete Step-by-Step Guide — Bangla Prescription Analyzer

This guide takes you from an empty GitHub repo to a trained model, an evaluated result, a Bangla description engine and a live web demo. Every phase ends with a **git checkpoint** so you can push manually.

---

## Phase 0 — Understand what you are building

**Important fact about the dataset:** the *Doctor's Handwritten Prescription BD* dataset contains **English (Latin-script) medicine brand names** — *Napa, Fexo, Montair, Sergel…* — handwritten by Bangladeshi doctors. It is not Bangla-script handwriting. So the project is:

| Stage | Input | Output | Language |
|---|---|---|---|
| 1. Recognition (deep learning) | cropped handwritten word image | one of 78 medicine names | English brand name |
| 2. Mapping | medicine name | generic (active ingredient), from the dataset CSV | English |
| 3. Description (knowledge base) | generic name | uses, side effects, cautions | **Bangla** |

This is the realistic way to build a "Bangla medical description analyzer" with this data: the model reads the handwriting, the Bangla part is the explanation for patients. (If you later want to read Bangla-script handwriting too — e.g. "খাবারের পরে" instructions — see Phase 9.)

**Dataset facts**
- 4,680 images, 78 medicine classes × 60 images, 15 generic names
- Already split: Training 3,120 / Validation 780 / Testing 780 (stratified)
- Each split folder has an image sub-folder + a CSV (`IMAGE`, `MEDICINE_NAME`, `GENERIC_NAME`)
- Several brands share one generic (e.g. *Napa, Ace, Aceta, Tamen → Paracetamol*). This matters: confusing two Paracetamol brands is much less harmful than confusing Paracetamol with Clonazepam — that's why `evaluate.py` reports **generic accuracy** too.

**Reference result:** a 2025 study on the same dataset got ~88% test accuracy with MobileNet. Aim to beat that with modern transfer learning.

---

## Phase 1 — Create the GitHub repository (on your PC)

1. On GitHub: **New repository** → name `bangla-prescription-analyzer` → Public → **don't** add README/.gitignore (we have them) → Create.
2. Unzip the project I gave you, then in a terminal:

```bash
cd bangla-prescription-analyzer
git init
git branch -M main
git add .
git commit -m "Initial project structure: data loader, training, evaluation, Bangla KB, Colab notebook"
git remote add origin https://github.com/<YOUR_USERNAME>/bangla-prescription-analyzer.git
git push -u origin main
```

3. Edit two placeholders and push again:
   - `notebooks/train_colab.ipynb` → `REPO_URL` and the "Open in Colab" badge: replace `YOUR_GITHUB_USERNAME`.
   ```bash
   git add notebooks/train_colab.ipynb
   git commit -m "Set repo URL in Colab notebook"
   git push
   ```

**Rules for this repo**
- Never commit the dataset, `kaggle.json`, or `*.pt` weights — `.gitignore` already blocks them.
- Commit small results (metrics JSON, plots) into `results/`.
- Optional: add a `LICENSE` (MIT is common) via GitHub → Add file → Create new file → `LICENSE` → choose template.

---

## Phase 2 — Set up Google Colab

1. Go to colab.research.google.com → **File → Open notebook → GitHub** → paste your repo URL → open `notebooks/train_colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU**.
3. Run Step 1 (`nvidia-smi`) and Step 2 (mount Drive). All checkpoints go to `MyDrive/rx_runs/...` so a disconnect doesn't lose your model.
4. Step 3 clones your repo and installs `timm gradio kagglehub` (Colab already has PyTorch).

> Workflow tip: edit code on your PC → `git push` → in Colab re-run the clone cell (it does `git pull`). Don't edit code inside Colab, or you'll have two diverging copies.

---

## Phase 3 — Get the data and explore it (EDA)

1. Notebook Step 4 downloads the dataset with `kagglehub`. If it asks for credentials: kaggle.com → Settings → API → **Create New Token**; in Colab open the 🔑 Secrets panel, add `KAGGLE_USERNAME` and `KAGGLE_KEY`, toggle notebook access.
2. Step 5 (EDA) shows:
   - image counts per split and class (should be 40/10/10 per class)
   - which brands map to which generic
   - a grid of sample images and image size statistics
3. Write down observations in `docs/EDA_notes.md` (varying widths, faint ink, overlapping strokes…). These go into your report.

`src/data.py` finds the split folders and CSV columns automatically, so naming differences between dataset versions don't break anything.

**Git checkpoint:** `git commit -m "Add EDA notes"`

---

## Phase 4 — Train the baseline model

Notebook Step 6:
```bash
python -m src.train --data_root "$DATA_ROOT" --out_dir "$RUN_DIR" \
    --model efficientnet_b0 --img_size 224 --epochs 40 --batch_size 64 --lr 3e-4
```

What the training script does (and why — useful for your report/viva):

| Choice | Why |
|---|---|
| ImageNet-pretrained EfficientNet-B0 (via `timm`) | only 40 images per class → transfer learning is essential |
| Pad to square with white, then resize | word images are wide; stretching distorts letters |
| Grayscale → 3 channels | colour carries no information, pretrained nets expect 3 channels |
| Augment: ±5° rotation, shear, small scale/shift, brightness/contrast, blur, random erasing | simulates different pens, scanning and writers |
| **No flips** | a mirrored word is not the same word |
| AdamW + warm-up + cosine LR, label smoothing 0.1 | stable fine-tuning, less over-confidence |
| Mixed precision (AMP) | ~2× faster on T4 |
| Early stopping on val top-1 (patience 8), best checkpoint saved | prevents over-fitting, keeps the best model |

Outputs in `RUN_DIR`: `best.pt`, `last.pt`, `label_map.json`, `config.json`, `history.json`. The next cell plots loss/accuracy curves — check that the val curve does not drift away from train (over-fitting).

---

## Phase 5 — Evaluate and analyse errors

Notebook Step 7:
```bash
python -m src.evaluate --data_root "$DATA_ROOT" --run_dir "$RUN_DIR"
```
Produces `test_metrics.json` (top-1, top-5, macro-F1, generic accuracy), `classification_report.csv` (per-class precision/recall), `confusion_matrix.png`, `misclassified.csv`.

Error analysis checklist:
- Which pairs get confused most? (e.g. *Fexo ↔ Fexofast*, *Montair ↔ Montex* — similar spellings)
- Are mistakes mostly within the same generic? (generic accuracy ≫ top-1 is good news)
- Look at the actual misclassified images — are some mislabelled or unreadable even to humans?

Use the **Testing split only once per final model**. Tune on Validation.

**Git checkpoint:** copy metrics/plots to `results/<model_name>/` (notebook Step 11), then
`git add results && git commit -m "Baseline EfficientNet-B0 results" && git push`

---

## Phase 6 — Improve the model (experiments)

Change one thing at a time, new `RUN_DIR` per experiment, record everything in the README results table.

| # | Experiment | Command change |
|---|---|---|
| 1 | Bigger/modern backbone | `--model convnext_tiny` or `efficientnet_b3` (lower `--batch_size 32`) |
| 2 | Lightweight for mobile | `--model mobilenetv3_large_100` |
| 3 | Image size | `--img_size 256` or `288` |
| 4 | Learning rate | `--lr 1e-4` / `5e-4` |
| 5 | More data | add RxHandBD (see dataset list) for overlapping medicine names |
| 6 | Ensemble | average softmax of 2–3 different models |
| 7 | Test-time augmentation | average predictions over small shifts/scales |
| 8 | Merge train+val, retrain | final model after hyper-parameters are fixed |

Ideas worth trying if you want a stronger research contribution:
- **CRNN + CTC** (sequence model that reads letters) — can recognise names *outside* the 78 classes, combined with a lexicon of all Bangladeshi brand names for spelling correction.
- **TrOCR fine-tuning** (`microsoft/trocr-base-handwritten`) — transformer OCR; compare against the classifier.
- **Hierarchical loss**: also predict the generic as a second head.

---

## Phase 7 — Build the Bangla knowledge base

1. `data/bangla_generic_info.json` already contains Bangla entries (name, class, uses, side effects, cautions — **no dosages**) for the common generics in this dataset.
2. Check coverage after training:
   ```bash
   python -m src.knowledge_base check --label_map "$RUN_DIR/label_map.json"
   ```
   It lists any generic from the dataset without an entry. Add those to the JSON using the same fields.
3. Optional enrichment from a big database (notebook Step 8): downloads `generic.csv` from the *Medicine Dataset of Bangladesh* (Hugging Face, MIT) and writes `data/kb_extra.json` with English descriptions; `--translate` adds machine translation to Bangla with BanglaT5. **Machine-translated medical text must be reviewed by a human** — curated entries always take priority.
4. Best practice: ask a pharmacist / medical student to review the Bangla text and credit them in the README.

**Git checkpoint:** `git commit -am "Complete Bangla KB coverage for all generics"`

---

## Phase 8 — Inference and web demo

- Command line: `python -m src.analyzer --run_dir "$RUN_DIR" --image word.png` → Bangla report with confidence, generic, uses, side effects, cautions, alternatives and a disclaimer. Below 60% confidence it warns the user.
- Gradio (notebook Step 10): `python app/app.py --run_dir "$RUN_DIR" --share` → public link for 72 h. Take screenshots for the README.
- Permanent hosting (free): **Hugging Face Spaces** → create a Gradio Space, upload `app/`, `src/`, `data/`, `requirements.txt`, and `best.pt` + `label_map.json` (Spaces supports large files). Set the app to `--run_dir` pointing at those files.
- Model weights for GitHub users: create a **GitHub Release** (Releases → Draft new release → attach `best.pt`), and link it in the README.

---

## Phase 9 — Extensions (to make it a full system)

1. **Full prescription page → words.** The current model needs cropped words. Add a text-detection step (e.g. docTR, EasyOCR or PaddleOCR detectors) to find word boxes on a whole prescription photo, then classify each crop; reject low-confidence crops as "not a medicine".
2. **Bangla-script handwriting.** Doctors also write instructions in Bangla ("১+০+১", "খাবারের পরে"). Train a second recogniser on Bangla handwriting datasets such as BN-HTRd or BanglaLekha-Isolated.
3. **Symptom → disease in Bangla.** Use the Bangla Symptoms-Disease dataset to let users type symptoms in Bangla and get likely conditions (clearly marked as non-diagnostic).
4. **Mobile app.** Export to ONNX/TFLite (`torch.onnx.export`) and use MobileNetV3.

---

## Phase 10 — Finish the repository

- README: fill the results table, add confusion-matrix image, Gradio screenshot, Colab badge, dataset citation, disclaimer.
- Keep `docs/` for EDA notes, experiment log, and a short report (problem, data, method, results, limitations).
- Tag a version: `git tag v1.0 && git push --tags`.

### Suggested commit sequence
```
1  Initial project structure
2  Set repo URL in Colab notebook
3  Add EDA notes
4  Baseline EfficientNet-B0 results
5  ConvNeXt-Tiny experiment
6  Complete Bangla KB coverage
7  Gradio demo screenshots + README results
8  v1.0
```

---

## Recommended additional datasets

| Dataset | What it adds | Use for |
|---|---|---|
| **RxHandBD** (Mendeley Data, 2026) — 5,578 handwritten prescription word images, 80/20 split, CSV labels | more handwriting from Bangladeshi prescriptions | extra training data for overlapping medicine names; external test set to prove generalisation |
| **Medicine Dataset of Bangladesh** (Hugging Face `ArnobBot/Medicine-Dataset-of-Bangladesh`, MIT) — medicine, generic, indication, drug class, manufacturer CSVs | thousands of brands → generic + descriptions | enlarge the knowledge base; brand lexicon for spelling correction |
| **WSAyan/medicinedb** (GitHub) — SQLite/CSV/JSON of Bangladeshi brands, generics, indications, pregnancy category | alternative medicine database | same as above, cross-checking |
| **Medicinal Products in Bangladesh** (Mendeley Data, Univ. of Dhaka) — generic & brand names, strengths, manufacturers | official-style brand↔generic list | verifying brand→generic mapping |
| **Bangla Symptoms-Disease dataset** (Mendeley Data, KUET) — 758 symptom-disease relations, 85 diseases, in Bangla | Bangla medical text | Phase 9 symptom analyser |
| **BN-HTRd / BanglaLekha-Isolated** | Bangla-script handwriting | Phase 9 Bangla instruction reader |

When merging another handwriting dataset: normalise labels (case, spaces), keep only classes that exist in both (or add new classes deliberately), and **never** let the same image/writer appear in both train and test.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | lower `--batch_size` (32/16) or `--img_size` |
| `Could not find split folder(s)` | print the tree: `!find "$DATA_ROOT" -maxdepth 3 -type d` and pass the folder that contains Training/Validation/Testing |
| Colab disconnected mid-training | weights are in Drive; restart from Step 1, the best checkpoint is kept |
| Val accuracy stuck low | confirm a pretrained model is used (no `--no_pretrained`), try `--lr 1e-4` |
| Bangla text shows boxes in matplotlib | only affects plots; reports in Markdown/Gradio render Bangla fine |
| `git push` rejected (large file) | you committed a `.pt` file: `git rm --cached file.pt`, commit, push |
