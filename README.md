# SSDCA

### Dual Cross-Attention Siamese Transformer for Rectal Tumor Regrowth Assessment in Watch-and-Wait Endoscopy

![Selling Figure](Figures/Selling%20Figure.jpg)

Increasing evidence supports watch-and-wait (WW) surveillance for patients with rectal cancer who show clinical complete response (cCR) at restaging following total neoadjuvant treatment (TNT). However, accurate methods to early detect local regrowth (LR) from follow-up endoscopy images during WW are essential to manage care and prevent distant metastases. We developed a Siamese Swin Transformer with Dual Cross-Attention (SSDCA) to combine longitudinal endoscopic images at restaging and follow-up and distinguish cCR from LR. SSDCA leverages pretrained Swin Transformers to extract domain agnostic features and enhance robustness to imaging variations. Dual cross attention is implemented to emphasize features from the paired scans without requiring any spatial alignment to predict response.

## Architecture

![Architecture](Figures/Architecture.jpg)

A pair of longitudinal images (restaging and follow-up) are independently processed by a shared Swin Transformer encoder. The Stage 4 features from both images are passed through a Dual Cross-Attention (DCA) module, which attends to semantically relevant features in one image at the spatial locations of the other, producing attention-refined residuals. These are pooled, concatenated, and passed through a classification head to predict cCR or LR.

## Training

[`Example.sh`](Example.sh) shows how to launch a training run via [`main.py`](main.py):

```bash
python main.py --num_classes 2 \
--model_config Swin_S \
--num_epochs 30 \
--batch_size 8 \
--learning_rate 2e-5 \
--json_fold fold_2.json \
--sampler balanced \
--save_dir "finetunedmodel_allSameCrop_115" \
--model_naming Siam-Swin_S_Fold2_TemporalChange_115 \
--seed 115
```

- `--model_config` selects the model/architecture (e.g. `Swin_S` for the Siamese Swin model used by SSDCA).
- `--json_fold` points to a JSON file defining the train/val/test split for that fold.
- `--sampler balanced` enables a balanced sampler to handle class imbalance.
- `--save_dir` and `--model_naming` control where checkpoints, logs, and TensorBoard files are written.
- `--seed` fixes the random seed for reproducibility.

Checkpoints, logs, and TensorBoard summaries are written to `<save_dir>/<model_naming>`.

## Installation

This project requires Python with PyTorch and the following packages:

```bash
pip install torch torchvision timm tensorboard scikit-learn pandas numpy matplotlib openpyxl tqdm pillow
```

A CUDA-capable GPU is recommended for training.
