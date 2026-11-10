# Training on Kaggle (free GPU)

None of our laptops has a GPU, so full training runs on Kaggle. Every member has their own free quota (about 30 GPU hours a week), so the three models train at the same time.

## One-time setup

1. Create a Kaggle account and **verify it with your phone number**. GPU and internet access in notebooks need this.
2. Create a GitHub personal access token (GitHub → Settings → Developer settings → Personal access tokens) with read access to the repo. The repo is private until submission.
3. In Kaggle, open a new notebook → **Add-ons → Secrets** → add a secret named `GITHUB_TOKEN` with that token.
4. Notebook settings: **Accelerator = GPU (T4 or P100)**, **Internet = On**.

## Notebook cells

```python
# 1. code
from kaggle_secrets import UserSecretsClient
token = UserSecretsClient().get_secret("GITHUB_TOKEN")
!git clone https://{token}@github.com/Nikhil04-ai/Devanagari_Word_Recognition_Project.git
%cd Devanagari_Word_Recognition_Project
!git checkout nikhil-crnn          # your own branch
!pip install -q rapidfuzz pyyaml   # Kaggle already has torch, transformers, pandas and pillow
```

```python
# 2. data (IIIT-HW-Dev, about 1.8 GB; download time on Kaggle not measured yet)
!python scripts/download_iiit.py --out /kaggle/working/IIIT-HW-Hindi_v1.tar.gz
!mkdir -p data/raw/iiit_hw_dev
!tar -xzf /kaggle/working/IIIT-HW-Hindi_v1.tar.gz -C data/raw/iiit_hw_dev
!tar -xzf data/raw/iiit_hw_dev/HindiSeg.tar.gz -C data/raw/iiit_hw_dev   # the images are in a second archive
```

The files `data/processed/words/*.csv` and `vocab.txt` are in the repo; their `image_path` column points into `data/raw/iiit_hw_dev/`, so the images must be extracted exactly there.

```python
# 3. check the GPU, then the sanity run, then the full run
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
!python -m src.crnn.train --config configs/crnn/overfit.yaml
!python -m src.crnn.train --config configs/crnn/full.yaml
!python -m src.crnn.evaluate --ckpt checkpoints/crnn_full/best.pt
```

Prabhat and Ravi use their own module and config (`src.trocr.train`, `src.char_cnn.train`).

## Time budget

Every config has `max_minutes`. When it runs out, training stops cleanly and `best.pt` is kept, so a run always ends with a usable model. Watch the first epoch's time in the log and, if needed, lower `epochs` or the data size so a run fits its budget.

## Keeping your files

- Files in `/kaggle/working` are kept only when you press **Save Version → Save & Run All (Commit)**. After an interactive session ends they are lost.
- For long runs use **Save & Run All**: it runs in the background and survives closing the browser (limit about 12 hours per session).
- After the run, open the **Output** tab and download `checkpoints/.../best.pt` and the `results/` folder.
- If a session stops early, start a new one, restore `checkpoints/<run>/last.pt`, and run the same command with `--resume`. The `epochs` setting must be the same as in the original run.

## If something goes wrong

| Problem | Fix |
|---|---|
| `torch.cuda.is_available()` is False | Notebook settings → Accelerator → GPU; phone verification done? |
| `git clone` asks for a password | The secret name or token is wrong; the token needs read access to the repo |
| Out of memory | Lower `batch_size` in the config |
| Loss is `inf` or `nan` | Lower `lr`; check that no label is longer than the output length |
| Session disconnected | Use `--resume` (see above); Colab is a backup if the Kaggle quota is used up |
