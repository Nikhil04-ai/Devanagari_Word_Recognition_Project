# 5. Model B: CRNN with CTC

*Owner: Nikhil. Architecture and setup below are taken from the code (`src/crnn/`). Results and observations are TODO until the full run finishes; delete the italic notes.*

## 5.1 Method

The CRNN of Shi et al. [1] reads a word image as a sequence. A convolutional network turns the image into a sequence of feature columns, a bidirectional LSTM lets every column use the context to its left and right, and a linear layer gives, for every column, a probability over the characters plus a special **blank** symbol. We train with the Connectionist Temporal Classification (CTC) loss [5], which needs no information about where each character is in the image: it sums over all alignments of the column sequence to the label.

At prediction time, **greedy decoding** takes the most likely symbol in every column, merges consecutive repeats and removes blanks. A blank between two equal symbols keeps both, for example `क क - म - - ल ल` gives `कमल`.

## 5.2 Architecture

| Stage | Layers | Output shape (height × width/4 columns) |
|---|---|---|
| Input | grayscale word image, height 64 | 1 × 64 × W |
| Conv block 1 | Conv 3×3 (64), BatchNorm, ReLU, MaxPool 2×2 | 64 × 32 × W/2 |
| Conv block 2 | Conv 3×3 (128), BatchNorm, ReLU, MaxPool 2×2 | 128 × 16 × W/4 |
| Conv block 3 | 2 × [Conv 3×3 (256), BatchNorm, ReLU], MaxPool 2×1 | 256 × 8 × W/4 |
| Conv block 4 | 2 × [Conv 3×3 (512), BatchNorm, ReLU], MaxPool 2×1 | 512 × 4 × W/4 |
| Sequence | reshape to T = W/4 columns of 2048 features, Linear to 256 | T × 256 |
| Recurrent | 2-layer bidirectional LSTM, 256 hidden units, dropout 0.3 | T × 512 |
| Output | Linear to (characters + blank), log-softmax | T × (C + 1) |

The network has about 7.7 million parameters (7,691,526 for a vocabulary of 70 characters; the exact count is in `metrics.json`). Because the width is divided by 4, an image of width 128 gives T = 32 time steps. CTC needs at least as many time steps as the label has characters plus one blank between equal neighbours, so we check that every label fits.

## 5.3 Training setup

| Setting | Value |
|---|---|
| Loss | CTC, blank = 0, computed in float32, `zero_infinity=True` |
| Optimiser | AdamW, learning rate 1e-3, weight decay 1e-4 |
| Schedule | One-cycle (15 % warm-up), gradient clipping at 5 |
| Batch size | 64 |
| Precision | mixed precision (fp16) on GPU |
| Model selection | lowest validation CER; early stopping after 8 epochs without improvement |
| Time budget | TODO minutes on a Kaggle GPU (TODO epochs completed) |
| Augmentation | none |
| Seed | 42 |

Before the full run, the model was trained on 100 training images until it reproduced them exactly (CER = 0), to confirm the implementation: TODO (result).

## 5.4 Results

*Fill from `results/crnn_full_greedy/metrics.json`. Test set, evaluated once.*

| Metric | Value |
|---|---|
| CER | TODO |
| WER (word error) | TODO |
| Word accuracy | TODO |
| Parameters | TODO |
| Training time (min) | TODO |
| Time per image (ms) | TODO |

*Add the learning curve (`checkpoints/crnn_full/log.csv`): training and validation loss, and validation CER per epoch.*

## 5.5 Observations

*Three short, evidence-based points, for example: how fast validation CER fell and where it flattened; whether seen words are much easier than unseen words; which kinds of error dominate (see Section 7). TODO.*

## 5.6 Limitations

Training used a limited time budget on a free GPU, no augmentation and greedy decoding only, and one run with one seed, so the numbers carry no confidence interval.
