# 6. Model C: TrOCR-style encoder-decoder

## 6.1 Method

TrOCR [2] is an encoder-decoder model. The **encoder** is an image Transformer: the word image is cut into 16×16 patches, which are embedded and processed with self-attention. The **decoder** is a text Transformer that generates the transcription one token at a time, attending to the encoder output through cross-attention; it is trained with teacher forcing (it sees the correct previous characters and predicts the next one) and, at prediction time, generates from a start token until an end token.

Unlike the CRNN (Section 5), which makes one prediction per image column and has no notion of context, the decoder is a character-level language model conditioned on the image. This can help with conjuncts and vowel signs, whose correct form depends on their neighbours, but it can also produce a plausible-looking word when the image is unclear.

## 6.2 Adaptation to Devanagari

We start from the public checkpoint `microsoft/trocr-small-handwritten`, which was trained on English text. Its visual encoder transfers to handwriting in general, but its 64,044-token English sub-word vocabulary does not fit Devanagari, so we **replace the decoder's token embeddings and output layer by a character vocabulary**: the 100 Unicode code points of our training split (`vocab.txt`, shared with the CRNN) plus four special tokens (`<pad>` = 0, `<bos>` = 1, `<eos>` = 2, `<unk>` = 3). Base letters, vowel signs and the halant are separate tokens, so a conjunct such as क्ष is the sequence क, ्, ष. The new embeddings are randomly initialised (normal, std 0.02); the encoder and all other decoder layers keep their pre-trained weights and are fine-tuned together with them. The embedding's padding index is moved to `<pad>`, because the English model used index 1, which is `<bos>` in our scheme. Labels are Unicode NFC code points, exactly as for the CRNN; the longest label has 23 characters, so sequences are limited to 32 tokens.

As a fallback in case the main approach fails to converge, the code also contains a smaller variant (`approach: custom_decoder`) that keeps the pre-trained encoder but replaces the decoder by a new 4-layer Transformer decoder trained from scratch.

## 6.3 Training setup

| Setting | Value |
|---|---|
| Starting checkpoint | `microsoft/trocr-small-handwritten` |
| Parameters | 28.8 M (the original model has about 62 M; most of the difference is the 64k-token English embedding and output layer that we removed) |
| Input size | 384 × 384 RGB (every word image is resized to this square by the TrOCR image processor) |
| Optimiser | AdamW, weight decay 0.01; learning rate 5e-5 for pre-trained layers and 3e-4 for the new embeddings and output layer |
| Schedule | 300 warm-up steps, then cosine decay |
| Batch size | 16, with gradient accumulation 2 (effective batch 32); gradient clipping 1.0 |
| Precision | mixed precision (fp16) |
| Decoding | greedy generation (up to 32 tokens) |
| Model selection | lowest validation CER (greedy) on a fixed random subset of 1,500 validation images, evaluated after every epoch; the test split is used once at the end |
| Training data used | all 69,853 training images (no subset) |
| Epochs / time budget | 5 epochs (about 90 min on a Colab GPU), well inside the 190 min cap; the best epoch was the last one (epoch 4) |
| Seed | 42 |


Before the full run the model was trained on 100 training images (`configs/trocr/overfit.yaml`) to check that it can memorise them: validation CER on those images reached 0.0000 at epoch 48 (49 epochs, 4.5 min), so the model, loss and decoding work end to end.

## 6.4 Results

Test split (12,869 images, greedy decoding, evaluated once; `results/trocr_full_greedy/metrics.json`). Training used the full budget of data, but only 5 epochs.

| Metric | Value |
|---|---|
| CER | 6.47 % |
| WER (word error) | 22.32 % |
| Word accuracy | 77.68 % |
| Training time (min) | 89.7 |
| Time per image (ms) | 70.6 (batch size 1, GPU) |

Learning curve (`log.csv`, validation on a fixed 1,500-image subset):

| Epoch | Train loss | Val CER | Val WER |
|---|---|---|---|
| 0 | 1.142 | 0.1917 | 0.5260 |
| 1 | 0.156 | 0.1392 | 0.4187 |
| 2 | 0.085 | 0.0984 | 0.3273 |
| 3 | 0.046 | 0.0897 | 0.2900 |
| 4 | 0.027 | 0.0870 | 0.2833 |

Validation CER was still falling at the last epoch, so the model was not fully converged after 5 epochs.

## 6.5 Observations

- **Accuracy vs. CRNN.** On the same test split TrOCR reaches CER 6.47 % against 6.59 % for the CRNN (Section 5), a difference of 0.12 points that is within what one seed could change. The gap is larger at word level: WER 22.32 % vs 26.05 % (word accuracy 77.68 % vs 73.95 %). Both are single runs, so we call TrOCR slightly better at word level and equal at character level, not clearly better.
- **Cost.** TrOCR has 28.9 M parameters against 7.7 M (3.7 times more), took 89.7 min against 82.1 min to train, and needs 70.6 ms per image against 3.6 ms (about 20 times slower). For a similar CER the CRNN is far cheaper.
- **Pre-training.** The model reached CER 0.19 after one epoch and 0.087 after five (validation), so English handwriting pre-training gave a quick start, although we did not run a from-scratch comparison and so cannot say how much of it is due to pre-training.

Limitations to state: the pre-training data was English, so the transfer to Devanagari is mostly visual; stretching word images to a 384 × 384 square distorts long words; one run with one seed; training stopped after 5 epochs while validation CER was still improving.
