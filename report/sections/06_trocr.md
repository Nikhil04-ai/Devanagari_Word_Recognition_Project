# 6. Model C: TrOCR-style encoder-decoder

## 6.1 Method

TrOCR [2] is an encoder-decoder model. The **encoder** is an image Transformer: the word image is cut into 16×16 patches, which are embedded and processed with self-attention. The **decoder** is a text Transformer that generates the transcription one token at a time, attending to the encoder output through cross-attention; it is trained with teacher forcing (it sees the correct previous characters and predicts the next one) and, at prediction time, generates from a start token until an end token.

Unlike the CRNN, whose output at each image column does not depend on its own earlier outputs, the decoder is a character-level language model conditioned on the image, which can help with signs whose form depends on their neighbours.

## 6.2 Adaptation to Devanagari

We start from the public checkpoint `microsoft/trocr-small-handwritten`, which was trained on English text. Its visual encoder transfers to handwriting in general, but its 64,044-token English sub-word vocabulary does not fit Devanagari, so we **replace the decoder's token embeddings and output layer by a character vocabulary**: the 100 Unicode code points of our training split (`vocab.txt`, shared with the CRNN) plus four special tokens (`<pad>` = 0, `<bos>` = 1, `<eos>` = 2, `<unk>` = 3). Base letters, vowel signs and the halant are separate tokens, so a conjunct such as क्ष is the sequence क, ्, ष. The new embeddings are randomly initialised (normal, std 0.02); the encoder and all other decoder layers keep their pre-trained weights and are fine-tuned together with them. Labels are Unicode NFC code points, exactly as for the CRNN; the longest label has 23 characters, so sequences are limited to 32 tokens.

## 6.3 Training setup

| Setting | Value |
|---|---|
| Starting checkpoint | `microsoft/trocr-small-handwritten` |
| Parameters | 28.9 M (the original model has about 62 M; most of the difference is the 64k-token English embedding and output layer that we removed) |
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

Table: Training setup of Model C (TrOCR-style).


Before the full run the model was trained on 100 training images (`configs/trocr/overfit.yaml`) to check that it can memorise them: validation CER on those images reached 0.0000 at epoch 48 (49 epochs, 4.5 min), so the model, loss and decoding work end to end.

## 6.4 Results

On the test split (12,869 images, greedy decoding, evaluated once; `results/trocr_full_greedy/metrics.json`) the model reaches CER 6.47 %, WER 22.32 % and word accuracy 77.68 % (Section 7.1), after 89.7 min of training, and needs 70.6 ms per image (batch size 1, GPU). Training used all the data, but only 5 epochs. On the fixed 1,500-image validation subset (`results/trocr_full_greedy/training_log.csv`), the validation CER fell from 0.1917 after the first epoch (epoch 0) to 0.0870 at the last and best epoch (epoch 4), with validation WER 0.2833 and train loss 0.027. Validation CER was still falling at the last epoch, so the model was not fully converged after 5 epochs.

## 6.5 Observations

The comparison with the CRNN, including cost, is in Section 7.3.

- **Pre-training.** The model reached CER 0.19 after one epoch and 0.087 after five (validation), so English handwriting pre-training gave a quick start, although we did not run a from-scratch comparison and so cannot say how much of it is due to pre-training.
- **Input shape.** Stretching word images to a 384 × 384 square distorts long words.
