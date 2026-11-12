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
| Training data used | TODO (all 69,853 training images, or a random subset of TODO images because of the time budget) |
| Epochs / time budget | TODO epochs, time cap 190 min |
| Seed | 42 |

Before the full run the model was trained on 100 images until it reproduced them: TODO (result of `configs/trocr/overfit.yaml`: epochs needed and final CER).

## 6.4 Results

*Fill from `results/trocr_full_greedy/metrics.json`. If training used a reduced budget, say so here and in Section 8.*

| Metric | Value |
|---|---|
| CER | TODO |
| WER (word error) | TODO |
| Word accuracy | TODO |
| Training time (min) | TODO |
| Time per image (ms) | TODO |

*Add the learning curve (training loss and validation CER per epoch, from `checkpoints/trocr_full/log.csv`).*

## 6.5 Observations

*Two or three evidence-based points, each tied to a number in Section 7: did pre-training on English handwriting help; how does the error compare with the CRNN, and at what cost in parameters, training time and time per image. TODO.*

Limitations to state: the pre-training data was English, so the transfer to Devanagari is mostly visual; stretching word images to a 384 × 384 square distorts long words; one run with one seed.
