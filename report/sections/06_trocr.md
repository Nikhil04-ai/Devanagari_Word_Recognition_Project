# 6. Model C: TrOCR-style encoder-decoder

*Owner: Prabhat. Describe what `src/trocr/` actually does. Items marked (check) are planned: correct them to match the final code. Delete the italic notes.*

## 6.1 Method

TrOCR [2] is an encoder-decoder model. The **encoder** is an image Transformer: the word image is cut into 16×16 patches, which are embedded and processed with self-attention. The **decoder** is a text Transformer that generates the transcription one token at a time, attending to the encoder output through cross-attention; it is trained with teacher forcing and, at prediction time, generates from a start token until an end token.

## 6.2 Adaptation to Devanagari

We start from the public checkpoint `microsoft/trocr-small-handwritten`, which was trained on English text. Its text vocabulary is built for English, so we **replace the decoder's token embeddings and output layer by a character vocabulary** made of the Devanagari code points of our training split plus special tokens (padding, start, end, unknown) *(check)*. The encoder and the remaining decoder layers keep their pre-trained weights and are fine-tuned together with the new embeddings. Labels are Unicode NFC code points, exactly as for the CRNN.

## 6.3 Training setup

| Setting | Value |
|---|---|
| Starting checkpoint | `microsoft/trocr-small-handwritten` |
| Parameters | TODO *(read from the loaded model)* |
| Input size | TODO *(read from the processor config)* |
| Optimiser | AdamW, TODO learning rates (pre-trained layers vs new embeddings) |
| Schedule | warm-up then decay |
| Batch size | TODO (with gradient accumulation TODO) |
| Precision | mixed precision (fp16) |
| Decoding | greedy generation *(check)* |
| Training data used | TODO (all of the training split, or a subset of TODO images because of the time budget) |
| Epochs / time budget | TODO |
| Seed | 42 |

Before the full run the model was trained on 100 images until it reproduced them (CER = 0): TODO (result).

## 6.4 Results

*From `results/trocr_full_greedy/metrics.json`. If training used a reduced budget, say so here and in Section 8.*

| Metric | Value |
|---|---|
| CER | TODO |
| WER (word error) | TODO |
| Word accuracy | TODO |
| Training time (min) | TODO |
| Time per image (ms) | TODO |

*Add the learning curve (training loss, validation CER per epoch).*

## 6.5 Observations

*Two or three evidence-based points, for example: did pre-training on English handwriting help; how does it compare with the CRNN in error and in training cost (Section 7). TODO.*
