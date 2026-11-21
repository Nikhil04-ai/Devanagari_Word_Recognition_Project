# 4. Model A: CNN character classifier

## 4.1 Task and data

The CNN classifies one 32×32 grayscale image of a handwritten Devanagari consonant or digit into one of 46 classes (DHCD, Section 3.1): 36 consonants, including the three conjuncts क्ष, त्र and ज्ञ, and the ten digits ० to ९. Of the 78,200 training images, 10 % (7,820, stratified by class, seed 42) are held out for validation and model selection. The model is trained on the remaining 70,380. The 13,800 official test images (300 per class) are used once, at the end.

## 4.2 Architecture and training

| Item | Value |
|---|---|
| Convolution blocks | 4 × [Conv 3×3, BatchNorm, ReLU, Conv 3×3, BatchNorm, ReLU, MaxPool 2×2], channels 32, 64, 128, 256; 32×32 input → 2×2 feature map |
| Head | global average pooling, dropout 0.4, linear layer to 46 classes |
| Loss | cross-entropy with label smoothing 0.1 |
| Optimiser, schedule | Adam, learning rate 1e-3, weight decay 1e-4, cosine schedule over 15 epochs, batch 128, mixed precision on the GPU |
| Augmentation | random rotation (±10°), shift (±10 %), scale (0.9 to 1.1) and shear (±5°), applied to each training batch on the GPU; no flipping, because a mirrored character is a different shape |
| Model selection | checkpoint with the best validation accuracy (epoch 15 of 15) |
| Parameters | 1,184,462 |
| Training time | 1.5 min for 15 epochs on a Colab T4 GPU (about 6 s per epoch) |
| Inference | 0.99 ms per image (T4, batch size 1) |

Before the full run, the model memorised 100 training images (accuracy 1.0 after 8 epochs, no augmentation or regularisation). This sanity check shows that the data loading, labels and training loop are correct.

## 4.3 Results

The numbers come from `results/char_cnn_full_argmax/metrics.json`.

| Metric | Value |
|---|---|
| Test accuracy | 99.72 % (13,761 of 13,800 correct) |
| Test error rate (1 − accuracy) | 0.28 % (39 images) |
| Macro F1 | 99.72 % |
| Validation accuracy (best epoch) | 99.71 % |

The shared metric code computes CER over Unicode code points. The conjuncts क्ष, त्र and ज्ञ are three code points each, so the reported CER (0.30 %) differs slightly from 1 − accuracy. For this model, accuracy is the main number.

Validation accuracy is already 97.5 % after the first epoch and above 99 % from epoch 7. It still rises slowly until the end of the cosine schedule. Training and validation accuracy stay close, so the model does not overfit noticeably. The training loss (on augmented images) stays near 0.72 because label smoothing does not let it fall to zero.

**Most confused pairs on the test set** (out of 300 test images per class):

| True | Predicted | Count |
|---|---|---|
| ध | घ | 4 |
| द | ढ | 4 |
| त्र | ब | 3 |
| ढ | द | 3 |
| ब | व | 2 |
| च, ट, छ, य, ज्ञ | य, ठ, द, म, श | 1 each |

## 4.4 Observations

1. **The remaining errors are mostly look-alike letters.** The five most frequent pairs (ध/घ, द/ढ in both directions, त्र/ब, ब/व) cause 16 of the 39 errors. Within each pair, the two letters differ only in a small stroke or loop; for example, ब has a diagonal stroke inside the bowl and व does not.
2. **Most classes are almost solved.** 17 of the 46 classes have no error at all, and no digit appears among the confused pairs. The weakest class is द, with an F1 of 98.7 %.
3. **The word models have the same difficulty** with look-alike shapes such as ध/घ (Section 7.5).

## 4.5 What this result does not show

DHCD contains only isolated, centred consonants and digits. It has no vowels, vowel signs or word context, and no segmentation is needed. This 99.72 % accuracy therefore says little about reading real words. It is also not directly comparable with the word models' CER and WER (Section 7).
