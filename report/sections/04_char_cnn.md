# 4. Model A: CNN character classifier

*Owner: Ravi. Describe what your code in `src/char_cnn/` actually does. The values in the table are the planned ones: correct them to match the final code, then delete the italic notes.*

## 4.1 Task and data

The CNN classifies one 32×32 grayscale image of a handwritten Devanagari consonant or digit into one of 46 classes (DHCD, Section 3.1). Training uses 78,200 images, of which 10 % are held out (stratified, seed 42) for validation; the 13,800 official test images are used once at the end.

## 4.2 Architecture and training

| Item | Value |
|---|---|
| Convolution blocks | 4 × [Conv 3×3, BatchNorm, ReLU, Conv 3×3, BatchNorm, ReLU, MaxPool 2×2], channels 32, 64, 128, 256 *(check)* |
| Head | global average pooling, dropout 0.4, linear layer to 46 classes *(check)* |
| Loss | cross-entropy with label smoothing 0.1 *(check)* |
| Optimiser, schedule | Adam, learning rate 1e-3, cosine schedule, about 30 epochs, batch 128 *(check)* |
| Augmentation | small rotation, shift, scale and shear; no flipping, because that changes the character *(check)* |
| Parameters | TODO |
| Training time | TODO |

## 4.3 Results

*From `results/char_cnn_full_argmax/metrics.json`. For a single character the character error rate equals 1 − accuracy.*

| Metric | Value |
|---|---|
| Test accuracy | TODO |
| Test error rate | TODO |
| Macro F1 | TODO |

*Add: the 10 most confused character pairs (for example look-alike letters such as ब and व) with a small figure or table, and the learning curve.*

## 4.4 Observations

*Two or three points: which characters are hardest and why; whether the accuracy is limited by look-alike shapes. TODO.*

## 4.5 What this result does not show

DHCD contains only isolated consonants and digits, with no vowels or vowel signs and no word context, so this accuracy says little about reading real words (Section 7).
