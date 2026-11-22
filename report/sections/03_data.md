# 3. Datasets and preprocessing

## 3.1 Character dataset: DHCD

Model A uses the Devanagari Handwritten Character Dataset (DHCD) from the UCI Machine Learning Repository [4]. It has 92,000 grayscale images of 32×32 pixels in 46 classes (36 consonants and 10 digits) with 2,000 images per class. The character is centred in a 28×28 region with 2 pixels of padding on every side. The data is split 85/15 into training and test images (78,200 and 13,800, that is 1,700 and 300 per class). DHCD contains **no vowels and no vowel signs (matras)**, and every image shows one isolated character. The dataset is licensed CC BY 4.0.

## 3.2 Word dataset: IIIT-HW-Dev

Models B and C use IIIT-HW-Dev [3], released by CVIT, IIIT Hyderabad: more than 95,000 handwritten Devanagari word images (jpg). Each image is one word cut out of a handwritten page; the folders are organised by writer and page. We use Version 1, which corrects segmentation errors of Version 0. The labels are the words themselves, in Unicode, in the files `train.txt`, `val.txt` and `test.txt` (one line per image: relative path, a space, the word). No image or label was removed (95,430 images in total). The splits are **by writer**: the 12 writers are divided into 7 (training), 2 (validation) and 3 (test), and no writer appears in two splits. The dataset also ships a vocabulary file with 11,030 entries and a lexicon file; we use neither, because our character list comes from the training labels.

| | Train | Validation | Test |
|---|---|---|---|
| Images | 69,853 | 12,708 | 12,869 |
| Writers | 7 | 2 | 3 |
| Pages | 1,823 | 332 | 336 |
| Distinct words (after NFC) | 9,495 | 8,566 | 6,907 |
| Longest word (code points) | 23 | 23 | 22 |
| Labels changed by NFC | 1,841 | 316 | 331 |
| Images whose word occurs in training | n/a | 100 % | 100 % |

Table: Statistics of the IIIT-HW-Dev splits (source: `data/processed/words/stats.json`, written by `scripts/prepare_iiit.py`).

## 3.3 Preprocessing

- **Unicode normalisation.** Every label is normalised to NFC. A Devanagari word can be stored as different code-point sequences, which would make a correct answer look wrong. About 2.5 % of the labels in every split are not in NFC form (1,841 of 69,853 in training), so normalisation changes real labels, and it merges some spellings of the same word.
- **Labels are code points.** The models predict Unicode code points, so a consonant, a vowel sign and the virama (halant) are separate symbols. The labels include a hyphen (in 2,329 training labels) and the Devanagari danda; the longest label has 23 code points, which fits the 32 time steps the CRNN has at its minimum width. The character list is built from the training split only and has 100 code points; no validation or test label contains a character outside it.
- **Images (word models).** Grayscale, resized to a height of 64 pixels keeping the aspect ratio, width limited to 512 pixels, narrow images padded on the right with white to at least 128 pixels, values scaled to [-1, 1]. Inside a batch, images are padded on the right with white to the widest one. No augmentation was used. About 17 % of the images (11,887 of 69,853 in training) are narrower than 128 pixels after resizing and are padded; 9 training images are wider than 512 pixels and are squeezed. The median width at height 64 is 162 pixels. All sampled images show dark ink on light paper.
- **Images (DHCD).** Used as provided, 32×32 grayscale.

## 3.4 A low-resource problem

The word dataset has about 95K images but a vocabulary of only about 11K entries, so the models see each word several times and meet few new words, and it comes from only 12 writers (7 for training, 2 for validation, 3 for testing), so the variety of handwriting is limited (see also Section 1).

## 3.5 Test words and test writers

Every word in the validation and test splits also occurs in the training split (100 %), and no writer is shared between splits. The test therefore measures **generalisation to new writers, not to new words**: a model can partly memorise the 9,495 training words. We cannot report error rates for unseen words; instead we report error by how often a word occurs in training (Section 7.4). An error rate on DHCD (isolated characters) is not comparable with one on IIIT-HW-Dev (whole words); see Section 7.2.
