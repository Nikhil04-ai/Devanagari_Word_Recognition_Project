# 7. Results and analysis

*Owner: Monu, with tables and figures from Ravi's `results/summary/`. Every number comes from `results/*/metrics.json`. Delete italic notes.*

## 7.1 Main comparison

All models are evaluated on their own test set, once, with the shared metric code. CER is the edit distance over Unicode code points divided by the number of reference characters; WER is the share of words that are not exactly right.

| Model | Test data | Level | Decoding | CER ↓ | WER ↓ | Accuracy ↑ | Parameters | Train time |
|---|---|---|---|---|---|---|---|---|
| A: CNN | DHCD test (13,800 isolated characters) | Character | argmax | TODO | n/a | TODO | TODO | TODO |
| B: CRNN-CTC | IIIT-HW-Dev test | Word | greedy | TODO | TODO | TODO | TODO | TODO |
| C: TrOCR-style | IIIT-HW-Dev test | Word | greedy | TODO | TODO | TODO | TODO | TODO |

*Add one bar chart of CER and WER (`results/summary/`).*

## 7.2 Why the character and word error rates are not directly comparable

The CNN is tested on **isolated consonants and digits** with no vowels, no vowel signs and no context. The word models are tested on **whole words** that contain vowel signs and conjuncts, and they must also find where each character is. A character error of X % for the CNN and a CER of Y % for a word model are therefore different kinds of numbers. The gap between them mostly reflects how much harder word recognition is than classifying a clean isolated character, not a ranking of the architectures. The fair comparison is between Models B and C, which use the same data and the same metrics.

## 7.3 CRNN against TrOCR-style

*TODO: which is better in CER and WER, by how much, and what it cost (parameters, training time, speed). State honestly if one model had a reduced training budget.*

## 7.4 Word frequency

Every test word occurs in the training split and the test writers are new, so the test measures generalisation to new writers, not to new words (Section 3.5). To see whether rarer words are harder, we group the test images by how often their word occurs in the training split.

| Occurrences of the word in training | Share of test images | CER B: CRNN-CTC | CER C: TrOCR-style |
|---|---|---|---|
| 1 to 4 | TODO | TODO | TODO |
| 5 to 19 | TODO | TODO | TODO |
| 20 or more | TODO | TODO | TODO |

*TODO: one sentence on the gap. Adjust the bucket limits after looking at the distribution in `train.csv`.*

## 7.5 Error analysis

*Produced by Ravi's error-analysis script from `results/*/test_predictions.csv`. Keep it to the points the data supports.*

- **Most confused characters.** *TODO: top pairs (true → predicted) for each model, in a small table.*
- **Kinds of errors.** *TODO: error rate on vowel signs (ा ि ी ु ू े ै ो ौ), on conjuncts and the virama (्), on anusvara and chandrabindu (ं ँ), on plain consonants.*
- **Word length.** *TODO: CER by word length, one small figure.*
- **Examples.** *TODO: 4 to 6 failure cases (image, truth, prediction) with a one-line explanation each.*

## 7.6 Low-resource discussion

*TODO: two or three sentences linking the results to the low-resource setting: the limited vocabulary, the small amount of labelled data, and whether pre-training (Model C) helped. Do not claim results from experiments that were not run.*
