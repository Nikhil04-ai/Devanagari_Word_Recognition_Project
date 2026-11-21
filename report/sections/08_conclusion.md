# 8. Conclusion, limitations and future work

## 8.1 Conclusion

We compared a CNN character classifier, a CRNN with CTC and a TrOCR-style Transformer on handwritten Devanagari, with one shared metric implementation and writer-disjoint test sets. On words by three unseen writers the two word models are equal at character level (CER 6.59 % and 6.47 %), but the TrOCR-style model reads more whole words correctly (WER 22.32 % against 26.05 %), at 3.7 times the parameters and about 20 times the prediction time. The CNN's 99.72 % accuracy on isolated characters is far above the word models, but it is not a like-for-like comparison: the CNN sees clean, centred consonants and digits, without vowel signs, conjuncts or the need to find where each character is. The most frequent word errors are confusions between look-alike consonants (त/न, ध/घ, म/ग), long and short vowel signs (ू/ु) and missing small marks (anusvara, nukta, danda), and conjuncts (the halant) are the hardest part of the script for both models.

## 8.2 Limitations

- **Limited budget.** Each word model trained for about 1.5 hours on a free GPU; the TrOCR-style model ran only 5 epochs and was still improving, so its result is probably not its best. It was not trained from scratch, so we cannot say whether pre-training helped.
- **Plain setup, one run.** No augmentation for the word models, no synthetic data, greedy decoding only, no hyper-parameter search, and one seed per model, so the 0.12-point CER difference between the word models may not be real.
- **Closed vocabulary, few writers.** Every test word occurs at least 5 times in training and the dataset has only 12 writers, so the test measures generalisation to new writers, not to new words.
- **Different data for the character model.** Model A cannot be ranked against Models B and C (Section 7.2).

## 8.3 Future work

- Train the TrOCR-style model to convergence and from scratch, to measure the value of pre-training directly.
- Train on 10 %, 25 % and 50 % of the training data to measure how each model behaves with less labelled data.
- Add data augmentation and synthetic Devanagari word images, which helped the dataset's original benchmark [3], and beam-search decoding.
- Evaluate on words that are absent from training, for example by holding out part of the vocabulary, to measure open-vocabulary reading.
- Recognise whole lines instead of single words, and repeat every run with several seeds.
