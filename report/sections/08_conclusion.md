# 8. Conclusion, limitations and future work

*Owner: Monu. About half a page. The limitations below are true of our setup already; check them against what actually happened and add any others.*

## 8.1 Conclusion

*TODO after results: three to four sentences. What was compared, the main result for the word models, the character-versus-word gap and the reason for it, and the most frequent kinds of error.*

## 8.2 Limitations

- **Limited training budget.** The word models were trained within a short time budget on a free GPU, so they are probably not fully converged. *(If a model used only part of the training data or fewer epochs, state it here.)*
- **Plain setup.** No data augmentation, greedy decoding only, no hyper-parameter search.
- **One run, one seed.** The numbers have no confidence interval, so small differences between models should not be over-interpreted.
- **Closed vocabulary, few writers.** Every validation and test word also occurs in the training split, so the models can partly memorise words, and the whole dataset has only 12 writers (3 in the test split). The test measures generalisation to new writers, not to new words.
- **Different data for the character model.** DHCD and IIIT-HW-Dev differ in content, so Model A cannot be ranked against Models B and C (Section 7.2).

## 8.3 Future work

- Train on 10 %, 25 % and 50 % of the data to measure how each model behaves with less labelled data.
- Add data augmentation and beam-search decoding.
- Use a word list or language model, with care for the closed vocabulary of the dataset.
- Recognise whole lines of text instead of single words.
- Repeat with several seeds and report mean and spread.
