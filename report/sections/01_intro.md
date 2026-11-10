# 1. Introduction

*Owner: Monu. Draft below uses only verified facts; edit freely, keep it to about half a page, and delete the italic notes.*

Reading handwritten text is an old problem, but it is far from solved for most Indian scripts. Hindi, written in the Devanagari script, is used by hundreds of millions of people, yet it has much less public labelled handwriting data and many fewer ready-made tools than English. It is a **low-resource** setting for optical character recognition (OCR).

Devanagari is hard for a machine for several reasons. A horizontal headline (shirorekha) joins the letters of a word, so characters cannot simply be cut apart. Vowel signs (matras) attach above, below, before or after a consonant: in कि the vowel sign is drawn before the consonant but stored after it in Unicode. Half-letters join into conjuncts such as क्ष. Several letters look alike, for example ब and व, or घ and ध. Handwriting adds large differences between writers.

In this project we recognise handwritten Devanagari at two levels and compare three model families:

1. a **convolutional neural network (CNN)** that classifies single handwritten characters (UCI DHCD dataset);
2. a **CRNN**, a CNN followed by a bidirectional LSTM and trained with CTC loss, that reads whole handwritten words (IIIT-HW-Dev dataset) [1];
3. a **TrOCR-style encoder-decoder**, a pre-trained Transformer model fine-tuned on the same word data [2].

Our contributions are:

- a common evaluation of the three models with one shared implementation of the character error rate (CER) and word error rate (WER);
- a comparison of character-level and word-level error rates on a low-resource script, with a clear statement of why the two levels are not directly comparable;
- an analysis of the errors: which characters, vowel signs and conjuncts are confused, and how error depends on word length;
- an open, reproducible code base with fixed splits, seeds and a results format shared by all models.

*TODO (after results): one sentence stating the main finding.*

The rest of the report is organised as follows. Section 2 reviews related work, Section 3 describes the data, Sections 4 to 6 describe the three models, Section 7 presents and analyses the results, and Section 8 concludes.
