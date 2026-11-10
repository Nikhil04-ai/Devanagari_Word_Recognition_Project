# 2. Related work

*Owner: Monu. About one page. Every paper you cite must be opened and read; check each entry in `references.md`. The paragraphs below are a safe starting point; add the Devanagari/Indic papers you find.*

**Sequence recognition with CNN, RNN and CTC.** Reading a line or word as a sequence of image columns avoids segmenting characters. The CRNN of Shi et al. [1] combines a convolutional feature extractor, a recurrent layer and Connectionist Temporal Classification (CTC) [5], which trains on unsegmented labels by summing over all alignments between the output sequence and the transcription. It is end-to-end trainable and small, and it is the baseline for sequence recognition in this project.

**Transformer OCR.** TrOCR [2] replaces the CNN and the recurrent layer by Transformers. An image Transformer encoder, initialised from pre-trained vision models such as ViT [6] and DeiT [7], reads image patches, and a text Transformer decoder generates the transcription one token at a time. The authors report that, with large-scale pre-training, it outperforms earlier methods on printed, handwritten and scene text recognition. We adapt it to Devanagari by fine-tuning on our word data.

**Indic handwriting datasets and recognisers.** For Devanagari, Dutta et al. [3] released IIIT-HW-Dev, a word-level handwriting benchmark with more than 95K word images, and benchmarked existing recognisers on it. For isolated characters, the DHCD dataset [4] provides 92,000 images in 46 classes. *TODO: add 2 to 3 papers on Devanagari or Indic handwriting recognition (for example results on DHCD or IIIT-HW-Dev), one sentence each, only after reading them.*

**Low-resource OCR.** *TODO: one short paragraph on why scripts with little labelled data are harder, and on transfer from pre-trained models, citing only sources you have read.*

**Position of this work.** We do not propose a new method. We train three standard model families on the same script with the same metrics and fixed splits, and examine how character-level and word-level errors behave.
