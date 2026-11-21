"""CTC decoding for the CRNN.

Where this file fits in the pipeline:
    model.py gives probabilities for every time step -> decode.py turns them into a text string.
The result is used for validation during training (train.py) and for the final test (evaluate.py).
"""

# ===== CONCEPTS USED IN THIS FILE =====
# CTC (Connectionist Temporal Classification): a loss and output format where the model predicts one symbol per
#    time step and a fixed rule turns the symbol path into text. greedy_decode applies that rule.
# Blank symbol: the special class 0 meaning "no character here". It is removed from the output text.
# Time steps: the columns of the CNN feature map. The model gives one probability vector per time step.
# Log-probabilities (log_probs): log of the class probabilities, shape (T, B, C) = time, batch, classes.
# Greedy decoding: take the single most likely class (argmax) at every time step and do no search.
#    Beam search would keep several best paths; it is slower. This file only does greedy.
# Merging repeats: the same symbol in neighbouring time steps counts as one character, because one letter
#    usually covers several columns.
# Padding and out_lens: short images are padded in a batch, so only the first out_lens steps are read.

# ===== VIVA Q&A FOR THIS FILE =====
# Q: What part of the project is this file?
# A: The last step of prediction: it converts the model's per-column probabilities into readable text.
# Q: How is it run?
# A: It is not run on its own. train.py (validation) and evaluate.py (test, through evaluate_loader) call it.
# Q: What does it read and write?
# A: It reads a log-probability tensor, the valid lengths and a CharVocab. It returns a list of strings
#    and writes nothing to disk.



def greedy_decode(log_probs, out_lens, vocab):
    """Best symbol per time step, merge repeats, drop blanks.

    log_probs: (T, B, C) tensor, out_lens: per-sample number of valid time steps.
    Returns a list of B strings.

    Example (blank shown as "_"): the best symbols  a a _ a b b _  become  "a", "a", "b"
    after merging repeats and removing blanks, so the text is "aab".
    Greedy means we trust the single most likely symbol at each step and do not search
    other paths (beam search would). It is fast and needs no extra parameters.
    """
    # Q: What does it do and why does it exist?
    # A: It converts the network output into text: best class per time step, merge equal neighbours, remove
    #    blanks. Without it the output would only be numbers, and CER and WER could not be computed.
    # Q: Why is prev updated even when a symbol is skipped, and why cut at n steps?
    # A: prev must hold the last symbol including blanks. Then "a _ a" keeps both letters (a blank separates
    #    them) and "a a" becomes one. Steps after n are padding, so reading them could add wrong letters.
    # argmax over the class axis -> (T, B), then swap axes -> (B, T), then to plain Python lists.
    best = log_probs.argmax(dim=2).transpose(0, 1).tolist()  # (B, T)
    texts = []
    # out_lens may be a tensor; convert each value to int so it can be used to slice.
    for seq, n in zip(best, [int(v) for v in out_lens]):
        ids, prev = [], -1  # prev = -1 means "no previous symbol yet"
        # Only look at the first n steps: later steps belong to the white padding of a short image.
        for k in seq[:n]:
            # Keep k only if it differs from the step before (merge repeats) and is not the blank.
            if k != prev and k != vocab.BLANK:
                ids.append(k)
            prev = k  # a blank between two equal symbols keeps both
        texts.append(vocab.indices_to_text(ids))
    return texts
