"""CTC decoding for the CRNN.

Where this file fits in the pipeline:
    model.py gives probabilities for every time step -> decode.py turns them into a text string.
The result is used for validation during training (train.py) and for the final test (evaluate.py).
"""


def greedy_decode(log_probs, out_lens, vocab):
    """Best symbol per time step, merge repeats, drop blanks.

    log_probs: (T, B, C) tensor, out_lens: per-sample number of valid time steps.
    Returns a list of B strings.

    Example (blank shown as "_"): the best symbols  a a _ a b b _  become  "a", "a", "b"
    after merging repeats and removing blanks, so the text is "aab".
    Greedy means we trust the single most likely symbol at each step and do not search
    other paths (beam search would). It is fast and needs no extra parameters.
    """
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
