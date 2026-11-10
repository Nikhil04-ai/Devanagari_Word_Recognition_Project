"""CTC decoding for the CRNN."""


def greedy_decode(log_probs, out_lens, vocab):
    """Best symbol per time step, merge repeats, drop blanks.

    log_probs: (T, B, C) tensor, out_lens: per-sample number of valid time steps.
    Returns a list of B strings.
    """
    best = log_probs.argmax(dim=2).transpose(0, 1).tolist()  # (B, T)
    texts = []
    for seq, n in zip(best, [int(v) for v in out_lens]):
        ids, prev = [], -1
        for k in seq[:n]:
            if k != prev and k != vocab.BLANK:
                ids.append(k)
            prev = k  # a blank between two equal symbols keeps both
        texts.append(vocab.indices_to_text(ids))
    return texts
