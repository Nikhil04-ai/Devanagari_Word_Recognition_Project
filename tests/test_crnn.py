"""Unit tests for the CRNN-CTC word recogniser: src/crnn/dataset.py, model.py and decode.py.

They check the pieces on tiny made-up data, without real images or a GPU:
the vocabulary, image loading, the network output shape, the CTC loss, greedy decoding and the
dataset / batching code. Run with: pytest tests/test_crnn.py
(The whole file is skipped if PyTorch is not installed.)
"""
# csv = write a small CSV file for the dataset test.
import csv

# pytest = the test runner.
import pytest

# importorskip: if torch is not installed, skip every test in this file instead of failing.
torch = pytest.importorskip("torch")
# Pillow: creates small test images. The noqa notes tell the style checker that these imports
# come after the torch line on purpose.
from PIL import Image  # noqa: E402

# Code under test from the dataset file: vocabulary, dataset class, batching function, image loader,
# and the helper that gives the fewest CTC time steps a label needs.
from src.crnn.dataset import (  # noqa: E402
    CharVocab, WordDataset, collate_fn, load_image, min_ctc_input_length,
)
# Greedy CTC decoder.
from src.crnn.decode import greedy_decode  # noqa: E402
# The network and the parameter counter.
from src.crnn.model import CRNN, count_params  # noqa: E402

# A tiny vocabulary of five characters. Index 0 is the CTC blank, so they get indices 1 to 5.
CHARS = ["क", "म", "ल", "ि", "त"]


def test_vocab_roundtrip_and_blank():
    """Proves: the vocabulary maps characters to indices (starting at 1) and back, and the blank is dropped.
    5 characters + the blank = 6 classes.
    """
    v = CharVocab(CHARS)
    # +1 because index 0 is reserved for the CTC blank.
    assert len(v) == len(CHARS) + 1  # +1 for the blank
    # क is 1, म is 2, ल is 3.
    assert v.encode("कमल") == [1, 2, 3]
    # Index 0 (blank) is skipped when turning indices back into text.
    assert v.indices_to_text([1, 0, 2, 3]) == "कमल"  # blank (0) is dropped


def test_vocab_rejects_unknown_character():
    """Proves: a character that is not in the vocabulary raises ValueError (strict mode is the default)."""
    with pytest.raises(ValueError):
        CharVocab(CHARS).encode("कxम")


def test_vocab_from_file_keeps_special_characters(tmp_path):
    """Proves: CharVocab.from_file reads one character per line and keeps the invisible zero-width
    joiner as its own entry (only the line break is removed from each line).
    """
    # p is a vocab.txt inside pytest's temporary folder.
    p = tmp_path / "vocab.txt"
    p.write_text("क\n‍\nम\n", encoding="utf-8")  # zero-width joiner on its own line
    # Read the file back into a CharVocab.
    v = CharVocab.from_file(p)
    # All three entries are kept, in file order.
    assert v.chars == ["क", "‍", "म"]


def test_min_ctc_input_length_counts_repeats():
    """Proves: min_ctc_input_length adds one extra time step for each pair of equal neighbours.
    CTC must place a blank between two equal letters, so such labels need more time steps.
    """
    assert min_ctc_input_length([1, 2, 3]) == 3
    assert min_ctc_input_length([1, 1, 2]) == 4  # a blank is needed between the two 1s


def test_load_image_height_width_and_range():
    """Proves: load_image returns a tensor of shape (1, 64, width) with values between -1 and 1.
    Height is always 64. Width is limited to between MIN_W (128) and MAX_W (512).
    """
    # A 40 x 20 white image: it is scaled to height 64, which gives width 128.
    narrow = load_image(Image.new("L", (40, 20), 255))
    # A 4000 x 50 white image: scaling would give a width far above 512, so it is squeezed to 512.
    wide = load_image(Image.new("L", (4000, 50), 255))
    # Note: here 40 x 20 becomes exactly 128 wide by resizing; the padding to MIN_W is not needed for this input.
    assert narrow.shape == (1, 64, 128)  # padded up to MIN_W
    assert wide.shape == (1, 64, 512)    # squeezed to MAX_W
    # White is +1.0 and nothing goes outside the range -1 to 1.
    assert float(narrow.max()) <= 1.0 and float(narrow.min()) >= -1.0


def test_model_output_shape_and_lengths():
    """Proves: the network output has shape (time steps, batch, classes) and holds log-probabilities.
    Input width 128 gives 128 // 4 = 32 time steps. Also checks output_lengths and the model size.
    """
    # 6 classes = 5 characters + the blank.
    m = CRNN(num_classes=len(CHARS) + 1)
    # A random batch: 2 images, 1 channel, 64 high, 128 wide.
    x = torch.randn(2, 1, 64, 128)
    out = m(x)
    # T = 128 // 4 = 32 time steps, batch 2, 6 classes.
    assert out.shape == (32, 2, len(CHARS) + 1)  # T = 128 // 4
    # exp of a log-softmax must sum to 1 over the class axis (dim 2), so the output is log-probabilities.
    assert torch.allclose(out.exp().sum(2), torch.ones(32, 2), atol=1e-4)  # log-softmax
    # Real widths 128 and 256 give 32 and 64 time steps.
    assert CRNN.output_lengths(torch.tensor([128, 256])).tolist() == [32, 64]
    # The model is a real size network (more than one million trainable parameters).
    assert count_params(m) > 1_000_000


def test_ctc_loss_is_finite_on_a_batch():
    """Proves: nn.CTCLoss accepts the model output with matching targets and gives a finite number."""
    m = CRNN(num_classes=len(CHARS) + 1)
    # Model output of a random batch (untrained, so the numbers are random).
    out = m(torch.randn(2, 1, 64, 128))
    # Both labels joined into one flat list: the first has 3 indices, the second 2.
    targets = torch.tensor([1, 2, 3, 4, 5])
    # blank=0 matches the vocabulary; zero_infinity=True turns impossible alignments into 0 instead of infinity.
    loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)(
        # Input lengths (32 time steps each) and target lengths (3 and 2).
        out, targets, torch.tensor([32, 32]), torch.tensor([3, 2]))
    # The loss must be finite (not nan or inf).
    assert torch.isfinite(loss)


def test_greedy_decode_merges_repeats_and_drops_blanks():
    """Proves: greedy decoding merges repeated symbols and removes blanks.
    Time steps  क क - म - - ल ल  must decode to  कमल.
    """
    v = CharVocab(CHARS)
    # time steps: क क - म - - ल ल   -> कमल
    # Best symbol at each time step (index 0 is the blank, 1 = क, 2 = म, 3 = ल).
    seq = [1, 1, 0, 2, 0, 0, 3, 3]
    # Fake log-probabilities: -10 everywhere, and 0 (the highest) for the chosen symbol at each step.
    lp = torch.full((len(seq), 1, len(v)), -10.0)
    for t, k in enumerate(seq):
        # Put the highest score at the chosen index for time step t.
        lp[t, 0, k] = 0.0
    # Input: log-probabilities, the number of valid steps, the vocabulary. Output: a list with one text.
    assert greedy_decode(lp, [len(seq)], v) == ["कमल"]


def test_greedy_decode_blank_separates_double_letters():
    """Proves: a blank between two equal symbols keeps both, so a double letter is possible.
    क - क  must decode to  कक  (without the blank it would be one क).
    """
    v = CharVocab(CHARS)
    seq = [1, 0, 1]  # क - क  -> कक
    # Fake log-probabilities with the best symbol set at each step, as in the first decode test.
    lp = torch.full((3, 1, len(v)), -10.0)
    for t, k in enumerate(seq):
        lp[t, 0, k] = 0.0
    assert greedy_decode(lp, [3], v) == ["कक"]


def test_greedy_decode_respects_valid_length():
    """Proves: decoding uses only the valid time steps. The last two steps are padding and must be ignored."""
    v = CharVocab(CHARS)
    # Best symbol per time step: क म ल ल. The last two steps are padding.
    seq = [1, 2, 3, 3]
    # Fake log-probabilities with the best symbol set at each step, as in the first decode test.
    lp = torch.full((4, 1, len(v)), -10.0)
    for t, k in enumerate(seq):
        lp[t, 0, k] = 0.0
    # Only the first 2 steps are valid, so the result is कम.
    assert greedy_decode(lp, [2], v) == ["कम"]  # the last two steps are padding


def test_dataset_and_collate(tmp_path):
    """Proves: WordDataset reads a CSV and images correctly, and collate_fn builds a padded batch.
    Two images of different width and two words of different length are batched together.
    """
    v = CharVocab(CHARS)
    # Rows for the CSV: (image file name, word).
    rows = []
    # Two test words: a 40 px wide image with कमल and a 300 px wide image with मत. Both are 50 px high.
    for i, (w, text) in enumerate([(40, "कमल"), (300, "मत")]):
        # Draw a white image and save it as a PNG in the temporary folder.
        Image.new("L", (w, 50), 255).save(tmp_path / f"{i}.png")
        rows.append((f"{i}.png", text))
    # Write the CSV with the header image_path,text that the dataset expects.
    csv_path = tmp_path / "train.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image_path", "text"])
        wr.writerows(rows)

    # root= says which folder the image paths in the CSV are relative to.
    ds = WordDataset(csv_path, v, root=str(tmp_path))
    # Two samples in the dataset.
    assert len(ds) == 2
    # One sample is (image tensor, label indices, text, path).
    x, y, text, path = ds[0]
    # Image has 1 channel and height 64; the label decodes back to the right word.
    assert x.shape[:2] == (1, 64) and v.indices_to_text(y.tolist()) == "कमल"

    # Batch the two samples (widths differ, so the shorter image is padded).
    batch = collate_fn([ds[0], ds[1]])
    # Batch of 2 images, each 64 high.
    assert batch["images"].shape[0] == 2 and batch["images"].shape[2] == 64
    # Padded width equals the widest real width in the batch.
    assert batch["images"].shape[3] == int(batch["widths"].max())
    # Label lengths: 3 and 2.
    assert batch["target_lengths"].tolist() == [3, 2]
    # All labels are joined into one flat list of 3 + 2 = 5 indices (the format CTC loss expects).
    assert batch["targets"].numel() == 5
