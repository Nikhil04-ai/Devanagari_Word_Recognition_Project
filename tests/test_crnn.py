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

# =====================================================================================================
# CONCEPTS USED IN THIS FILE
# =====================================================================================================
# unit test: a small automatic check of one small piece of code (one function or class) on its own.
#     Every test_* function below is a unit test.
# regression test: a test that stays in the project so that an old bug, once fixed, cannot come back
#     unnoticed. For example the blank-between-double-letters test guards the CTC merge rule.
# pytest: the test runner. It finds files named test_*.py and runs every function named test_*.
# assert: a line that must be true. If it is false, pytest marks the test as FAILED and shows the values.
# pytest.raises(ValueError): says "this code MUST raise this error". If no error is raised, the test fails.
#     Used in test_vocab_rejects_unknown_character.
# pytest.importorskip("torch"): if PyTorch is not installed, the whole file is skipped, not failed.
# fixture: a helper that pytest builds and passes into a test by argument name.
# tmp_path: a built-in pytest fixture. It gives a fresh empty temporary folder for each test, deleted
#     later. Used in the vocab-file test and the dataset test so no real file in the project is touched.
# fake / synthetic data: tiny made-up input (white PIL images, random tensors, hand-written score tables)
#     instead of the real IIIT-HW-Dev dataset. The tests check the logic of the code, not accuracy.
# determinism: the same input always gives the same output. The decode, vocab and padding tests are
#     fully deterministic. The model tests use torch.randn (random numbers) but only check shapes
#     and properties that are true for ANY random numbers, so they never fail by chance.
# vocabulary (CharVocab): the list of characters the model knows, with an index for each one.
# blank symbol: CTC's extra class at index 0. It means "no character here" and it separates double
#     letters. It never appears in a real word, so real characters start at index 1.
# roundtrip: encode text to indices and decode back; you must get the same text. See the vocab test.
# CTC (Connectionist Temporal Classification): a way to train a model to read a word image without
#     telling it where each letter sits. The model gives one guess per image column, and CTC sums over
#     all ways those guesses can collapse to the true word.
# CTC loss (nn.CTCLoss): the training error for CTC. Smaller means the guesses fit the true word better.
#     zero_infinity=True turns an impossible case (label too long for the time steps) into 0.
# time step: one image column fed to the LSTM. The model turns a width-W image into W // 4 time steps.
# log-probabilities (log-softmax): log of the class probabilities. exp of them sums to 1 per time step.
#     nn.CTCLoss expects this format. See test_model_output_shape_and_lengths.
# tensor shape: written (B, C, H, W) = (batch, channels, height, width) for images, and
#     (T, B, classes) = (time steps, batch, classes) for the model output.
# batch: several samples processed together. A batch of 2 means 2 images in one tensor.
# padding: extra white columns added on the right so images of different width can share one tensor.
#     CRNN needs the real widths (batch["widths"]) so that the padding is ignored.
# collate_fn: the function that turns a list of samples into one batch dictionary.
# greedy decoding: at each time step take only the single best class, merge neighbouring repeats, then
#     drop blanks. Beam search (not used) would keep several candidates instead.
# CRNN: Convolutional Recurrent Neural Network = CNN (looks at the image) + BiLSTM (reads columns left
#     and right) + linear layer (one score per class per column).
# zero-width joiner (ZWJ): an invisible Unicode character used in Devanagari. The vocab file must keep it.
# trainable parameters: the weights the optimiser changes. count_params counts them (the model size).
# atol: "absolute tolerance" in torch.allclose. Floating point numbers are only equal up to a tiny error.
# =====================================================================================================
# Q&A (whole file)
# =====================================================================================================
# Q: What part of the project does this file protect?
# A: The CRNN-CTC word recogniser: src/crnn/dataset.py (vocabulary, image loading, dataset, batching),
#    src/crnn/model.py (the network) and src/crnn/decode.py (greedy CTC decoding).
# Q: How is it run?
# A: From the repository root with: pytest tests/test_crnn.py (pytest -q tests/test_crnn.py for short output).
# Q: Why is no real dataset needed?
# A: Each test builds its own tiny input: a 5-letter vocabulary, white images made with Pillow, random
#    tensors, or hand-made score tables. The tests check shapes, rules and logic, which do not depend on
#    real handwriting. This also makes the tests fast and runnable on a CPU.
# Q: What happens if PyTorch is missing?
# A: pytest.importorskip skips the whole file, so the result is "skipped", not "failed".

# A tiny vocabulary of five characters. Index 0 is the CTC blank, so they get indices 1 to 5.
CHARS = ["क", "म", "ल", "ि", "त"]


def test_vocab_roundtrip_and_blank():
    """Proves: the vocabulary maps characters to indices (starting at 1) and back, and the blank is dropped.
    5 characters + the blank = 6 classes.
    """
    # Q: What does this test check?
    # A: That CharVocab turns text into indices (starting at 1), turns indices back into the same text, drops
    #    index 0 (the blank) when decoding, and reports 5 characters + 1 blank = 6 classes.
    # Q: Why does it matter? What bug would make it fail?
    # A: If the indices started at 0, a real letter would collide with the CTC blank and the loss would learn
    #    wrong targets. If len() forgot the blank, the network's last layer would have one class too few.
    # Q: How is the input built and why is it enough?
    # A: CHARS is five Devanagari letters typed in the file. The rules (shift by 1, skip 0) do not depend on
    #    which letters are used, so five letters prove them.
    # Q: Which project code does it exercise?
    # A: CharVocab.__init__, __len__, encode and indices_to_text in src/crnn/dataset.py.
    v = CharVocab(CHARS)
    # +1 because index 0 is reserved for the CTC blank.
    assert len(v) == len(CHARS) + 1  # +1 for the blank
    # क is 1, म is 2, ल is 3.
    assert v.encode("कमल") == [1, 2, 3]
    # Index 0 (blank) is skipped when turning indices back into text.
    assert v.indices_to_text([1, 0, 2, 3]) == "कमल"  # blank (0) is dropped


def test_vocab_rejects_unknown_character():
    """Proves: a character that is not in the vocabulary raises ValueError (strict mode is the default)."""
    # Q: What does this test check?
    # A: That encoding text with a letter that is not in the vocabulary (the plain Latin "x") raises ValueError.
    # Q: Why does it matter? What bug would make it fail?
    # A: Silently skipping or mis-numbering an unknown letter would train the model on a wrong label without
    #    any warning. Strict mode (the default) stops the run so the data problem is found early.
    # Q: How is the input built and why is it enough?
    # A: One word "कxम" with a single bad letter in the middle. One unknown character is enough to trigger the check.
    # Q: Which project code does it exercise?
    # A: CharVocab.encode (strict=True) in src/crnn/dataset.py.
    with pytest.raises(ValueError):
        CharVocab(CHARS).encode("कxम")


def test_vocab_from_file_keeps_special_characters(tmp_path):
    """Proves: CharVocab.from_file reads one character per line and keeps the invisible zero-width
    joiner as its own entry (only the line break is removed from each line).
    """
    # Q: What does this test check?
    # A: That CharVocab.from_file reads one character per line and keeps the invisible zero-width joiner as
    #    its own vocabulary entry, in file order.
    # Q: Why does it matter? What bug would make it fail?
    # A: If the loader used .strip() on each line, the joiner would be treated as whitespace and removed. The
    #    vocabulary would be shorter than the one used in training, so class indices would not match.
    # Q: How is the input built and why is it enough?
    # A: A tiny vocab.txt is written into tmp_path (a temporary folder) with three lines: क, the joiner, म.
    #    The real vocab.txt is never touched.
    # Q: Which project code does it exercise?
    # A: CharVocab.from_file in src/crnn/dataset.py (it removes only "\r\n" from each line).
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
    # Q: What does this test check?
    # A: That the fewest time steps a label needs is its length plus one for every pair of equal neighbours:
    #    [1, 2, 3] needs 3 and [1, 1, 2] needs 4.
    # Q: Why does it matter? What bug would make it fail?
    # A: CTC must put a blank between two equal letters. If this count ignored that, a short image could
    #    be accepted for a label it cannot fit, and the CTC loss would be infinite.
    # Q: How is the input built and why is it enough?
    # A: Two small lists of indices: one without repeats and one with a repeat. They cover both branches.
    # Q: Which project code does it exercise?
    # A: min_ctc_input_length in src/crnn/dataset.py.
    assert min_ctc_input_length([1, 2, 3]) == 3
    assert min_ctc_input_length([1, 1, 2]) == 4  # a blank is needed between the two 1s


def test_load_image_height_width_and_range():
    """Proves: load_image returns a tensor of shape (1, 64, width) with values between -1 and 1.
    Height is always 64. Width is limited to between MIN_W (128) and MAX_W (512).
    """
    # Q: What does this test check?
    # A: That load_image gives a tensor (1, 64, W): one channel, height always 64, width between 128 and 512,
    #    and pixel values inside -1 to +1.
    #    It uses two blank white Pillow images, 40 x 20 (small) and 4000 x 50 (very wide), to check the
    #    minimum width and the cap at 512. A wrong height would break the CNN and the Linear layer size.
    # Q: Does the narrow image really test the padding?
    # A: No. 40 x 20 is scaled to width 40 * 64 / 20 = 128, which is already MIN_W, so no padding happens.
    #    A narrower input (for example 10 x 20) would be needed to test the padding itself.
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
    # Q: What does this test check?
    # A: That the CRNN returns shape (time steps, batch, classes) = (32, 2, 6), that the values are
    #    log-probabilities, that output_lengths gives width // 4, and that the model has over 1 million weights.
    # Q: Why does it matter? What bug would make it fail?
    # A: nn.CTCLoss needs time first and log-probabilities. If someone changed the pooling or forgot
    #    log_softmax, the shape or the "sums to 1" check would fail. A wrong output_lengths would make CTC
    #    read padded columns.
    # Q: How is the input built and why is it enough?
    # A: torch.randn(2, 1, 64, 128) is a random batch of 2 images. The shape rules do not depend on the
    #    pixel values, so random numbers are enough. No seed is needed because the checks hold for any values.
    # Q: Which project code does it exercise?
    # A: CRNN.forward, CRNN.output_lengths and count_params in src/crnn/model.py.
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
    # Q: What does this test check?
    # A: That the model output, together with targets and lengths in the CTC format, gives a loss that is a
    #    real finite number (not nan or inf).
    # Q: Why does it matter? What bug would make it fail?
    # A: It proves the model output and nn.CTCLoss fit together: log-probabilities, time-first layout and
    #    blank=0. A wrong layout would crash or give nan, and training would not work at all.
    # Q: How is the input built and why is it enough?
    # A: A random batch of 2 images and two tiny labels joined into one flat list [1,2,3,4,5] with lengths
    #    3 and 2. This is the same format collate_fn produces. The 32 time steps are enough for both labels.
    # Q: Which project code does it exercise?
    # A: CRNN.forward in src/crnn/model.py together with torch.nn.CTCLoss, set up as in src/crnn/train.py.
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
    # Q: What does this test check?
    # A: That greedy_decode turns the time steps क क - म - - ल ल into the word कमल: repeated neighbours are
    #    merged and blanks are removed.
    # Q: Why does it matter? What bug would make it fail?
    # A: Without merging, the output would be "ककमलल". Without removing blanks, blank indices would leak into
    #    the text. Either way the predicted words and the CER would be wrong.
    # Q: How is the fake input built and why is it enough?
    # A: A table of log-probabilities filled with -10, with 0 (the maximum) at the chosen symbol of each step.
    #    The argmax is then exactly the symbol we want, so the result is fully predictable (no randomness).
    # Q: Which project code does it exercise?
    # A: greedy_decode in src/crnn/decode.py (used in validation and in the final test).
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
    # Q: What does this test check?
    # A: That a blank between two equal symbols keeps both: क - क decodes to कक, not to one क.
    # Q: Why does it matter? What bug would make it fail?
    # A: Real words have double letters. If the decoder forgot the previous symbol over a blank (for
    #    example by skipping blanks before merging), such words would always lose a letter.
    # Q: How is the fake input built and why is it enough?
    # A: Three time steps with best symbols [1, 0, 1], made by the same -10 / 0 score table as before.
    #    That is the smallest sequence that shows the blank-separates-repeats rule.
    # Q: Which project code does it exercise?
    # A: greedy_decode in src/crnn/decode.py (the prev variable is updated even for a blank).
    v = CharVocab(CHARS)
    seq = [1, 0, 1]  # क - क  -> कक
    # Fake log-probabilities with the best symbol set at each step, as in the first decode test.
    lp = torch.full((3, 1, len(v)), -10.0)
    for t, k in enumerate(seq):
        lp[t, 0, k] = 0.0
    assert greedy_decode(lp, [3], v) == ["कक"]


def test_greedy_decode_respects_valid_length():
    """Proves: decoding uses only the valid time steps. The last two steps are padding and must be ignored."""
    # Q: What does this test check?
    # A: That decoding only reads the first n time steps given in out_lens. With n = 2 the answer is कम,
    #    even though 4 steps exist.
    # Q: Why does it matter? What bug would make it fail?
    # A: In a batch, short images are padded with white columns, and the model also gives guesses for them.
    #    If the decoder read those steps, junk letters would be added to short words and the CER would rise.
    # Q: How is the fake input built and why is it enough?
    # A: Four time steps with best symbols [1, 2, 3, 3]. The last two stand for padding. If they were used
    #    the result would be कमल, so the test can tell the difference.
    # Q: Which project code does it exercise?
    # A: greedy_decode in src/crnn/decode.py (the seq[:n] slice).
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
    # Q: What does this test check?
    # A: That WordDataset reads a CSV and its images (right length, image shape, right label) and that
    #    collate_fn builds a batch: padded images, label lengths [3, 2] and one flat targets list of 5.
    # Q: Why does it matter? What bug would make it fail?
    # A: If the batch were built wrongly (wrong padding width, wrong label lengths, labels not joined), the
    #    CTC loss would get misaligned targets and train on garbage. Images of different width are the
    #    exact case that needs the custom collate function.
    # Q: How is the fake input built and why is it enough?
    # A: Two white PNG files (40 and 300 px wide) and a train.csv with the header image_path,text are written
    #    into tmp_path. Two different widths and two different label lengths test the padding and the
    #    flat-target logic. The content of the images does not matter here.
    # Q: Which project code does it exercise?
    # A: WordDataset.__init__, __len__, __getitem__ and collate_fn in src/crnn/dataset.py (and load_image).
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
