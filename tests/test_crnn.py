import csv

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from src.crnn.dataset import (  # noqa: E402
    CharVocab, WordDataset, collate_fn, load_image, min_ctc_input_length,
)
from src.crnn.decode import greedy_decode  # noqa: E402
from src.crnn.model import CRNN, count_params  # noqa: E402

CHARS = ["क", "म", "ल", "ि", "त"]


def test_vocab_roundtrip_and_blank():
    v = CharVocab(CHARS)
    assert len(v) == len(CHARS) + 1  # +1 for the blank
    assert v.encode("कमल") == [1, 2, 3]
    assert v.indices_to_text([1, 0, 2, 3]) == "कमल"  # blank (0) is dropped


def test_vocab_rejects_unknown_character():
    with pytest.raises(ValueError):
        CharVocab(CHARS).encode("कxम")


def test_vocab_from_file_keeps_special_characters(tmp_path):
    p = tmp_path / "vocab.txt"
    p.write_text("क\n‍\nम\n", encoding="utf-8")  # zero-width joiner on its own line
    v = CharVocab.from_file(p)
    assert v.chars == ["क", "‍", "म"]


def test_min_ctc_input_length_counts_repeats():
    assert min_ctc_input_length([1, 2, 3]) == 3
    assert min_ctc_input_length([1, 1, 2]) == 4  # a blank is needed between the two 1s


def test_load_image_height_width_and_range():
    narrow = load_image(Image.new("L", (40, 20), 255))
    wide = load_image(Image.new("L", (4000, 50), 255))
    assert narrow.shape == (1, 64, 128)  # padded up to MIN_W
    assert wide.shape == (1, 64, 512)    # squeezed to MAX_W
    assert float(narrow.max()) <= 1.0 and float(narrow.min()) >= -1.0


def test_model_output_shape_and_lengths():
    m = CRNN(num_classes=len(CHARS) + 1)
    x = torch.randn(2, 1, 64, 128)
    out = m(x)
    assert out.shape == (32, 2, len(CHARS) + 1)  # T = 128 // 4
    assert torch.allclose(out.exp().sum(2), torch.ones(32, 2), atol=1e-4)  # log-softmax
    assert CRNN.output_lengths(torch.tensor([128, 256])).tolist() == [32, 64]
    assert count_params(m) > 1_000_000


def test_ctc_loss_is_finite_on_a_batch():
    m = CRNN(num_classes=len(CHARS) + 1)
    out = m(torch.randn(2, 1, 64, 128))
    targets = torch.tensor([1, 2, 3, 4, 5])
    loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)(
        out, targets, torch.tensor([32, 32]), torch.tensor([3, 2]))
    assert torch.isfinite(loss)


def test_greedy_decode_merges_repeats_and_drops_blanks():
    v = CharVocab(CHARS)
    # time steps: क क - म - - ल ल   -> कमल
    seq = [1, 1, 0, 2, 0, 0, 3, 3]
    lp = torch.full((len(seq), 1, len(v)), -10.0)
    for t, k in enumerate(seq):
        lp[t, 0, k] = 0.0
    assert greedy_decode(lp, [len(seq)], v) == ["कमल"]


def test_greedy_decode_blank_separates_double_letters():
    v = CharVocab(CHARS)
    seq = [1, 0, 1]  # क - क  -> कक
    lp = torch.full((3, 1, len(v)), -10.0)
    for t, k in enumerate(seq):
        lp[t, 0, k] = 0.0
    assert greedy_decode(lp, [3], v) == ["कक"]


def test_greedy_decode_respects_valid_length():
    v = CharVocab(CHARS)
    seq = [1, 2, 3, 3]
    lp = torch.full((4, 1, len(v)), -10.0)
    for t, k in enumerate(seq):
        lp[t, 0, k] = 0.0
    assert greedy_decode(lp, [2], v) == ["कम"]  # the last two steps are padding


def test_dataset_and_collate(tmp_path):
    v = CharVocab(CHARS)
    rows = []
    for i, (w, text) in enumerate([(40, "कमल"), (300, "मत")]):
        Image.new("L", (w, 50), 255).save(tmp_path / f"{i}.png")
        rows.append((f"{i}.png", text))
    csv_path = tmp_path / "train.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image_path", "text"])
        wr.writerows(rows)

    ds = WordDataset(csv_path, v, root=str(tmp_path))
    assert len(ds) == 2
    x, y, text, path = ds[0]
    assert x.shape[:2] == (1, 64) and v.indices_to_text(y.tolist()) == "कमल"

    batch = collate_fn([ds[0], ds[1]])
    assert batch["images"].shape[0] == 2 and batch["images"].shape[2] == 64
    assert batch["images"].shape[3] == int(batch["widths"].max())
    assert batch["target_lengths"].tolist() == [3, 2]
    assert batch["targets"].numel() == 5
