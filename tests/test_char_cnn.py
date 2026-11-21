"""Tests for Model A (the character CNN): data loading, split, model shapes, training, resume, predict.

Run from the repository root:  python -m pytest tests/test_char_cnn.py
The tests do not need the real dataset. They build a tiny fake DHCD in a temporary folder,
so they run in seconds on a CPU. They check that the pieces work together, not the real accuracy.
"""
# csv: read the training log. os: check that files exist.
import csv
import os

# numpy: make fake images. pytest: the test framework.
import numpy as np
import pytest

# If torch is not installed, skip all tests in this file instead of failing.
torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

# The imports come after importorskip on purpose (the `noqa: E402` marks that). They need torch.
from src.char_cnn.dataset import CLASSES, CHARS, NUM_CLASSES, load_split, random_affine, stratified_split, to_input  # noqa: E402
from src.char_cnn.model import CharCNN  # noqa: E402
from src.char_cnn.predict import load_model, predict, to_dhcd  # noqa: E402
from src.char_cnn.train import load_config, run  # noqa: E402

# ======================================================================================================
# CONCEPTS USED IN THIS FILE
# unit / integration test: a unit test checks one small function; test_train_resume_and_predict is closer to an
#   integration test because it runs data loading, model, training, checkpoints and prediction together.
# assert: a line that must be true; if it is false, pytest marks the test as failed and shows the line.
# pytest / test function: any function named test_* in tests/ is found and run by `python -m pytest`.
# importorskip: pytest.importorskip("torch") skips the whole file if torch is missing, instead of failing.
# fixture: a helper that pytest builds and passes in by argument name. Here only the built-in tmp_path is used.
# tmp_path: a fresh empty temporary folder (a pathlib.Path) for each test; pytest deletes it later. The fake
#   dataset and the checkpoints are written there, so the real project folders are never touched.
# fake / synthetic data: _make_dhcd draws tiny 32x32 black images with a white bar instead of real handwriting.
#   It is enough because these tests check that the code runs and the shapes are right, not the accuracy.
# DHCD: Devanagari Handwritten Character Dataset, 46 classes (36 consonants + 10 digits), 32x32 grayscale PNGs.
# class index <-> character: CLASSES / CHARS fix the order, so class 0 is "क" and class 45 is "९".
# uint8 tensor: image pixels stored as whole numbers 0..255; the model needs floats, so to_input converts them.
# cache (.npz): the first load_split writes cache_<split>.npz; later loads read it, which is faster.
# stratified split: split train/validation so that every class keeps the same share in validation
#   (test_stratified_split_keeps_every_class).
# seed: a number that fixes the random generator, so the split is the same on every run (seed=42 above).
# data leakage: the same image sitting in both train and validation, which would make validation look too good.
# augmentation: random small rotation / shift / scale / shear of the images (random_affine) to add variety.
# batch and tensor shape: B images go through the network together; (B,1,32,32) = batch, channel, height, width.
# logits: the raw 46 scores the model outputs; argmax picks the index of the largest one as the predicted class.
# epoch: one full pass over the training images. max_minutes: a time budget that stops training after an epoch.
# checkpoint: a saved file of the training state. last.pt = latest epoch (used to resume), best.pt = best
#   validation accuracy, log.csv = one row of numbers per epoch.
# resume: continue training from last.pt instead of starting again (like after a Kaggle disconnect).
# inversion / centring: to_dhcd turns dark ink on white paper into white ink on black and puts it in the middle.
# ======================================================================================================

# ======================================================================================================
# Q&A (FILE LEVEL)
# Q: Which part of the project does this file protect? A: Model A, the character CNN in src/char_cnn/: the
#    dataset code (dataset.py), the network (model.py), training and resume (train.py) and prediction (predict.py).
# Q: How is it run? A: From the repository root: python -m pytest tests/test_char_cnn.py. It needs torch and Pillow.
# Q: Why is no real dataset needed? A: _make_dhcd builds a tiny fake DHCD in tmp_path (3 images x 46 classes x 2
#    splits). The tests check that the pieces work together and have the right shapes, not the real accuracy.
# Q: Why does it finish in seconds? A: The images are tiny, there are only 138 per split, and training runs
#    for at most 3 epochs on the CPU.
# ======================================================================================================


def _make_dhcd(root, per_class=3):
    """A tiny fake DHCD: every class gets a few 32x32 images with a different white bar.

    root: a pathlib folder (pytest's tmp_path). per_class: images per class and split.
    Returns root. Creates root/DevanagariHandwrittenCharacterDataset/{Train,Test}/<class>/<k>.png.
    """
    # Q: What does this helper do? A: It writes a fake DHCD folder: root/DevanagariHandwrittenCharacterDataset/
    #    {Train,Test}/<class>/<k>.png, with per_class images in each class folder and returns root.
    # Q: How is each fake image built and why is it enough? A: A black 32x32 uint8 array with one white bar. The bar's
    #    position depends on the class index (so classes differ) and its width on k (so images of one class differ).
    #    The loader only needs valid 32x32 PNGs in the right folders, so this is enough.
    # Q: Which project code uses this folder? A: load_split in dataset.py and run() in train.py read it, exactly as
    #    they would read the real dataset.
    for split in ("Train", "Test"):
        for idx, cls in enumerate(CLASSES):
            d = root / "DevanagariHandwrittenCharacterDataset" / split / cls
            d.mkdir(parents=True)
            for k in range(per_class):
                a = np.zeros((32, 32), dtype=np.uint8)  # black image
                # The bar's position depends on the class index, so each class looks different.
                # The width changes with k, so the images of one class are not all identical.
                a[2 + (idx % 23): 6 + (idx % 23), 2 + idx // 23 * 12: 14 + idx // 23 * 12 + k] = 255
                Image.fromarray(a).save(d / f"{k}.png")
    return root


def test_mapping_is_complete():
    """46 classes, all characters different, and the order starts/ends as expected."""
    # Q: What does this test check? A: That there are exactly 46 classes, that no two characters are the same, and that
    #    the order starts with "क", has the last consonant "ज्ञ" at index 35 and the first digit "०" at index 36.
    # Q: Why does it matter? What bug would make it fail? A: The network outputs a number 0..45 and CHARS turns it back
    #    into a character. A duplicated, missing or re-ordered entry would silently give wrong characters everywhere.
    # Q: How is the input built? A: There is no fake input; it only reads the constants CHARS and NUM_CLASSES.
    # Q: Which function of the project does it exercise? A: The class tables in src/char_cnn/dataset.py.
    assert NUM_CLASSES == 46 == len(set(CHARS))
    # Index 0 = first consonant, 35 = last consonant, 36 = first digit.
    assert CHARS[0] == "क" and CHARS[35] == "ज्ञ" and CHARS[36] == "०"


def test_load_split_and_cache(tmp_path):
    """load_split gives the right shape / type / labels, writes a cache, and the cache gives the same data."""
    # Q: What does this test check? A: load_split returns a uint8 tensor of shape (138,32,32), labels that cover all 46
    #    classes, writes cache_train.npz, and a second call (read from the cache) returns identical data and paths.
    # Q: Why does it matter? What bug would make it fail? A: A wrong shape or dtype would break training. A cache that
    #    saves or restores data differently (different order, values or paths) would make later runs differ from the first.
    # Q: How is the fake input built and why is it enough? A: _make_dhcd(tmp_path) makes 3 images per class. Loading
    #    reads real PNG files from folders, so the same code path as the real dataset runs, only with fewer files.
    # Q: Which function of the project does it exercise? A: load_split (and its cache) in src/char_cnn/dataset.py.
    root = _make_dhcd(tmp_path)
    x, y, paths = load_split(str(root), "Train")
    assert x.shape == (46 * 3, 32, 32) and x.dtype == torch.uint8
    assert sorted(set(y.tolist())) == list(range(46))  # every class index appears
    assert os.path.exists(root / "cache_train.npz")  # the first load wrote the cache
    x2, y2, paths2 = load_split(str(root), "Train")  # from the cache
    assert torch.equal(x, x2) and torch.equal(y, y2) and paths == paths2


def test_load_from_zip_matches_folders(tmp_path):
    """Reading from a zip file must give exactly the same data as reading from folders."""
    # Q: What does this test check? A: That loading the Test split from a zip file gives exactly the same images and
    #    labels as loading it from folders, and that the returned paths point inside the zip ("dhcd.zip/...png").
    # Q: Why does it matter? What bug would make it fail? A: People may put the original UCI zip in data_dir without
    #    unzipping. If the zip loader used another file order or label mapping, the same dataset would train differently.
    # Q: How is the fake input built and why is it enough? A: The fake dataset is made once, then shutil.make_archive
    #    zips it. use_cache=False on both loads so a cache file cannot hide a difference between the two loaders.
    # Q: Which function of the project does it exercise? A: load_split and _load_from_zip in src/char_cnn/dataset.py.
    import shutil

    root = _make_dhcd(tmp_path / "folders")
    # use_cache=False so no cache file hides a difference between the two loaders.
    x, y, _ = load_split(str(root), "Test", use_cache=False)
    zdir = tmp_path / "zipped"
    zdir.mkdir()
    # Zip the fake dataset into zipped/dhcd.zip, then load it through the zip path.
    shutil.make_archive(str(zdir / "dhcd"), "zip", root)
    xz, yz, pz = load_split(str(zdir), "Test", use_cache=False)
    assert torch.equal(x, xz) and torch.equal(y, yz)
    assert pz[0].endswith(".png") and "dhcd.zip/" in pz[0]  # paths point inside the zip


def test_stratified_split_keeps_every_class():
    """Validation gets 10 % of every class, and train and validation do not overlap."""
    # Q: What does this test check? A: With 10 images per class and val_fraction 0.1, validation gets exactly 1 image
    #    of every class (46 in total) and no index is in both train and validation.
    # Q: Why does it matter? What bug would make it fail? A: A plain random split could leave a class out of
    #    validation. Overlap between train and validation is data leakage and would make validation accuracy too high.
    # Q: How is the fake input built and why is it enough? A: np.repeat(np.arange(46), 10) makes a label array only;
    #    the split works on labels, so no images are needed.
    # Q: Which function of the project does it exercise? A: stratified_split in src/char_cnn/dataset.py (seed 42).
    labels = np.repeat(np.arange(46), 10)  # 10 images for each of the 46 classes
    tr, va = stratified_split(labels, 0.1, seed=42)
    # 10 % of 10 = 1 image per class -> 46 in total, and all 46 classes are present.
    assert len(va) == 46 and len(set(labels[va])) == 46
    assert not set(tr) & set(va)  # no image in both sets (no leakage)


def test_model_and_augmentation_shapes():
    """Augmentation keeps the batch shape, and the model maps (B,1,32,32) to (B,46)."""
    # Q: What does this test check? A: random_affine returns a tensor with the same shape as its input, and CharCNN
    #    turns a batch of shape (4,1,32,32) into scores of shape (4,46).
    # Q: Why does it matter? What bug would make it fail? A: A wrong shape would crash training, e.g. a missing channel
    #    dimension, a wrong pooling size or a final layer that is not 46 wide.
    # Q: How is the fake input built and why is it enough? A: 4 random uint8 images through to_input. Shape checks do
    #    not depend on the pixel values, so random numbers are enough.
    # Q: Which function of the project does it exercise? A: to_input and random_affine (dataset.py), CharCNN (model.py).
    x = to_input(torch.randint(0, 255, (4, 32, 32), dtype=torch.uint8))  # (4,1,32,32)
    assert random_affine(x).shape == x.shape
    assert CharCNN()(x).shape == (4, 46)


def test_train_resume_and_predict(tmp_path):
    """Train for 1 epoch, stop on the time limit, resume to the end, then predict with best.pt."""
    # Q: What does this test check? A: The whole life of a run: train, stop early on the time limit after epoch 0 with
    #    last.pt, best.pt and log.csv written, resume to epoch 2, keep one log (header + 3 rows), then predict.
    # Q: Why does it matter? What bug would make it fail? A: Kaggle can disconnect. If resume restarted the log,
    #    repeated epochs, or the checkpoint could not be loaded for prediction, hours of training would be lost.
    # Q: How is the fake input built and why is it enough? A: A fake DHCD in tmp_path, 3 epochs, val_fraction 0.34 (so
    #    each class gets 1 validation image) and max_minutes=0.0001 to force an early stop. It tests flow, not accuracy.
    # Q: Which function of the project does it exercise? A: load_config and run (train.py), then load_model and predict
    #    (predict.py). Prediction on a blank white image only needs to return one of the 46 characters.
    root = _make_dhcd(tmp_path)
    # Small settings for speed. val_fraction=0.34 so 3 images per class give at least 1 validation image.
    cfg = load_config(None, data_dir=str(root), epochs=3, batch_size=32, val_fraction=0.34,
                      ckpt_dir=str(tmp_path / "ckpt"))
    cfg["max_minutes"] = 0.0001  # cut short after epoch 0, like a Kaggle disconnect
    out = run(cfg)
    assert out["last_epoch"] == 0  # only epoch 0 ran
    # The three output files must exist.
    for name in ("last.pt", "best.pt", "log.csv"):
        assert os.path.exists(tmp_path / "ckpt" / name)

    cfg["max_minutes"] = 0  # no time limit now
    out2 = run(cfg, resume=True)  # continue from last.pt
    assert out2["last_epoch"] == 2  # epochs 1 and 2 ran, so the last is 2 (3 epochs in total)
    rows = list(csv.reader(open(tmp_path / "ckpt" / "log.csv", encoding="utf-8")))
    assert len(rows) == 1 + 3  # header + 3 epochs: the resumed run appended, it did not restart the log

    m = load_model(str(tmp_path / "ckpt" / "best.pt"))
    img = Image.new("RGB", (80, 60), "white")  # dark-on-light input of another size
    assert predict(m, img) in CHARS  # the answer must be one of the 46 characters


def test_to_dhcd_inverts_and_centres():
    """to_dhcd turns dark ink on white paper into a centred white character on black, 32x32."""
    # Q: What does this test check? A: A 50x70 picture of a black square on white paper becomes a 32x32 array with the
    #    ink white in the middle (pixel [16,16] above 200) and a black border (pixel [0,0] equals 0).
    # Q: Why does it matter? What bug would make it fail? A: The CNN was trained on white-on-black DHCD images. If
    #    to_dhcd did not invert, crop and centre, real photos or scans would look different and be misclassified.
    # Q: How is the fake input built and why is it enough? A: A white numpy array with a 10x10 black block, turned into
    #    a PIL image. The ink is the only dark area, so inversion, cropping and centring are easy to verify.
    # Q: Which function of the project does it exercise? A: to_dhcd in src/char_cnn/predict.py.
    a = np.full((50, 70), 255, dtype=np.uint8)  # white paper
    a[10:20, 30:40] = 0  # black ink on white paper
    out = to_dhcd(Image.fromarray(a))
    assert out.shape == (32, 32)
    # The centre pixel is bright (ink, inverted to white) and the corner is black (the padding border).
    assert out[16, 16] > 200 and out[0, 0] == 0  # ink is white and centred, border is black
