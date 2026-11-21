"""Data for Model A: the UCI Devanagari Handwritten Character Dataset (DHCD).

Put the UCI zip (dataset #389) in data/raw/dhcd/ as it is: the code reads it directly. Or unzip it:
    data/raw/dhcd/DevanagariHandwrittenCharacterDataset/{Train,Test}/<class_folder>/*.png
    (data/raw/dhcd/{Train,Test}/... works too)

46 classes (36 consonants + 10 digits), 32x32 grayscale PNGs, white character on a black
background. Train: 1,700 images per class, Test: 300 per class.

The images are tiny (92,000 x 1 KB), so we load them all into one uint8 tensor once and keep
it in memory. That is much faster on Kaggle than opening PNG files every epoch. The first load
writes a cache file (cache_<split>.npz) next to the data so later runs start in seconds.

Augmentation runs on a whole batch at once on the GPU (random_affine below): small rotation,
shift, scale and shear. Never a flip, because a mirrored character is a different shape.

Where this file fits in the pipeline:
    This is the first step for Model A. It turns the raw PNG files into tensors, defines the
    class list (index <-> Devanagari character), splits train / validation, normalises the pixels
    and augments the images. train.py, evaluate.py and predict.py all import from here, so every
    stage uses the same class order and the same normalisation.
"""
# os: file and folder paths.
import os

# numpy: image arrays and the random generator for the split. torch: tensors.
# F (torch.nn.functional): affine_grid and grid_sample, used for augmentation.
import numpy as np
import torch
import torch.nn.functional as F

# Class folder -> Devanagari character. The folder names come from the dataset itself
# (verified from the UCI zip). The numbers follow the order of the Devanagari alphabet.
CLASS_TO_CHAR = {
    "character_1_ka": "क", "character_2_kha": "ख", "character_3_ga": "ग", "character_4_gha": "घ",
    "character_5_kna": "ङ", "character_6_cha": "च", "character_7_chha": "छ", "character_8_ja": "ज",
    "character_9_jha": "झ", "character_10_yna": "ञ", "character_11_taamatar": "ट",
    "character_12_thaa": "ठ", "character_13_daa": "ड", "character_14_dhaa": "ढ",
    "character_15_adna": "ण", "character_16_tabala": "त", "character_17_tha": "थ",
    "character_18_da": "द", "character_19_dha": "ध", "character_20_na": "न", "character_21_pa": "प",
    "character_22_pha": "फ", "character_23_ba": "ब", "character_24_bha": "भ", "character_25_ma": "म",
    "character_26_yaw": "य", "character_27_ra": "र", "character_28_la": "ल", "character_29_waw": "व",
    "character_30_motosaw": "श", "character_31_petchiryakha": "ष", "character_32_patalosaw": "स",
    "character_33_ha": "ह", "character_34_chhya": "क्ष", "character_35_tra": "त्र",
    "character_36_gya": "ज्ञ",
    "digit_0": "०", "digit_1": "१", "digit_2": "२", "digit_3": "३", "digit_4": "४",
    "digit_5": "५", "digit_6": "६", "digit_7": "७", "digit_8": "८", "digit_9": "९",
}
# Class index i <-> CLASSES[i]. Alphabet order (not alphabetical order of folder names).
# The network predicts a number 0..45; this list turns the number back into a folder name / character.
CLASSES = list(CLASS_TO_CHAR)
CHARS = [CLASS_TO_CHAR[c] for c in CLASSES]
NUM_CLASSES = len(CLASSES)  # 46

# Pixel statistics used to normalise the input (roughly the DHCD mean/std of values in [0, 1]).
# Normalising (x - mean) / std gives inputs near zero mean and unit spread, which trains more steadily.
MEAN, STD = 0.17, 0.35


def find_split_dir(data_dir, split):
    """Return the folder that holds the class sub-folders for 'Train' or 'Test'.

    data_dir: dataset root (str). split: "Train" or "Test".
    Why: the unzipped data can sit directly in data_dir or inside the
    DevanagariHandwrittenCharacterDataset folder, so we try both places.
    Raises FileNotFoundError if neither exists.
    """
    # Try the two known layouts in order and return the first folder that exists.
    for cand in (os.path.join(data_dir, split),
                 os.path.join(data_dir, "DevanagariHandwrittenCharacterDataset", split)):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        f"No '{split}' folder under {data_dir}. Unzip the UCI DHCD zip into {data_dir} first.")


def find_zip(data_dir):
    """The UCI zip itself, if it is in data_dir (reading one zip is much faster than 92,000 files on Windows).

    Returns the path of the first .zip file (sorted by name), or None if there is no zip.
    """
    if os.path.isdir(data_dir):
        # sorted() makes the choice the same on every machine.
        for name in sorted(os.listdir(data_dir)):
            if name.lower().endswith(".zip"):
                return os.path.join(data_dir, name)
    return None


def _load_from_zip(zip_path, split):
    """Read all PNGs of one split straight from the UCI zip, without unzipping.

    Input: zip_path (str), split ("Train" or "Test").
    Returns three lists, sorted by class index: images (each uint8 array 32x32),
    labels (int class indices), paths (strings like "<zip>/<name inside zip>").
    """
    # Imports here because they are only needed on this path.
    import io
    import zipfile

    from PIL import Image

    # Folder name -> class index, so the label follows the CLASSES order above.
    cls_index = {c: i for i, c in enumerate(CLASSES)}
    images, labels, paths = [], [], []
    with zipfile.ZipFile(zip_path) as z:
        # sorted() so the load order does not depend on the zip's internal order.
        for name in sorted(z.namelist()):
            # A wanted file looks like .../<split>/<class_folder>/<file>.png, so split is parts[-3].
            parts = name.split("/")
            if not name.lower().endswith(".png") or len(parts) < 3 or parts[-3] != split:
                continue
            # A class folder we do not know means the dataset is not what we expect: stop loudly.
            if parts[-2] not in cls_index:
                raise ValueError(f"unknown class folder {parts[-2]} in {zip_path}")
            # Read the PNG bytes from the zip; convert to grayscale ("L"); force 32x32; store as uint8.
            with Image.open(io.BytesIO(z.read(name))) as im:
                images.append(np.asarray(im.convert("L").resize((32, 32)), dtype=np.uint8))
            labels.append(cls_index[parts[-2]])
            paths.append(f"{zip_path}/{name}".replace("\\", "/"))
    if not images:
        raise FileNotFoundError(f"no '{split}' images found in {zip_path}")
    # Sort by class (stable keeps the file order inside a class) so the zip gives the same
    # order as the folder loader in load_split. Both routes then produce identical data.
    order = np.argsort(np.asarray(labels), kind="stable")  # same order as the folder loader: by class
    return [images[i] for i in order], [labels[i] for i in order], [paths[i] for i in order]


def load_split(data_dir, split, use_cache=True):
    """Load a whole split. Returns (images uint8 [N,32,32], labels int64 [N], relative paths list).

    Reads the unzipped folders if they exist, otherwise the UCI zip placed in data_dir.
    Order of tries: 1) the cache file, 2) unzipped folders, 3) the zip.
    use_cache: if True, read the cache when present and write it after a first load.
    """
    # Cache file name, e.g. cache_train.npz. Loading it takes seconds instead of reading 92,000 files.
    cache = os.path.join(data_dir, f"cache_{split.lower()}.npz")
    if use_cache and os.path.exists(cache):
        # allow_pickle=False is a safety choice: the cache only holds plain arrays.
        z = np.load(cache, allow_pickle=False)
        return torch.from_numpy(z["images"]), torch.from_numpy(z["labels"]), [str(p) for p in z["paths"]]

    try:
        split_dir = find_split_dir(data_dir, split)
    except FileNotFoundError:
        # No unzipped folders: fall back to the zip. If there is no zip either, re-raise the original error.
        zip_path = find_zip(data_dir)
        if zip_path is None:
            raise
        images, labels, paths = _load_from_zip(zip_path, split)
        return _finish(images, labels, paths, cache if use_cache else None)

    from PIL import Image

    images, labels, paths = [], [], []
    # Go through the classes in CLASSES order so the label idx is the class index.
    for idx, cls in enumerate(CLASSES):
        folder = os.path.join(split_dir, cls)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"missing class folder {folder}")
        # sorted() for a fixed file order on every machine.
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(".png"):
                continue
            path = os.path.join(folder, name)
            # Grayscale, forced to 32x32, stored as uint8 (0..255).
            with Image.open(path) as im:
                images.append(np.asarray(im.convert("L").resize((32, 32)), dtype=np.uint8))
            labels.append(idx)
            # Forward slashes so paths look the same on Windows and Linux.
            paths.append(path.replace("\\", "/"))
    return _finish(images, labels, paths, cache if use_cache else None)


def _finish(images, labels, paths, cache):
    """Turn the lists into arrays, optionally save the cache, and return torch tensors.

    images: list of uint8 [32,32] arrays; labels: list of ints; paths: list of str;
    cache: file path to write, or None for no cache.
    Returns (images uint8 [N,32,32], labels int64 [N], paths list).
    """
    images = np.stack(images)  # list of N arrays (32,32) -> one array (N,32,32)
    labels = np.asarray(labels, dtype=np.int64)  # int64 is the label type CrossEntropyLoss expects
    if cache:
        np.savez(cache, images=images, labels=labels, paths=np.asarray(paths))
    # from_numpy shares memory with the array (no copy).
    return torch.from_numpy(images), torch.from_numpy(labels), paths


def stratified_split(labels, val_fraction=0.1, seed=42):
    """Split indices into train / val so that every class keeps the same share in val.

    labels: class index per image (array-like, length N). val_fraction: share for validation.
    seed: makes the split the same on every run.
    Returns (train_idx, val_idx): two sorted numpy arrays of indices into the data.
    Why stratified: a plain random split could give a class too few validation images.
    Why a validation set: we choose the best epoch and stop early with it, so the Test split stays untouched.
    """
    # A fixed seed gives the same split every run, so results can be compared and repeated.
    g = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)  # positions of all images of class c
        g.shuffle(idx)  # random order inside the class
        n_val = int(round(len(idx) * val_fraction))  # same share of every class goes to validation
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    # Sorted so the order of the data is stable (the split itself is already random).
    return np.sort(np.array(train_idx)), np.sort(np.array(val_idx))


def to_input(images_uint8):
    """uint8 [N,32,32] -> float [N,1,32,32], normalised. Works on CPU or GPU tensors.

    Steps: to float, scale 0..255 to 0..1, add a channel dimension, then (x - MEAN) / STD.
    The channel dimension is needed because Conv2d expects (batch, channels, height, width).
    """
    # .float() makes float32; div_(255.0) is in place (no extra memory); unsqueeze(1): (N,32,32) -> (N,1,32,32).
    x = images_uint8.float().div_(255.0).unsqueeze(1)
    return (x - MEAN) / STD


def random_affine(x, degrees=10.0, translate=0.1, scale=(0.9, 1.1), shear=5.0):
    """Random rotation / shift / scale / shear for a batch x [B,1,H,W] (no flip).

    Builds one 2x3 affine matrix per image and resamples with grid_sample. The background is
    black (the lowest value), so the border is filled with the normalised value of 0.

    Inputs: x = normalised float batch (B,1,H,W). degrees: max rotation (+/-). translate: max
    shift as a fraction of the image size. scale: (min, max) zoom factor. shear: max shear (+/-) in degrees.
    Returns a tensor with the same shape as x.
    Why augment: handwriting varies; small random changes make the model see more variety
    and overfit less. Each image gets its own random numbers.
    """
    b = x.shape[0]
    dev = x.device  # create the random numbers on the same device (GPU or CPU) as the data
    rad = lambda d: d * torch.pi / 180.0  # noqa: E731
    # One random value per image (shape (B,)), uniform in the allowed range.
    angle = rad(torch.empty(b, device=dev).uniform_(-degrees, degrees))
    sh = rad(torch.empty(b, device=dev).uniform_(-shear, shear))
    s = torch.empty(b, device=dev).uniform_(*scale)
    tx = torch.empty(b, device=dev).uniform_(-translate, translate) * 2  # grid coords span [-1, 1]
    ty = torch.empty(b, device=dev).uniform_(-translate, translate) * 2
    cos, sin = torch.cos(angle), torch.sin(angle)
    # rotation * shear, divided by scale (grid_sample maps output -> input coordinates)
    # Each inner stack builds one matrix row for all images: (B,) x3 -> (B,3).
    # The outer stack joins the two rows: -> (B,2,3), one 2x3 affine matrix per image.
    theta = torch.stack([
        torch.stack([cos, -sin + torch.tan(sh) * cos, tx], 1),
        torch.stack([sin, cos + torch.tan(sh) * sin, ty], 1),
    ], 1) / s.view(b, 1, 1)
    theta[:, :, 2] *= s.view(b, 1)  # keep the shift in grid units
    # affine_grid makes the sampling grid (B,H,W,2): for each output pixel, where to read in the input.
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    # Normalised black. Pixels that come from outside the image must look like background.
    background = (0.0 - MEAN) / STD
    # grid_sample pads with 0 outside the image. So we shift the data so background becomes 0,
    # resample, and shift it back. Then the new border is background, not a grey band.
    out = F.grid_sample(x - background, grid, align_corners=False, padding_mode="zeros")
    return out + background
