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
"""
import os

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
CLASSES = list(CLASS_TO_CHAR)
CHARS = [CLASS_TO_CHAR[c] for c in CLASSES]
NUM_CLASSES = len(CLASSES)  # 46

# Pixel statistics used to normalise the input (roughly the DHCD mean/std of values in [0, 1]).
MEAN, STD = 0.17, 0.35


def find_split_dir(data_dir, split):
    """Return the folder that holds the class sub-folders for 'Train' or 'Test'."""
    for cand in (os.path.join(data_dir, split),
                 os.path.join(data_dir, "DevanagariHandwrittenCharacterDataset", split)):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        f"No '{split}' folder under {data_dir}. Unzip the UCI DHCD zip into {data_dir} first.")


def find_zip(data_dir):
    """The UCI zip itself, if it is in data_dir (reading one zip is much faster than 92,000 files on Windows)."""
    if os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            if name.lower().endswith(".zip"):
                return os.path.join(data_dir, name)
    return None


def _load_from_zip(zip_path, split):
    import io
    import zipfile

    from PIL import Image

    cls_index = {c: i for i, c in enumerate(CLASSES)}
    images, labels, paths = [], [], []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            parts = name.split("/")
            if not name.lower().endswith(".png") or len(parts) < 3 or parts[-3] != split:
                continue
            if parts[-2] not in cls_index:
                raise ValueError(f"unknown class folder {parts[-2]} in {zip_path}")
            with Image.open(io.BytesIO(z.read(name))) as im:
                images.append(np.asarray(im.convert("L").resize((32, 32)), dtype=np.uint8))
            labels.append(cls_index[parts[-2]])
            paths.append(f"{zip_path}/{name}".replace("\\", "/"))
    if not images:
        raise FileNotFoundError(f"no '{split}' images found in {zip_path}")
    order = np.argsort(np.asarray(labels), kind="stable")  # same order as the folder loader: by class
    return [images[i] for i in order], [labels[i] for i in order], [paths[i] for i in order]


def load_split(data_dir, split, use_cache=True):
    """Load a whole split. Returns (images uint8 [N,32,32], labels int64 [N], relative paths list).

    Reads the unzipped folders if they exist, otherwise the UCI zip placed in data_dir.
    """
    cache = os.path.join(data_dir, f"cache_{split.lower()}.npz")
    if use_cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return torch.from_numpy(z["images"]), torch.from_numpy(z["labels"]), [str(p) for p in z["paths"]]

    try:
        split_dir = find_split_dir(data_dir, split)
    except FileNotFoundError:
        zip_path = find_zip(data_dir)
        if zip_path is None:
            raise
        images, labels, paths = _load_from_zip(zip_path, split)
        return _finish(images, labels, paths, cache if use_cache else None)

    from PIL import Image

    images, labels, paths = [], [], []
    for idx, cls in enumerate(CLASSES):
        folder = os.path.join(split_dir, cls)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"missing class folder {folder}")
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(".png"):
                continue
            path = os.path.join(folder, name)
            with Image.open(path) as im:
                images.append(np.asarray(im.convert("L").resize((32, 32)), dtype=np.uint8))
            labels.append(idx)
            paths.append(path.replace("\\", "/"))
    return _finish(images, labels, paths, cache if use_cache else None)


def _finish(images, labels, paths, cache):
    images = np.stack(images)
    labels = np.asarray(labels, dtype=np.int64)
    if cache:
        np.savez(cache, images=images, labels=labels, paths=np.asarray(paths))
    return torch.from_numpy(images), torch.from_numpy(labels), paths


def stratified_split(labels, val_fraction=0.1, seed=42):
    """Split indices into train / val so that every class keeps the same share in val."""
    g = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        g.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.sort(np.array(train_idx)), np.sort(np.array(val_idx))


def to_input(images_uint8):
    """uint8 [N,32,32] -> float [N,1,32,32], normalised. Works on CPU or GPU tensors."""
    x = images_uint8.float().div_(255.0).unsqueeze(1)
    return (x - MEAN) / STD


def random_affine(x, degrees=10.0, translate=0.1, scale=(0.9, 1.1), shear=5.0):
    """Random rotation / shift / scale / shear for a batch x [B,1,H,W] (no flip).

    Builds one 2x3 affine matrix per image and resamples with grid_sample. The background is
    black (the lowest value), so the border is filled with the normalised value of 0.
    """
    b = x.shape[0]
    dev = x.device
    rad = lambda d: d * torch.pi / 180.0  # noqa: E731
    angle = rad(torch.empty(b, device=dev).uniform_(-degrees, degrees))
    sh = rad(torch.empty(b, device=dev).uniform_(-shear, shear))
    s = torch.empty(b, device=dev).uniform_(*scale)
    tx = torch.empty(b, device=dev).uniform_(-translate, translate) * 2  # grid coords span [-1, 1]
    ty = torch.empty(b, device=dev).uniform_(-translate, translate) * 2
    cos, sin = torch.cos(angle), torch.sin(angle)
    # rotation * shear, divided by scale (grid_sample maps output -> input coordinates)
    theta = torch.stack([
        torch.stack([cos, -sin + torch.tan(sh) * cos, tx], 1),
        torch.stack([sin, cos + torch.tan(sh) * sin, ty], 1),
    ], 1) / s.view(b, 1, 1)
    theta[:, :, 2] *= s.view(b, 1)  # keep the shift in grid units
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    background = (0.0 - MEAN) / STD
    out = F.grid_sample(x - background, grid, align_corners=False, padding_mode="zeros")
    return out + background
