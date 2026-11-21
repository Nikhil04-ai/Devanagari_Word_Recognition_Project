"""Prediction API for Model A, the character CNN (Contract 5).

    from src.char_cnn.predict import load_model, predict
    m = load_model("checkpoints/char_cnn_full/best.pt")
    predict(m, PIL.Image.open("some_char.png"))   # -> "क"

Any input works: RGB or grayscale, any size, dark ink on light paper or the DHCD style
(white on black). The image is turned into the DHCD format first: white character on black,
cropped tightly around the ink, padded to a square, 28x28 plus a 2 px border = 32x32.

Where this file fits in the pipeline:
    This is the last step for Model A. It is the small, easy interface other code (and the
    demo) uses: load a checkpoint once, then call predict() on any picture of one character.
    "Contract 5" is the project's agreed prediction interface.
"""
# unicodedata: normalise the Devanagari text that we return.
import unicodedata

# numpy: image arrays. torch: run the model. PIL: image handling.
import numpy as np
import torch
from PIL import Image

from .dataset import CHARS, NUM_CLASSES, to_input
from .model import CharCNN


def load_model(ckpt_path: str, device: str = "cpu"):
    """Load a trained checkpoint for prediction.

    ckpt_path: path to a .pt file written by train.py. device: "cpu" (default) or "cuda".
    Returns a dict {"model": CharCNN in eval mode, "device": device, "chars": list of 46 characters}.
    Why a dict: predict() needs the model, the device and the class characters together.
    """
    # weights_only=False because the checkpoint also holds plain Python objects (cfg, chars).
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Same architecture as in training, then fill it with the saved weights.
    model = CharCNN(NUM_CLASSES, dropout=ck["cfg"].get("dropout", 0.4)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()  # dropout off, BatchNorm uses its running statistics
    # Prefer the character list saved in the checkpoint; fall back to the one in dataset.py.
    return {"model": model, "device": device, "chars": ck.get("chars", CHARS)}


def to_dhcd(image: Image.Image) -> np.ndarray:
    """PIL image -> uint8 array (32, 32), white character on black background.

    Why: the network was trained on DHCD images (white character, black background, 32x32).
    A photo or scan looks different, so we convert it to the same style first.
    """
    # Any mode (RGB, RGBA, ...) -> grayscale 2-D array, values 0..255.
    a = np.asarray(image.convert("L"), dtype=np.uint8)
    if a.mean() > 127:  # light background -> invert so the ink is white
        # Mean above the middle grey means most pixels are bright, so it is dark ink on light paper.
        a = 255 - a
    ys, xs = np.nonzero(a > 40)  # ink pixels
    # Crop to the box around the ink (threshold 40 ignores faint noise). If no ink is found, keep the whole image.
    if len(ys):
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    side = max(h, w)
    # Put the crop in the middle of a black square, so resizing does not stretch the character.
    square = np.zeros((side, side), dtype=np.uint8)
    square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = a
    # Shrink (or enlarge) to 28x28 with bilinear smoothing.
    small = np.asarray(Image.fromarray(square).resize((28, 28), Image.BILINEAR), dtype=np.uint8)
    # Add a 2 px black border on every side: 28 + 2 + 2 = 32.
    return np.pad(small, 2)


@torch.no_grad()
def predict(model, image) -> str:
    """Classify one character image.

    model: the dict returned by load_model. image: a PIL image of a single character.
    Returns the predicted Devanagari character as a string (NFC-normalised).
    """
    # to_dhcd gives (32,32) uint8 -> tensor -> unsqueeze(0) adds the batch dimension: (1,32,32)
    # -> to_input: float, normalised, with a channel: (1,1,32,32). Same preprocessing as in training.
    x = to_input(torch.from_numpy(to_dhcd(image)).unsqueeze(0).to(model["device"]))
    # Model output (1, 46) logits; argmax(1) = index of the best class.
    idx = int(model["model"](x).argmax(1))
    # NFC gives one standard form of the Unicode text, so comparisons with reference text are reliable.
    return unicodedata.normalize("NFC", model["chars"][idx])


# Command-line use: python -m src.char_cnn.predict [checkpoint] image1.png image2.png ...
if __name__ == "__main__":
    import sys

    # First argument is the checkpoint path; the default is used when no argument is given.
    # Note: if you give only images, the first image would be read as the checkpoint, so give the checkpoint first.
    m = load_model(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/char_cnn_full/best.pt")
    for p in sys.argv[2:]:
        print(p, predict(m, Image.open(p)))
