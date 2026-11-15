"""Prediction API for Model A, the character CNN (Contract 5).

    from src.char_cnn.predict import load_model, predict
    m = load_model("checkpoints/char_cnn_full/best.pt")
    predict(m, PIL.Image.open("some_char.png"))   # -> "क"

Any input works: RGB or grayscale, any size, dark ink on light paper or the DHCD style
(white on black). The image is turned into the DHCD format first: white character on black,
cropped tightly around the ink, padded to a square, 28x28 plus a 2 px border = 32x32.
"""
import unicodedata

import numpy as np
import torch
from PIL import Image

from .dataset import CHARS, NUM_CLASSES, to_input
from .model import CharCNN


def load_model(ckpt_path: str, device: str = "cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CharCNN(NUM_CLASSES, dropout=ck["cfg"].get("dropout", 0.4)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return {"model": model, "device": device, "chars": ck.get("chars", CHARS)}


def to_dhcd(image: Image.Image) -> np.ndarray:
    """PIL image -> uint8 array (32, 32), white character on black background."""
    a = np.asarray(image.convert("L"), dtype=np.uint8)
    if a.mean() > 127:  # light background -> invert so the ink is white
        a = 255 - a
    ys, xs = np.nonzero(a > 40)  # ink pixels
    if len(ys):
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    side = max(h, w)
    square = np.zeros((side, side), dtype=np.uint8)
    square[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = a
    small = np.asarray(Image.fromarray(square).resize((28, 28), Image.BILINEAR), dtype=np.uint8)
    return np.pad(small, 2)


@torch.no_grad()
def predict(model, image) -> str:
    x = to_input(torch.from_numpy(to_dhcd(image)).unsqueeze(0).to(model["device"]))
    idx = int(model["model"](x).argmax(1))
    return unicodedata.normalize("NFC", model["chars"][idx])


if __name__ == "__main__":
    import sys

    m = load_model(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/char_cnn_full/best.pt")
    for p in sys.argv[2:]:
        print(p, predict(m, Image.open(p)))
