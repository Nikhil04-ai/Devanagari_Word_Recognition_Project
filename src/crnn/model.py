"""CRNN: CNN -> BiLSTM -> per-column character probabilities (Shi et al., TPAMI 2017).

Input  : (B, 1, 64, W) image batch, white = +1.0
Output : (T, B, num_classes) log-probabilities, T = W // 4, ready for nn.CTCLoss.

Where this file fits in the pipeline:
    dataset.py (images) -> model.py (this file, the network) -> decode.py (probabilities -> text)
    train.py trains this network with CTC loss. evaluate.py tests it.

How the network works, in three steps:
    1. CNN: looks at the image and builds features. The height shrinks to 4, the width shrinks by 4.
    2. Each remaining image column (a "time step") becomes one feature vector, left to right.
    3. BiLSTM: reads the columns in both directions, so each column knows its left and right
       neighbours. A linear layer then gives one probability per character for each column.
CTC (see train.py) later aligns these T columns to the real label, so we never need to know
where each character sits in the image.
"""
# torch = tensors and layers; nn = the layer classes (Conv2d, LSTM, Linear ...).
import torch
import torch.nn as nn


def _conv(cin, cout):
    """One conv block as a list of layers: Conv 3x3 -> BatchNorm -> ReLU.

    cin / cout: number of input / output channels. Returns a list so it can be unpacked with *
    inside nn.Sequential. Padding=1 keeps height and width the same (the pooling layers shrink them).
    bias=False because BatchNorm directly after the conv has its own shift, so a bias adds nothing.
    """
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for word images.

    num_classes: number of outputs per column = number of characters + 1 (index 0 is the CTC blank).
    hidden: size of the LSTM hidden state and of the projection before it.
    lstm_layers: how many stacked LSTM layers.
    dropout: dropout probability (between LSTM layers and before the final linear layer).
    """

    def __init__(self, num_classes, hidden=256, lstm_layers=2, dropout=0.3):
        super().__init__()
        # CNN part: (B, 1, 64, W) -> (B, 512, 4, W/4). Channels grow, height shrinks.
        self.cnn = nn.Sequential(
            *_conv(1, 64), nn.MaxPool2d(2, 2),                                   # H 64 -> 32, W / 2
            *_conv(64, 128), nn.MaxPool2d(2, 2),                                 # H 32 -> 16, W / 4
            # Later pools use (2, 1): they halve only the height, so the width (= time steps) stays W / 4.
            *_conv(128, 256), *_conv(256, 256), nn.MaxPool2d((2, 1), (2, 1)),    # H 16 -> 8
            *_conv(256, 512), *_conv(512, 512), nn.MaxPool2d((2, 1), (2, 1)),    # H 8 -> 4
        )
        # Each column has 512 channels x 4 rows = 2048 numbers. This layer squeezes them to `hidden`.
        self.proj = nn.Linear(512 * 4, hidden)
        # Bidirectional LSTM: one pass left-to-right, one right-to-left, outputs joined (2 * hidden).
        # PyTorch only allows dropout between LSTM layers, so it is switched off when there is 1 layer.
        self.rnn = nn.LSTM(hidden, hidden, num_layers=lstm_layers, bidirectional=True,
                           dropout=dropout if lstm_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)  # regularisation: randomly zeroes features during training
        # Final classifier: one score per class (characters + blank) for every column.
        self.fc = nn.Linear(2 * hidden, num_classes)

    @staticmethod
    def output_lengths(widths):
        """Number of CTC time steps for images of the given widths.

        widths: integer tensor (B,) with each image's real width in pixels (before batch padding).
        Returns a tensor (B,). It is width // 4 because the two (2, 2) poolings halve the width twice.
        CTC needs this so it ignores the padded columns of short images.
        """
        return widths // 4

    def forward(self, x):
        """Run the network.

        x: float tensor (B, 1, 64, W).
        Returns log-probabilities of shape (T, B, num_classes) with T = W // 4.
        """
        f = self.cnn(x)                                  # (B, 512, 4, W/4)
        b, c, h, w = f.shape
        # Move width to the first axis so each column is one time step, and merge channel and height.
        # nn.LSTM (default batch_first=False) and nn.CTCLoss both expect time first.
        f = f.permute(3, 0, 1, 2).reshape(w, b, c * h)   # (T, B, 2048)
        f = self.proj(f)                                 # (T, B, hidden)
        f, _ = self.rnn(f)                               # (T, B, 2*hidden)
        # log_softmax over the class axis (dim 2): nn.CTCLoss expects log-probabilities, not raw scores.
        return self.fc(self.drop(f)).log_softmax(2)      # (T, B, num_classes)


def count_params(model):
    """Count the trainable parameters of a model (an int). Used for logging and the results table."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
