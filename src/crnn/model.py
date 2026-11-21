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

# ===== CONCEPTS USED IN THIS FILE =====
# CRNN: Convolutional Recurrent Neural Network = CNN + recurrent layer + classifier. It is the class CRNN below.
# CNN: convolution layers that scan the image with small filters to find strokes and shapes. Built in self.cnn.
# Conv2d 3x3: a layer with 3x3 filters sliding over the image. padding=1 keeps height and width unchanged (_conv).
# Channels: the number of feature maps. They grow 1 -> 64 -> 128 -> 256 -> 512 in self.cnn.
# BatchNorm2d: rescales each channel's values per batch so training is faster and more stable. Used in _conv.
# ReLU: activation function, max(0, x). It adds non-linearity, so stacked layers can learn complex shapes (_conv).
# MaxPool2d: keeps the largest value in each window, so the feature map gets smaller. (2, 2) halves height and
#    width; (2, 1) halves only height. Used in self.cnn.
# Tensor shape (B, C, H, W): batch size, channels, height, width. The comments in forward() track these shapes.
# Time step: one image column, read from left to right like one letter slot in a sentence. T = W // 4 of them.
# Linear layer (self.proj, self.fc): a matrix multiply that turns each vector into a vector of another size.
# LSTM / BiLSTM: a recurrent layer that reads the columns one after another and remembers context. Bidirectional
#    = one pass left-to-right plus one right-to-left, so each column sees both neighbours (self.rnn).
# Dropout: randomly zeroes some values during training so the model does not memorise the data (self.drop).
# Logits and log_softmax: raw class scores become log-probabilities that sum to 1 (in prob. space) per column.
# CTC and blank symbol: the loss in train.py. Class index 0 is the blank, "no character here". The network
#    gives num_classes = characters + 1 scores for every time step.
# nn.Module / nn.Sequential: PyTorch base class for networks / a container that runs layers in order.
# Trainable parameters: the weights that training changes. count_params() counts them.

# ===== VIVA Q&A FOR THIS FILE =====
# Q: What part of the project is this file?
# A: It defines the CRNN network, the model for the word-image recognition task (IIIT-HW-Dev words).
# Q: How is it run?
# A: It is not run on its own. train.py and evaluate.py import CRNN and count_params from it.
# Q: What does it read and write?
# A: It reads nothing from disk. It takes an image batch (B, 1, 64, W) and returns log-probabilities (T, B, C).
# Q: Why does the model output one prediction per column, not one per letter?
# A: The word length is unknown. The model predicts for every column and CTC loss (train.py) aligns the
#    columns with the real letters, so no letter positions are needed.


def _conv(cin, cout):
    """One conv block as a list of layers: Conv 3x3 -> BatchNorm -> ReLU.

    cin / cout: number of input / output channels. Returns a list so it can be unpacked with *
    inside nn.Sequential. Padding=1 keeps height and width the same (the pooling layers shrink them).
    bias=False because BatchNorm directly after the conv has its own shift, so a bias adds nothing.
    """
    # Q: What does it do and why does it exist?
    # A: It returns the three layers Conv 3x3, BatchNorm, ReLU. The CNN repeats this block six times, so one
    #    helper avoids writing the same three layers again and again.
    # Q: Why bias=False and padding=1?
    # A: BatchNorm right after the conv already adds a learned shift, so a conv bias would be redundant.
    #    padding=1 with a 3x3 filter keeps height and width the same; only the pooling layers shrink them.
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for word images.

    num_classes: number of outputs per column = number of characters + 1 (index 0 is the CTC blank).
    hidden: size of the LSTM hidden state and of the projection before it.
    lstm_layers: how many stacked LSTM layers.
    dropout: dropout probability (between LSTM layers and before the final linear layer).
    """
    # Q: What does this class do and why does it exist?
    # A: It is the whole word recogniser: CNN to see the image, BiLSTM to read the columns in order, and a
    #    linear layer that scores every class for every column. Its output is trained with CTC loss.
    # Q: Why a CNN followed by an LSTM, not only a CNN?
    # A: The CNN finds local shapes. A word is a sequence of letters whose meaning depends on neighbours, and
    #    the BiLSTM carries that left/right context from column to column.

    def __init__(self, num_classes, hidden=256, lstm_layers=2, dropout=0.3):
        # Q: What does __init__ do and why does it exist?
        # A: It creates all layers (cnn, proj, rnn, drop, fc) and stores them as attributes, so PyTorch can
        #    find their weights, train them and save them. forward() then uses them.
        # Q: Why 4 rows (512 * 4 = 2048) before the projection, and why (2, 1) pooling at the end?
        # A: The input height 64 is halved four times: 64 -> 32 -> 16 -> 8 -> 4. The last two pools keep the
        #    width, so the number of time steps stays W / 4. proj must take 512 * 4 inputs to match this.
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
        # Q: What does it do and why does it exist?
        # A: It converts each image's real pixel width into its number of valid time steps. CTC loss and the
        #    decoder need it, because a batch is padded to the widest image and padding is not text.
        # Q: Why is it a staticmethod and why width // 4?
        # A: It does not use any layer or weight, so it needs no self and can be called as CRNN.output_lengths.
        #    The two (2, 2) poolings each halve the width, so width is divided by 4 (rounded down).
        return widths // 4

    def forward(self, x):
        """Run the network.

        x: float tensor (B, 1, 64, W).
        Returns log-probabilities of shape (T, B, num_classes) with T = W // 4.
        """
        # Q: What does it do and why does it exist?
        # A: It is the forward pass: image -> CNN features -> one vector per column -> BiLSTM -> class
        #    log-probabilities per column. PyTorch calls it when you write model(images).
        # Q: Why permute to time-first and why log_softmax over dim 2?
        # A: nn.LSTM (default) and nn.CTCLoss both expect (T, B, ...), so width moves to the first axis. CTCLoss
        #    needs log-probabilities, and dim 2 is the class axis, so each column's classes are normalised.
        f = self.cnn(x)                                 # (B, 512, 4, W/4)
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
    # Q: What does it do and why does it exist?
    # A: It adds up the number of values in every weight that training can change. The number is printed in
    #    train.py and saved in metrics.json, so models can be compared by size.
    # Q: Why the requires_grad filter?
    # A: Only parameters with requires_grad=True are trained. Frozen ones are not counted as trainable.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
