"""CRNN: CNN -> BiLSTM -> per-column character probabilities (Shi et al., TPAMI 2017).

Input  : (B, 1, 64, W) image batch, white = +1.0
Output : (T, B, num_classes) log-probabilities, T = W // 4, ready for nn.CTCLoss.
"""
import torch
import torch.nn as nn


def _conv(cin, cout):
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    def __init__(self, num_classes, hidden=256, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            *_conv(1, 64), nn.MaxPool2d(2, 2),                                   # H 64 -> 32, W / 2
            *_conv(64, 128), nn.MaxPool2d(2, 2),                                 # H 32 -> 16, W / 4
            *_conv(128, 256), *_conv(256, 256), nn.MaxPool2d((2, 1), (2, 1)),    # H 16 -> 8
            *_conv(256, 512), *_conv(512, 512), nn.MaxPool2d((2, 1), (2, 1)),    # H 8 -> 4
        )
        self.proj = nn.Linear(512 * 4, hidden)
        self.rnn = nn.LSTM(hidden, hidden, num_layers=lstm_layers, bidirectional=True,
                           dropout=dropout if lstm_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, num_classes)

    @staticmethod
    def output_lengths(widths):
        """Number of CTC time steps for images of the given widths."""
        return widths // 4

    def forward(self, x):
        f = self.cnn(x)                                  # (B, 512, 4, W/4)
        b, c, h, w = f.shape
        f = f.permute(3, 0, 1, 2).reshape(w, b, c * h)   # (T, B, 2048)
        f = self.proj(f)
        f, _ = self.rnn(f)                               # (T, B, 2*hidden)
        return self.fc(self.drop(f)).log_softmax(2)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
