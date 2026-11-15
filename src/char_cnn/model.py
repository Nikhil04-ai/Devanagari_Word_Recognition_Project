"""Model A: a small CNN that classifies one 32x32 character image into 46 classes.

    input  (B, 1, 32, 32)
    block1 Conv3x3(32)-BN-ReLU, Conv3x3(32)-BN-ReLU, MaxPool2   -> (B,  32, 16, 16)
    block2 Conv3x3(64)-BN-ReLU, Conv3x3(64)-BN-ReLU, MaxPool2   -> (B,  64,  8,  8)
    block3 Conv3x3(128) ... MaxPool2                            -> (B, 128,  4,  4)
    block4 Conv3x3(256) ... MaxPool2                            -> (B, 256,  2,  2)
    global average pooling                                      -> (B, 256)
    Dropout(0.4), Linear(256, 46)                               -> (B, 46) class scores (logits)
"""
from torch import nn


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CharCNN(nn.Module):
    def __init__(self, num_classes=46, channels=(32, 64, 128, 256), dropout=0.4):
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            layers.append(conv_block(c_in, c))
            c_in = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torch

    m = CharCNN()
    print(m(torch.randn(4, 1, 32, 32)).shape, f"{count_params(m):,} parameters")
