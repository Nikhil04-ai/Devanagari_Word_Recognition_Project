"""Model A: a small CNN that classifies one 32x32 character image into 46 classes.

    input  (B, 1, 32, 32)
    block1 Conv3x3(32)-BN-ReLU, Conv3x3(32)-BN-ReLU, MaxPool2   -> (B,  32, 16, 16)
    block2 Conv3x3(64)-BN-ReLU, Conv3x3(64)-BN-ReLU, MaxPool2   -> (B,  64,  8,  8)
    block3 Conv3x3(128) ... MaxPool2                            -> (B, 128,  4,  4)
    block4 Conv3x3(256) ... MaxPool2                            -> (B, 256,  2,  2)
    global average pooling                                      -> (B, 256)
    Dropout(0.4), Linear(256, 46)                               -> (B, 46) class scores (logits)

Where this file fits in the pipeline:
    dataset.py gives the images -> model.py (this file) is the network -> train.py trains it
    -> evaluate.py tests it -> predict.py uses it on new images.
B = batch size (how many images go through the network together).
The 46 classes are 36 consonants + 10 digits of the DHCD dataset.
"""
# nn gives us the layers (Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, Dropout).
from torch import nn


def conv_block(c_in, c_out):
    """One building block of the CNN: two 3x3 convolutions, then a 2x2 max-pool.

    Input:  c_in  = number of input channels, c_out = number of output channels.
    Returns: an nn.Sequential. For an input (B, c_in, H, W) it outputs (B, c_out, H/2, W/2).
    Why it exists: the network is the same block repeated 4 times, so we write it once (DRY).
    """
    return nn.Sequential(
        # 3x3 conv with padding=1 keeps H and W the same: (B, c_in, H, W) -> (B, c_out, H, W).
        # bias=False because BatchNorm right after it has its own shift (bias), so a conv bias is useless.
        # BatchNorm keeps the layer outputs at a stable scale, which makes training faster and steadier.
        # ReLU adds non-linearity (without it, stacked convs would act like one linear layer).
        # inplace=True saves memory by overwriting the input tensor.
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        # Second conv: c_out -> c_out. Two 3x3 convs see a 5x5 area, like one 5x5 conv but with fewer weights.
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        # Max-pool 2x2 halves the height and width: (B, c_out, H, W) -> (B, c_out, H/2, W/2).
        # It keeps the strongest response in each 2x2 area and makes the model less sensitive to small shifts.
        nn.MaxPool2d(2),
    )


class CharCNN(nn.Module):
    """The character classifier: 4 conv blocks, global average pooling, dropout, one linear layer.

    Input to forward: float tensor (B, 1, 32, 32), normalised grayscale images.
    Output of forward: (B, num_classes) raw scores (logits). No softmax here:
    the loss (CrossEntropyLoss) applies it, and for the final answer argmax on logits is enough.
    """

    def __init__(self, num_classes=46, channels=(32, 64, 128, 256), dropout=0.4):
        """num_classes: size of the output layer. channels: width of each conv block.
        dropout: chance of zeroing a feature during training (0.4 = 40 %)."""
        super().__init__()
        # c_in starts at 1 because the images are grayscale (1 channel).
        layers, c_in = [], 1
        for c in channels:
            layers.append(conv_block(c_in, c))  # each block takes the previous block's channels
            c_in = c
        # Shapes: (B,1,32,32) -> (B,32,16,16) -> (B,64,8,8) -> (B,128,4,4) -> (B,256,2,2).
        self.features = nn.Sequential(*layers)
        # Global average pooling: average each channel's 2x2 map to one number -> (B, 256, 1, 1).
        # Compared with flatten + big Linear layer it has no weights, so fewer parameters and less overfitting.
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Dropout randomly switches off features while training so the model cannot rely on a few of them.
        # It is automatically turned off in model.eval().
        self.dropout = nn.Dropout(dropout)
        # c_in is now 256 (the last block's channels). Final layer maps 256 features -> 46 class scores.
        self.fc = nn.Linear(c_in, num_classes)

    def forward(self, x):
        """x: (B, 1, 32, 32) -> logits (B, num_classes)."""
        # features: (B,1,32,32) -> (B,256,2,2); pool: -> (B,256,1,1); flatten(1): -> (B,256).
        x = self.pool(self.features(x)).flatten(1)
        # dropout on the (B,256) features, then linear: (B,256) -> (B,46).
        return self.fc(self.dropout(x))


def count_params(model):
    """Number of trainable parameters (weights) in a model. Used to report model size."""
    # numel() = number of values in one weight tensor; requires_grad=True means it is trained.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Quick self-test: run this file directly to check the output shape and the parameter count.
if __name__ == "__main__":
    import torch

    m = CharCNN()
    # Random fake batch of 4 images. Expected output shape: torch.Size([4, 46]).
    print(m(torch.randn(4, 1, 32, 32)).shape, f"{count_params(m):,} parameters")
