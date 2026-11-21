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

# CONCEPTS USED IN THIS FILE
# - CNN (convolutional neural network): a network that finds patterns in images with small sliding filters. The whole
#   CharCNN class is one.
# - Convolution (Conv2d 3x3): a 3x3 filter slides over the image and takes weighted sums, so it detects strokes and
#   edges. Used twice per conv_block; padding=1 keeps the height and width.
# - Channel / feature map: each filter makes one output map. Channels grow 1 -> 32 -> 64 -> 128 -> 256 (the channels
#   tuple in CharCNN.__init__).
# - BatchNorm (BatchNorm2d): rescales each channel's outputs using the batch mean and spread, which makes training
#   faster and steadier. Placed after every conv in conv_block.
# - ReLU: the function max(0, x). It adds non-linearity so stacked layers can learn more than one linear map. Placed
#   after every BatchNorm.
# - Max-pool (MaxPool2d(2)): keeps the biggest value in each 2x2 area, halving height and width and making the model
#   less sensitive to small shifts. Last layer of conv_block.
# - Global average pooling (AdaptiveAvgPool2d(1)): averages each channel's whole map to one number, so the (B, 256, 2,
#   2) maps become (B, 256). Has no weights.
# - Dropout: randomly zeroes a share (0.4) of the features during training so the model cannot rely on a few of them.
#   Applied in forward just before the linear layer; off in eval mode.
# - Linear (fully connected) layer: multiplies the 256 features by a weight matrix to give 46 class scores. It is
#   self.fc.
# - Logits: raw class scores before softmax. forward returns them; the loss in train.py applies softmax.
# - Parameters: all the trainable numbers (weights) of the model. count_params adds them up.
# - nn.Module / nn.Sequential / forward: PyTorch's building blocks. Module holds layers, Sequential runs layers in
#   order, forward defines how data flows.
# - Batch size B and tensor shape: B images go through together; shapes like (B, 1, 32, 32) mean batch, channels,
#   height, width.

# FILE Q&A (viva)
# Q: What part of the project is this file?
# A: Model A, the character classifier: a small CNN that maps one 32x32 grayscale image to 46 class scores. It only
#    defines the network.
# Q: How is it run?
# A: It is imported by train.py, evaluate.py and predict.py. Running 'python -m src.char_cnn.model' runs a self-test:
#    a random batch of 4 images, prints the output shape [4, 46] and the parameter count.
# Q: What does it read and write?
# A: Nothing on disk. Training weights are saved and loaded by train.py, evaluate.py and predict.py.


def conv_block(c_in, c_out):
    """One building block of the CNN: two 3x3 convolutions, then a 2x2 max-pool.

    Input:  c_in  = number of input channels, c_out = number of output channels.
    Returns: an nn.Sequential. For an input (B, c_in, H, W) it outputs (B, c_out, H/2, W/2).
    Why it exists: the network is the same block repeated 4 times, so we write it once (DRY).
    """
    # Q: What does conv_block do and why does it exist?
    # A: It builds one block: (Conv3x3, BatchNorm, ReLU) twice, then a 2x2 max-pool. It changes channels from c_in to
    #    c_out and halves height and width. The network uses it 4 times, so it is written once.
    # Q: Why two convs per block and why bias=False?
    # A: Two 3x3 convs see a 5x5 area with fewer weights than one 5x5 conv and add one more ReLU. bias=False because
    #    the BatchNorm right after the conv has its own shift, so a conv bias would be redundant.
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
    # Q: What is CharCNN?
    # A: The whole Model A network as an nn.Module: 4 conv blocks, global average pooling, dropout and one linear
    #    layer. It turns a (B, 1, 32, 32) batch into (B, 46) class scores.
    # Q: Why does the network not end with a softmax?
    # A: CrossEntropyLoss applies log-softmax itself (more stable), and argmax of the logits already gives the same
    #    class as argmax of the probabilities.

    def __init__(self, num_classes=46, channels=(32, 64, 128, 256), dropout=0.4):
        """num_classes: size of the output layer. channels: width of each conv block.
        dropout: chance of zeroing a feature during training (0.4 = 40 %)."""
        # Q: What does __init__ do and why does it exist?
        # A: It creates the layers: one conv_block per entry in channels, then the pooling, dropout and linear layers.
        #    They are stored as attributes so PyTorch registers and trains their weights.
        # Q: Why does c_in start at 1, and why is the linear layer built with c_in?
        # A: Images are grayscale, so the first block takes 1 channel. After the loop c_in is the last block's
        #    channels (256), which is the feature size after pooling, so Linear(256, 46) matches. A wrong size would
        #    give a shape error.
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
        # Q: What does forward do and why does it exist?
        # A: It runs one batch through the network: conv blocks, global average pool, flatten, dropout, linear.
        #    PyTorch calls it when you write model(x). It returns the logits.
        # Q: Why flatten(1) before the linear layer?
        # A: The pool gives (B, 256, 1, 1). flatten(1) keeps the batch dimension and joins the rest into (B, 256),
        #    which is what Linear expects. Dropout is applied to these features and is only active in train mode.
        # features: (B,1,32,32) -> (B,256,2,2); pool: -> (B,256,1,1); flatten(1): -> (B,256).
        x = self.pool(self.features(x)).flatten(1)
        # dropout on the (B,256) features, then linear: (B,256) -> (B,46).
        return self.fc(self.dropout(x))


def count_params(model):
    """Number of trainable parameters (weights) in a model. Used to report model size."""
    # Q: What does count_params do and why does it exist?
    # A: It adds up the number of values in every trained weight tensor. train.py prints it at start and evaluate.py
    #    stores it in metrics.json, to report the model size.
    # Q: Why filter on requires_grad?
    # A: Only parameters with requires_grad=True are updated by training, so only they count as trainable. In this
    #    model all parameters are trainable.
    # numel() = number of values in one weight tensor; requires_grad=True means it is trained.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Quick self-test: run this file directly to check the output shape and the parameter count.
if __name__ == "__main__":
    import torch

    m = CharCNN()
    # Random fake batch of 4 images. Expected output shape: torch.Size([4, 46]).
    print(m(torch.randn(4, 1, 32, 32)).shape, f"{count_params(m):,} parameters")
