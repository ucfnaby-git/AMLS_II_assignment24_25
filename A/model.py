import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

#########################
# Convolution and Wavelet Modules
#########################


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    """A simple convolution with 'same' padding."""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias
    )


def dwt_init(x):
    # Discrete Wavelet Transform (DWT)
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def iwt_init(x):
    # Inverse Wavelet Transform (IWT)
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_channel = in_channel // (r**2)
    out_height = r * in_height
    out_width = r * in_width

    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel : 2 * out_channel, :, :] / 2
    x3 = x[:, 2 * out_channel : 3 * out_channel, :, :] / 2
    x4 = x[:, 3 * out_channel : 4 * out_channel, :, :] / 2

    h = torch.zeros([in_batch, out_channel, out_height, out_width], device=x.device)
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


class DWT(nn.Module):
    """Wavelet downsampling (DWT)."""

    def __init__(self):
        super(DWT, self).__init__()

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    """Wavelet upsampling (IWT)."""

    def __init__(self):
        super(IWT, self).__init__()

    def forward(self, x):
        return iwt_init(x)


#########################
# Building Block
#########################


class BBlock(nn.Module):
    """
    Basic block: applies a convolution, optional batch normalization, and a ReLU activation.
    A residual scaling factor is applied.
    """

    def __init__(
        self,
        conv,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):
        super(BBlock, self).__init__()
        layers = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(act)
        self.body = nn.Sequential(*layers)
        self.res_scale = res_scale

    def forward(self, x):
        return self.body(x).mul(self.res_scale)


#########################
# MWCNN Network (Modified for sRGB)
#########################


class MWCNN(nn.Module):
    def __init__(self, n_feats=64, conv=default_conv):
        """
        Constructs the Multi-level Wavelet CNN for image denoising using sRGB images.
        For sRGB input, the DWT converts 3 channels into 12 channels.
        The head is modified to accept 12 channels and the tail is modified so that after IWT the final output has 3 channels.
        """
        super(MWCNN, self).__init__()
        act = nn.ReLU(True)
        self.DWT = DWT()
        self.IWT = IWT()
        n = 3  # number of blocks per stage

        # Head: input is sRGB; DWT(x) yields 12 channels.
        self.head = nn.Sequential(BBlock(conv, 12, 160, 3, act=act))
        # Encoder Stage 1
        self.d_l1 = nn.Sequential(
            *[BBlock(conv, 160, 160, 3, act=act) for _ in range(n)]
        )
        # Encoder Stage 2: DWT on 160 channels gives 640 channels.
        self.d_l2 = nn.Sequential(
            BBlock(conv, 640, n_feats * 4, 3, act=act),
            *[BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act) for _ in range(n)]
        )
        # Bottleneck processing
        self.pro_l3 = nn.Sequential(
            BBlock(conv, n_feats * 16, n_feats * 4, 3, act=act),
            *[BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act) for _ in range(n * 2)],
            BBlock(conv, n_feats * 4, n_feats * 16, 3, act=act)
        )
        # Decoder Stage 2
        self.i_l2 = nn.Sequential(
            *[BBlock(conv, n_feats * 4, n_feats * 4, 3, act=act) for _ in range(n)],
            BBlock(conv, n_feats * 4, 640, 3, act=act)
        )
        # Decoder Stage 1
        self.i_l1 = nn.Sequential(
            *[BBlock(conv, 160, 160, 3, act=act) for _ in range(n)]
        )
        # Tail: projects back to DWT domain so that after IWT we obtain 3 channels.
        self.tail = nn.Sequential(conv(160, 12, 3))

    def forward(self, x):
        x1 = self.d_l1(self.head(self.DWT(x)))
        x2 = self.d_l2(self.DWT(x1))
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        x_ = self.IWT(self.i_l2(x_)) + x1
        out = self.IWT(self.tail(self.i_l1(x_))) + x
        return out
