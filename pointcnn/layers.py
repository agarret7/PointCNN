import torch
import torch.nn as nn
import numpy as np

try:
    from .util import endchannels
except:
    from util import endchannels

def SeparableConv2d(in_channels, out_channels, kernel_size):
    """
    Separable convolution (is this correct?)
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = (1, kernel_size)),
        nn.Conv2d(out_channels, in_channels, kernel_size = (kernel_size, 1), groups = in_channels)
    )

def DepthwiseConv2d(in_channel, depth_multiplier):
    """
    Factory function to generate depthwise 2d convolutional layer.
    :param in_channel: TODO
    :param out_channel: TODO
    :param depth_multiplier: TODO
    :return: TODO
    """
    return nn.Conv2d(in_channels, depth_multiplier * in_channels, groups = in_channels)

class BatchNorm(nn.Module):
    """
    PyTorch Linear layers transform shape in the form (N,*,in_features) ->
    (N,*,out_features). BatchNorm normalizes over axis 1. Thus, BatchNorm
    following a linear layer ONLY has the desired behavior if there are no
    additional (*) dimensions. To get the desired behavior, we first transpose
    the channel dim into the last dim, then tranpsose out.
    """

    def __init__(self, D, num_features, *args, **kwargs):
        super(BatchNorm, self).__init__()
        if D == 1:
            self.bn = nn.BatchNorm1d(num_features, *args, **kwargs)
        elif D == 2:
            self.bn = nn.BatchNorm2d(num_features, *args, **kwargs)
        elif D == 3:
            self.bn = nn.BatchNorm3d(num_features, *args, **kwargs)
        else:
            raise ValueError("Dimensionality %i not supported" % D)

        self.forward = endchannels(self.bn, make_contiguous = True)

def MLP(layer_sizes, activation_layer = nn.ReLU(), batch_norm = True):
    """
    Creates a fully connected MLP of arbitrary depth.
    :param layer_sizes: Sizes of MLP hidden layers.
    :param activation_layer: Activation function to be applied in between layers.
    :return: Multilayer perceptron module
    """
    if isinstance(layer_sizes, np.ndarray):
        layer_sizes = layer_sizes.tolist()
    if batch_norm:
        return nn.Sequential(*[
            nn.Sequential(nn.Linear(C_in, C_out),
                          activation_layer,
                          BatchNorm(D = 2, num_features = C_out, momentum = 0.9)
            ) for (C_in, C_out) in zip(layer_sizes, layer_sizes[1:])
        ])
    else:
        return nn.Sequential(*[
            nn.Sequential(nn.Linear(C_in, C_out),
                          activation_layer,
            ) for (C_in, C_out) in zip(layer_sizes, layer_sizes[1:])
        ])

if __name__ == "__main__":
    ftr_map = torch.autograd.Variable(torch.FloatTensor(2,8,100,100))
    layer = SeparableConv2d(8, 16, 2)
    out = layer(ftr_map)
    print(out)

    test = nn.SpatialConvolutionLocal(8, 16, 100, 100, 100, 100)
