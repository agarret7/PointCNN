import torch
import torch.nn as nn
import numpy as np

try:
    pass
    # from .util import endchannels
except:
    # from util import endchannels
    pass

def endchannels(f, make_contiguous = False):
    class wrapped_layer(nn.Module):
        def __init__(self):
            super(wrapped_layer, self).__init__()
        def forward(self, x):
            x = x.permute(0,3,1,2)
            x = f(x)
            x = x.permute(0,2,3,1)
            return x
    return wrapped_layer()

def DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size, depth_multiplier):
    """
    Depthwise separable convolution
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size, groups = in_channels),
        nn.Conv2d(in_channels * depth_multiplier, out_channels, 1)
    )

class LayerNorm(nn.Module):
    """
    Batch Normalization over ONLY the mini-batch layer
    (suitable for nn.Linear layers).
    """

    def __init__(self, N, D, *args, **kwargs):
        super(LayerNorm, self).__init__()
        if D == 1:
            self.bn = nn.BatchNorm1d(N, *args, **kwargs)
        elif D == 2:
            self.bn = nn.BatchNorm2d(N, *args, **kwargs)
        elif D == 3:
            self.bn = nn.BatchNorm3d(N, *args, **kwargs)
        else:
            raise ValueError("Dimensionality %i not supported" % D)

        self.forward = lambda x: self.bn(x.unsqueeze(0)).squeeze(0)

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
                          LayerNorm(D = 2, momentum = 0.9)
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
