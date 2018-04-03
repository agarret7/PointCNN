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
            self.f = f
        def forward(self, x):
            x = x.permute(0,3,1,2)
            x = self.f(x)
            x = x.permute(0,2,3,1)
            return x
    return wrapped_layer()

class Dense(nn.Module):

    def __init__(self, in_features, out_features, drop_rate = 0, with_bn = True,
                 activation = nn.ReLU()):
        super(Dense, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        # self.bn = LayerNorm(out_channels) if with_bn else None
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else None

    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        # if self.bn:
        #     x = self.bn(x)
        if self.drop:
            x = self.drop(x)
        return x

class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, with_bn = True,
                 activation = nn.ReLU()):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias = not with_bn)
        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels) if with_bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x

class SepConv(nn.Module):
    """
    Depthwise separable convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size, depth_multiplier = 1,
                 with_bn = True, activation = nn.ReLU()):
        super(SepConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size, groups = in_channels),
            nn.Conv2d(in_channels * depth_multiplier, out_channels, 1, bias = not with_bn)
        )

        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels) if with_bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x

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
