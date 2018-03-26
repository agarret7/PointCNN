import time

import torch
import torch.nn as nn

import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    from .context import pytorch_knn_cuda
except SystemError:
    from context import pytorch_knn_cuda

def endchannels(f, make_contiguous = False):
    def wrapped_func(x):
        if make_contiguous:
            return torch.transpose(f(torch.transpose(x, 1, -1).contiguous()), -1, 1)
        else:
            return torch.transpose(f(torch.transpose(x, 1, -1)), -1, 1)
    return wrapped_func

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

def apply_along_dim(xs, f, dim):
    """
    PyTorch analog to np.apply_along_axis.
    :param xs:
    :param dim:
    """
    return torch.stack([f(x) for x in torch.unbind(xs, dim)], dim)

def zipwith_matmul(xs, ys):
    """
    Given two lists of 2D matrices of appropriate size, zips them
    together with matrix multiplication.
    :param xs:
    :param ys:
    """
    # xs of shape [N, n, m]
    # ys of shape [N, m, p]
    # return shape [N, n, p]
    N = len(xs)
    return torch.stack([torch.mm(xs[i], ys[i]) for i in range(N)], dim = 0)

def knn_indices_func(ps, P, k):
    """
    Indexing function based on K-Nearest Neighbors search.
    :type ps: FloatTensor (N, N_rep, D)
    :type P: FloatTensor (N, *, D)
    :type k: int
    :rtype: FloatTensor (N, N_rep, N_neighbors)
    :param ps: Representative point
    :param P: Point cloud to get indices from
    :param k: Number of nearest neighbors to collect.
    :return: Array of indices, P_idx, into P such that P[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in P[n].
    """
    ps = ps.data.numpy()
    P = P.data.numpy()

    def single_batch_knn(p, P_particular):
        nbrs = NearestNeighbors(k + 1, algorithm = "ball_tree").fit(P_particular)
        indices = nbrs.kneighbors(p)[1]
        return indices[:,1:]

    region_idx = np.stack([
        single_batch_knn(p, P[n]) for n, p in enumerate(ps)
    ], axis = 0)
    return torch.from_numpy(region_idx)

def knn_indices_func_gpu(ps, P, k):

    def single_batch_knn(p, P_particular):
        nbrs_f = pytorch_knn_cuda.KNearestNeighbor(k + 1)
        # knn_cuda(k + 1, )
        indices = nbrs_f(P_particular, p)[0]
        return indices[:,1:]

    region_idx = torch.stack([
        single_batch_knn(p, P[n]) for n, p in enumerate(ps)
    ], dim = 0)
    return region_idx


if __name__ == "__main__":
    from torch.autograd import Variable
    N_rep = 100
    N = 2
    num_points = 1000
    D = 3
    test_P  = np.random.rand(N,num_points,D).astype(np.float32)
    idx = np.random.choice(test_P.shape[1], N_rep, replace = False)
    test_ps = test_P[:,idx,:]
    test_P = Variable(torch.from_numpy(test_P)).cuda()
    test_ps = Variable(torch.from_numpy(test_ps)).cuda()
    out = knn_indices_func_gpu(test_ps, test_P, 5)
    print(out)
