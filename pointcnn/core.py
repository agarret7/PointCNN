"""
PyTorch implementation of the PointCNN paper, as specified in:
  https://arxiv.org/pdf/1801.07791.pdf

Author: Austin J. Garrett

I make liberal use of the mypy static type checker for Python.
It should be mostly intuitive, but further documentation can be found at:
  http://mypy-lang.org/
"""

# Standard Modules
import time

# External Modules
import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable

# Internal Modules
try:
    from .util import knn_indices_func, knn_indices_func_gpu, plot
    from .layers import MLP, LayerNorm, Conv, SepConv, Dense, end_channels
    # from .context import timed
except SystemError:
    from util import knn_indices_func, knn_indices_func_gpu, plot
    from layers import MLP, LayerNorm, Conv, SepConv, Dense, end_channels
    # from context import timed

class XConv(nn.Module):
    """
    Vectorized pointwise convolution.
    """

    def __init__(self, C_in : int, C_out : int, dims : int, K : int,
                 P : int, C_mid : int, depth_multiplier : int) -> None:
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param P: Number of representative points
        :param C_mid: Dimensionality of lifted point features.
        :param depth_multiplier: Depth multiplier for internal depthwise separable convolution.
        """
        super(XConv, self).__init__()

        if __debug__:
            # Only needed for assertions.
            self.C_in = C_in
            self.C_mid = C_mid
            self.dims = dims
            self.K = K

        self.P = P

        # Additional processing layers
        # self.pts_layernorm = LayerNorm(2, momentum = 0.9)

        # Main dense linear layers
        self.dense1 = Dense(dims, C_mid)
        self.dense2 = Dense(C_mid, C_mid)

        # Layers to generate X
        self.x_trans = nn.Sequential(
            end_channels(Conv(
                in_channels = dims,
                out_channels = K*K,
                kernel_size = (1, K),
                with_bn = False
            )),
            Dense(K*K, K*K, with_bn = False),
            Dense(K*K, K*K, with_bn = False, activation = None)
        )

        self.end_conv = end_channels(SepConv(
            in_channels = C_mid + C_in,
            out_channels = C_out,
            kernel_size = (1, K),
            depth_multiplier = depth_multiplier
        )).cuda()

    def forward(self, x : Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """
        Applies XConv to the input data.
        :type rep_pt: (N, P, dims)
        :type pts: (N, P, K, dims)
        :type fts: (N, P, K, C_in)
        :rtype: (TODO: shape)
        :param x: (rep_pt, pts, fts)
        :param rep_pt: Representative point
        :param pts: Regional point cloud such that fts[:,p_idx,:] is the feature associated with pts[:,p_idx,:]
        :param fts: Regional features such that pts[:,p_idx,:] is the feature associated with fts[:,p_idx,:]
        :return: Features aggregated into point rep_pt.
        """
        rep_pt, pts, fts = x
        N = len(pts)

        #== RUNTIME ASSERTIONS ==#
        if fts is not None:
            assert(rep_pt.size()[0] == pts.size()[0] == fts.size()[0])  # Check N is equal.
            assert(rep_pt.size()[1] == pts.size()[1] == fts.size()[1])  # Check P is equal.
            assert(pts.size()[2] == fts.size()[2] == self.K)            # Check K is equal.
            assert(fts.size()[3] == self.C_in)                          # Check C_in is equal.
        else:
            assert(rep_pt.size()[0] == pts.size()[0])                   # Check N is equal.
            assert(rep_pt.size()[1] == pts.size()[1])                   # Check P is equal.
            assert(pts.size()[2] == self.K)                             # Check K is equal.
        assert(rep_pt.size()[2] == pts.size()[3] == self.dims)          # Check dims is equal.
        #========================#

        P = rep_pt.size()[1]  # (N, P, K, dims)
        p_center = torch.unsqueeze(rep_pt, dim = 2)  # (N, P, 1, dims)

        # Move pts to local coordinate system of rep_pt.
        pts_local = pts - p_center  # (N, P, K, dims)
        # pts_local = self.pts_layernorm(pts - p_center)
    
        # Individually lift each point into C_mid space.
        fts_lifted0 = self.dense1(pts_local)
        fts_lifted  = self.dense2(fts_lifted0)  # (N, P, K, C_mid)

        if fts is None:
            fts_cat = fts_lifted
        else:
            fts_cat = torch.cat((fts_lifted, fts), -1)  # (N, P, K, C_mid + C_in)

        # Learn the (N, K, K) X-transformation matrix.
        X_shape = (N, P, self.K, self.K)
        X = self.x_trans(pts_local)
        X = X.view(*X_shape)

        """
        X = self.mid_conv(pts_local)
        X = X.contiguous().view(*X_shape)
        X = self.mid_dwconv1(X)
        X = X.contiguous().view(*X_shape)
        X = self.mid_dwconv2(X)
        X = X.contiguous().view(*X_shape)
        """

        # Weight and permute fts_cat with the learned X.
        fts_X = torch.matmul(X, fts_cat)
        fts_p = self.end_conv(fts_X).squeeze(dim = 2)
        return fts_p

class PointCNN(nn.Module):
    """
    TODO: Insert documentation
    """

    def __init__(self, C_in : int, C_out : int, dims : int, K : int, D : int, P : int,
                 r_indices_func : Callable[[Tensor, Tensor, int, int], LongTensor]) -> None:
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param P: Number of representative points.
        :param D: "Spread" of neighboring points.
        :param r_indices_func: Selector function of the type,
            INP
            ======
            rep_pts : (N, P, dims) Representative points
            pts  : (N, *, dims) Point cloud
            K : Number of points for each region.
            D : "Spread" of neighboring points (analogous to stride).

            OUT
            ======
            pts_idx : (N, P, K) Array of indices into pts such that
            pts[pts_idx] is the set of points in the "region" around rep_pt.

        a representative point rep_pt and a point cloud pts. From these it returns an
        array of K
        :param C_mid: Dimensionality of lifted point features.
        :param mlp_width: Number of hidden layers in MLPs.
        """
        super(PointCNN, self).__init__()

        C_mid = C_out // 2 if C_in == 0 else C_out // 4
        depth_multiplier = min(int(np.ceil(C_out / C_in)), 4)

        self.r_indices_func = r_indices_func
        self.dense = Dense(C_in, C_out // 2) if C_in != 0 else None
        self.x_conv = XConv(C_out // 2 if C_in != 0 else C_in, C_out, dims, K, P, C_mid, depth_multiplier)
        self.D = D

    def select_region(self, pts : Tensor, pts_idx : LongTensor) -> Tensor:
        """
        Selects
        :type pts: (N, *, dims)
        :type pts_idx: (N, P, K)
        :rtype pts_region: (P, K, dims)
        :param pts: Point cloud to select regional points from
        :param pts_idx: Indices of points in region to be selected
        :return:
        """
        regions = torch.stack([
            pts[n][idx,:] for n, idx in enumerate(torch.unbind(pts_idx, dim = 0))
        ], dim = 0)
        return regions

    def forward(self, x : Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """
        Given a set of representative points, a point cloud, and its
        corresponding features, return a new set of representative points with
        features projected from the point cloud.
        :type rep_pts: (N, *, dims)
        :type pts: (N, K, dims)
        :type fts: (N, K, C_in)
        :rtype: (N, P, dims)
        :param rep_pts: Representative points
        :param pts: Regional point cloud such that fts[:,p_idx,:] is the feature associated with pts[:,p_idx,:]
        :param fts: Regional features such that pts[:,p_idx,:] is the feature associated with fts[:,p_idx,:]
        :return:
        """
        rep_pts, pts, fts = x
        fts = self.dense(fts) if fts is not None else fts

        # This step takes ~97% of the time. Prime target for optimization: KNN on GPU.
        pts_idx = self.r_indices_func(rep_pts.cpu(), pts.cpu(), self.x_conv.K, self.D).cuda()
        # -------------------------------------------------------------------------- #

        pts_regional = self.select_region(pts, pts_idx)
        fts_regional = self.select_region(fts, pts_idx) if fts is not None else fts
        fts_p = self.x_conv((rep_pts, pts_regional, fts_regional))

        if False:
            # Draw neighborhood points, for debugging.
            t = 10
            n = 0
            test_point = rep_pts[n,t,:].cpu().data.numpy()
            neighborhood = pts_regional[n,t,:,:].cpu().data.numpy()
            plt.scatter(pts[n][:,0], pts[n][:,1])
            plt.scatter(test_point[0], test_point[1], s = 100, c = 'green')
            plt.scatter(neighborhood[:,0], neighborhood[:,1], s = 100, c = 'red')
            plt.show()
        return fts_p

class RandPointCNN(nn.Module):
    """ PointCNN with randomly subsampled representative points. """

    def __init__(self, *args, **kwargs):
        super(RandPointCNN, self).__init__()
        self.pointcnn = PointCNN(*args, **kwargs)

        # This is safe because PointCNN requires P.
        self.P = args[5] if len(args) > 5 else kwargs['P']

    def forward(self, x :Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        pts, fts = x
        if 0 < self.P < pts.size()[1]:
            # Select random set of indices of subsampled points.
            idx = np.random.choice(pts.size()[1], self.P, replace = False).tolist()
            rep_pts = pts[:,idx,:]
        else:
            # All input points are representative points.
            rep_pts = pts
        rep_pts_fts = self.pointcnn((rep_pts, pts, fts))
        return rep_pts, rep_pts_fts

if __name__ == "__main__":
    np.random.seed(0)

    N = 1
    num_points = 1000
    dims = 2
    C_in = 4
    K = 10
    D = 1

    layer1 = RandPointCNN(C_in,   8, dims, K, D, 1000, knn_indices_func).cuda()
    layer2 = RandPointCNN(   8,  16, dims, K, D,  500, knn_indices_func).cuda()
    layer3 = RandPointCNN(  16,  32, dims, K, D,  250, knn_indices_func).cuda()
    layer4 = RandPointCNN(  32,  64, dims, K, D,  125, knn_indices_func).cuda()
    layer5 = RandPointCNN(  64, 128, dims, K, D,   50, knn_indices_func).cuda()

    pts  = np.random.rand(N,num_points,dims).astype(np.float32)
    fts  = np.random.rand(N,num_points,C_in).astype(np.float32)
    pts = Variable(torch.from_numpy(pts)).cuda()
    fts = Variable(torch.from_numpy(fts)).cuda()

    if True:
        pts, fts = layer1((pts, fts))
    else:
        plot(pts, fts)
        pts, fts = layer1((pts, fts))
        plot(pts, fts)
        pts, fts = layer2((pts, fts))
        plot(pts, fts)
        pts, fts = layer3((pts, fts))
        plot(pts, fts)
        pts, fts = layer4((pts, fts))
        plot(pts, fts)
        pts, fts = layer5((pts, fts))
        plot(pts, fts)
