"""
Author: Austin J. Garrett

PyTorch implementation of the PointCNN paper, as specified in:
  https://arxiv.org/pdf/1801.07791.pdf
Original paper by: Yangyan Li, Rui Bu, Mingchao Sun, Baoquan Chen
"""

# Standard Modules
from itertools import product, groupby
import time

# External Modules
from torch import FloatTensor
from typing import Tuple, Callable, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Internal Modules
from PointCNN.core.util_funcs import UFloatTensor, ULongTensor
from PointCNN.core.util_layers import Conv, SepConv, Dense, EndChannels

# plt.ion()

class XConv(nn.Module):
    """ Convolution over a single point and its neighbors.  """

    def __init__(self, C_in : int, C_out : int, dims : int, K : int,
                 P : int, C_mid : int, depth_multiplier : int) -> None:
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param P: Number of representative points.
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
            EndChannels(Conv(
                in_channels = dims,
                out_channels = K*K,
                kernel_size = (1, K),
                with_bn = False
            )),
            Dense(K*K, K*K, with_bn = False),
            Dense(K*K, K*K, with_bn = False, activation = None)
        )

        self.end_conv = EndChannels(SepConv(
            in_channels = C_mid + C_in,
            out_channels = C_out,
            kernel_size = (1, K),
            depth_multiplier = depth_multiplier
        )).cuda()

    def forward(self, x : Tuple[UFloatTensor,            # (N, P, dims)
                                UFloatTensor,            # (N, P, K, dims)
                                Optional[UFloatTensor]]  # (N, P, K, C_in)
               ) -> UFloatTensor:                        # (N, K, C_out)
        """
        Applies XConv to the input data.
        :param x: (rep_pt, pts, fts) where
          - rep_pt: Representative point.
          - pts: Regional point cloud such that fts[:,p_idx,:] is the feature
          associated with pts[:,p_idx,:].
          - fts: Regional features such that pts[:,p_idx,:] is the feature
          associated with fts[:,p_idx,:].
        :return: Features aggregated into point rep_pt.
        """
        rep_pt, pts, fts = x

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

        # Needed for precached results
        # t0 = time.time()
        # rand_idx = torch.randperm(pts.size()[1])
        # rep_pt = rep_pt[:,rand_idx,:]
        # pts = pts[:,rand_idx,:]
        # fts = fts[:,rand_idx,:]
        # print("perm", time.time() - t0)

        N = len(pts)
        P = rep_pt.size()[1]  # (N, P, K, dims)
        p_center = torch.unsqueeze(rep_pt, dim = 2)  # (N, P, 1, dims)

        # Move pts to local coordinate system of rep_pt.
        # t0 = time.time()
        pts_local = pts - p_center  # (N, P, K, dims)
        # print("localizing", time.time() - t0)
        # pts_local = self.pts_layernorm(pts - p_center)

        # Individually lift each point into C_mid space.
        # t0 = time.time()
        fts_lifted0 = self.dense1(pts_local)
        fts_lifted  = self.dense2(fts_lifted0)  # (N, P, K, C_mid)
        # print("lifting", time.time() - t0)

        # t0 = time.time()
        if fts is None:
            fts_cat = fts_lifted
        else:
            fts_cat = torch.cat((fts_lifted, fts), -1)  # (N, P, K, C_mid + C_in)
        # print("cat", time.time() - t0)

        # Learn the (N, K, K) X-transformation matrix.
        # t0 = time.time()
        X_shape = (N, P, self.K, self.K)
        X = self.x_trans(pts_local)
        X = X.view(*X_shape)
        # print("X-CONV", time.time() - t0)


        # Weight and permute fts_cat with the learned X.
        # t0 = time.time()
        fts_X = torch.matmul(X, fts_cat)
        # print("matmul", time.time() - t0)
        # t0 = time.time()
        fts_p = self.end_conv(fts_X).squeeze(dim = 2)
        # print("end-conv", time.time() - t0)

        # Needed for precached results
        # t0 = time.time()
        # rand_idx_inv = torch.LongTensor(len(rand_idx))
        # rand_idx_inv[rand_idx] = torch.LongTensor(range(len(rand_idx)))
        # print("inverse", time.time() - t0)

        # print("START LAYER")
        # print(rep_pt.size())
        # print(pts.size())

        # plt.scatter(rep_pt.cpu().numpy()[0,:,0], rep_pt.cpu().numpy()[0,:,1], c = 'r', s = 1)

        # for n in np.random.randint(0,rep_pt.size()[1],(100)):
        #     p = rep_pt.cpu().numpy()[0,n]

        #     # for n,p in enumerate(rep_pt[0].cpu().numpy()):
        #     nbhd = pts[0,n,:,:2].cpu().numpy()
        #     for nb in nbhd:
        #         plt.plot([p[0],nb[0]], [p[1],nb[1]], c = 'k', lw = 0.5)

        # plt.show()
        # input("PAUSE")
        # plt.cla()

        return fts_p # [:,rand_idx_inv,:]

class PointCNN(nn.Module):
    """ Pointwise convolutional model. """

    def __init__(self, C_in : int, C_out : int, dims : int, K : int, D : int, P : int,
                 r_indices_func : Callable[[UFloatTensor,  # (N, P, dims)
                                            UFloatTensor,  # (N, x, dims)
                                            int, int],
                                           ULongTensor],    # (N, P, K)
                 sampling_method : str = "fast_fps") -> None:
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param D: "Spread" of neighboring points.
        :param P: Number of representative points.
        :param r_indices_func: Selector function of the type,
          INPUTS
          rep_pts : Representative points.
          pts  : Point cloud.
          K : Number of points for each region.
          D : "Spread" of neighboring points.

          OUTPUT
          pts_idx : Array of indices into pts such that pts[pts_idx] is the set
          of points in the "region" around rep_pt.
        """
        super(PointCNN, self).__init__()

        C_mid = C_out // 2 if C_in == 0 else C_out // 4
        depth_multiplier = min(int(np.ceil(C_out / C_in)), 4)

        self.r_indices_func = lambda rep_pts, pts: r_indices_func(rep_pts, pts, K, D)
        self.dense = Dense(C_in, C_out // 2) if C_in != 0 else None
        self.x_conv = XConv(C_out // 2 if C_in != 0 else C_in, C_out, dims, K, P, C_mid, depth_multiplier)
        self.K = K
        self.D = D
        self.P = P
        self.sampling_method = sampling_method

    def select_region(self, pts : UFloatTensor,  # (N, x, dims)
                      pts_idx : ULongTensor      # (N, P, K)
                     ) -> UFloatTensor:          # (P, K, dims)
        """
        Selects neighborhood points based on output of r_indices_func.
        :param pts: Point cloud to select regional points from.
        :param pts_idx: Indices of points in region to be selected.
        :return: Local neighborhoods around each representative point.
        """
        regions = torch.stack([
            pts[n][idx,:] for n, idx in enumerate(torch.unbind(pts_idx, dim = 0))
        ], dim = 0)
        return regions

    def forward(self,
                x : Tuple[FloatTensor,        # (N, P, dims)
                          FloatTensor,        # (N, x, dims)
                          FloatTensor],       # (N, x, C_in)
                pts_idx : FloatTensor = None  # TODO
               ) -> FloatTensor:              # (N, P, C_out)
        """
        Given a set of representative points, a point cloud, and its
        corresponding features, return a new set of representative points with
        features projected from the point cloud.
        :param x: (rep_pts, pts, fts) where
          - rep_pts: Representative points.
          - pts: Regional point cloud such that fts[:,p_idx,:] is the
          feature associated with pts[:,p_idx,:].
          - fts: Regional features such that pts[:,p_idx,:] is the feature
          associated with fts[:,p_idx,:].
        :return: Features aggregated to rep_pts.
        """
        if len(x) == 2:
            pts, fts = x

            if 0 < self.P < pts.size()[1]:
                # Select random set of indices of subsampled points.
                if self.sampling_method == "rand":
                    idx = np.random.choice(pts.size()[1], self.P, replace = False).tolist()
                    rep_pts = pts[:,idx,:]
                elif self.sampling_method == "fps":
                    # t0 = time.time()
                    idx = self.batch_fps(pts, self.P)
                    rep_pts = torch.stack([pts[n][i,:] for n,i in enumerate(idx)])
                    # print("BATCH FPS:", time.time() - t0)
                elif self.sampling_method == "fast_fps":
                    # t0 = time.time()
                    idx = self.fast_fps(pts, self.P)
                    # print("FPS", time.time() - t0)
                    rep_pts = torch.stack([pts[n][i,:] for n,i in enumerate(idx)])
                else:
                    raise ValueError("Unrecognized sampling method %s" % self.sampling_method)
            else:
                # All input points are representative points.
                rep_pts = pts
        else:
            rep_pts, pts, fts = x

        fts = self.dense(fts) if fts is not None else fts

        # This step takes ~97% of the time. Prime target for optimization: KNN on GPU.
        if type(pts_idx) == type(None):
            # t0 = time.time()
            pts_idx = self.r_indices_func(rep_pts, pts)
            # print("KNN:", time.time() - t0)
            # print(pts_idx.size())
        else:
            pts_idx = pts_idx[:,:,:self.K*self.D:self.D].cuda()
        # -------------------------------------------------------------------------- #

        # plt.scatter(pts[0,:,0], pts[0,:,1], c = 'k', s = 0.1)
        # plt.scatter(rep_pts[0,:,0], rep_pts[0,:,1], c = 'r', s = 1)
        # plt.show()
        # input("PAUSE")
        # plt.cla()

        # t0 = time.time()
        pts_regional = self.select_region(pts, pts_idx)
        # print("SELECT REGION PTS:", time.time() - t0)

        # print("START LAYER")
        # print(rep_pts.size())
        # print(pts_regional.size())

        # plt.scatter(rep_pts.cpu().numpy()[0,:,0], rep_pts.cpu().numpy()[0,:,1], c = 'r', s = 1)

        # for n in np.random.randint(0,rep_pts.size()[1],(100)):
        #     p = rep_pts.cpu().numpy()[0,n]

        #     # for n,p in enumerate(rep_pts[0].cpu().numpy()):
        #     nbhd = pts_regional[0,n,:,:2].cpu().numpy()
        #     for nb in nbhd:
        #         plt.plot([p[0],nb[0]], [p[1],nb[1]], c = 'k', lw = 0.5)

        # plt.show()
        # input("PAUSE")
        # plt.cla()

        fts_regional = self.select_region(fts, pts_idx) if fts is not None else fts
        # t1 = time.time()
        fts_p = self.x_conv((rep_pts, pts_regional, fts_regional))
        # t1 = time.time() - t1
        # print("X-CONV:", t1)

        return (rep_pts, fts_p) if len(x) == 2 else fts_p

    def batch_fps(self, batch_pts, K):
        """ Found here: 
        https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
        """

        if isinstance(batch_pts, torch.autograd.Variable):
            batch_pts = batch_pts.data
        if batch_pts.is_cuda:
            batch_pts = batch_pts.cpu()

        calc_distances = lambda p0, pts: ((p0 - pts)**2).sum(dim = 1)

        def fps(x):
            pts, K = x
            D = pts.size()[1]
            farthest_idx = torch.IntTensor(K)
            farthest_idx.zero_()
            farthest_idx[0] = np.random.randint(len(pts))
            distances = calc_distances(pts[farthest_idx[0]], pts)

            for i in range(1, K):
                farthest_idx[i] = torch.max(distances, dim = 0)[1]
                farthest_pts = pts[farthest_idx[i]]
                distances = torch.min(distances, calc_distances(farthest_pts, pts))

            return farthest_idx

        batch_pts = list(map(fps, [(pts,K) for pts in batch_pts]))
        return torch.stack(batch_pts, dim = 0).long().cuda()

    def fast_fps(self, batch_pts, K):

        cell_size = torch.FloatTensor([1.2, 1.2, 1.2]).cuda()

        def fps(x):
            pts, K = x
            N = len(pts)

            lower = torch.min(pts, dim = 0)[0]
            upper = torch.max(pts, dim = 0)[0]
            dims = upper - lower
            idx_collapse = (dims / cell_size).int() + 1
            idx_collapse[0] = idx_collapse[1] * idx_collapse[2]
            idx_collapse[1] = idx_collapse[2]
            idx_collapse[2] = 1

            bin_idx = ((pts - upper.cuda()) / cell_size).int()
            bin_idx *= idx_collapse
            bin_idx = torch.sum(bin_idx, dim = 1)
            sorted_bins, p_idx = torch.sort(bin_idx, dim = 0)

            densities = [len(list(group)) for key, group in groupby(sorted_bins.tolist())]

            bin_prob = 1.0 / len(densities)
            p_probs = []

            for d in densities:
                single_bins = [bin_prob / d] * d
                p_probs += single_bins

            return torch.from_numpy(np.random.choice(p_idx, size = K, replace = False, p = p_probs))

        batch_pts = list(map(fps, [(pts,K) for pts in batch_pts]))
        return torch.stack(batch_pts, dim = 0).long().cuda()

class RandPointCNN(nn.Module):
    """ PointCNN with randomly subsampled representative points. """

    def __init__(self, C_in : int, C_out : int, dims : int, K : int, D : int, P : int,
                 r_indices_func : Callable[[UFloatTensor,  # (N, P, dims)
                                            UFloatTensor,  # (N, x, dims)
                                            int, int],
                                           ULongTensor],   # (N, P, K)
                 sampling_method : str = "rand") -> None:
        """ See documentation for PointCNN. """
        super(RandPointCNN, self).__init__()
        self.pointcnn = PointCNN(C_in, C_out, dims, K, D, P, r_indices_func)
        self.P = P
        self.sampling_method = sampling_method

    def forward(self, x : Tuple[UFloatTensor,  # (N, x, dims)
                                UFloatTensor]  # (N, x, dims)
               ) -> Tuple[UFloatTensor,        # (N, P, dims)
                          UFloatTensor]:       # (N, P, C_out)
        """
        Given a point cloud, and its corresponding features, return a new set
        of randomly-sampled representative points with features projected from
        the point cloud.
        :param x: (pts, fts) where
         - pts: Regional point cloud such that fts[:,p_idx,:] is the
        feature associated with pts[:,p_idx,:].
         - fts: Regional features such that pts[:,p_idx,:] is the feature
        associated with fts[:,p_idx,:].
        :return: Randomly subsampled points and their features.
        """
        pts, fts = x
        if 0 < self.P < pts.size()[1]:
            # Select random set of indices of subsampled points.
            if self.sampling_method == "rand":
                idx = np.random.choice(pts.size()[1], self.P, replace = False).tolist()
                rep_pts = pts[:,idx,:]
            elif self.sampling_method == "fps":
                # t0 = time.time()
                idx = self.batch_fps(pts, self.P)
                rep_pts = torch.stack([pts[n][i,:] for n,i in enumerate(idx)])
                # print("BATCH FPS:", time.time() - t0)
            else:
                raise ValueError("Unrecognized sampling method %s" % self.sampling_method)
        else:
            # All input points are representative points.
            rep_pts = pts
        # t0 = time.time()
        rep_pts_fts = self.pointcnn((rep_pts, pts, fts))
        # print("TOTAL:", time.time() - t0)
        return rep_pts, rep_pts_fts

    def batch_fps(self, batch_pts, K):
        """ Found here: 
        https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
        """

        if isinstance(batch_pts, torch.autograd.Variable):
            batch_pts = batch_pts.data
        if batch_pts.is_cuda:
            batch_pts = batch_pts.cpu()

        calc_distances = lambda p0, pts: ((p0 - pts)**2).sum(dim = 1)

        def fps(x):
            pts, K = x
            D = pts.size()[1]
            farthest_idx = torch.IntTensor(K)
            farthest_idx.zero_()
            farthest_idx[0] = np.random.randint(len(pts))
            distances = calc_distances(pts[farthest_idx[0]], pts)

            for i in range(1, K):
                farthest_idx[i] = torch.max(distances, dim = 0)[1]
                farthest_pts = pts[farthest_idx[i]]
                distances = torch.min(distances, calc_distances(farthest_pts, pts))

            return farthest_idx

        batch_pts = list(map(fps, [(pts,K) for pts in batch_pts]))
        return torch.stack(batch_pts, dim = 0).long().cuda()
