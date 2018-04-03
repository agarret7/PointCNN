import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

try:
    from .util import knn_indices_func, knn_indices_func_gpu
    from .layers import MLP, LayerNorm, Conv, SepConv, endchannels
    # from .context import timed
except SystemError:
    from util import knn_indices_func, knn_indices_func_gpu
    from layers import MLP, LayerNorm, Conv, SepConv, endchannels
    # from context import timed

class XConv(nn.Module):
    """
    Vectorized pointwise convolution.
    """

    def __init__(self, C_in, C_out, D, N_neighbors, N_rep, C_lifted = None,
                 mlp_width = 2):
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param D: Spatial dimensionality of points.
        :param N_neighbors: Number of neighbors to convolve over.
        :param C_lifted: Dimensionality of lifted point features.
        :param mlp_width: Number of hidden layers in MLPs.
        """
        super(XConv, self).__init__()

        if C_lifted == None:
            C_lifted = C_in  # Not optimal?

        if __debug__:
            # Only needed for assertions.
            self.C_in = C_in
            self.C_lifted = C_lifted
            self.D = D
            self.N_neighbors = N_neighbors

        self.N_rep = N_rep

        # Additional processing layers
        # self.pts_layernorm = LayerNorm(2, momentum = 0.9)

        # Main dense linear layers
        self.mlp_lift = MLP([D] + [self.C_lifted] * (mlp_width - 1), batch_norm = False)

        # Layers to generate X
        self.mid_conv = endchannels(Conv(D, N_neighbors**2, (1, N_neighbors))).cuda()
        self.mid_dwconv1 = endchannels(SepConv(
            in_channels = N_neighbors,
            out_channels = N_neighbors**2,
            kernel_size = (1, N_neighbors),
            depth_multiplier = N_neighbors
        )).cuda()
        self.mid_dwconv2 = endchannels(SepConv(
            in_channels = N_neighbors,
            out_channels = N_neighbors**2,
            kernel_size = (1, N_neighbors),
            depth_multiplier = N_neighbors
        )).cuda()

        # Final 
        self.mlp = MLP([N_neighbors] * mlp_width, batch_norm = False)
        self.end_conv = endchannels(SepConv(
            in_channels = C_lifted + C_in,
            out_channels = C_out,
            kernel_size = (1, N_neighbors),
            depth_multiplier = 4
        )).cuda()

    # @timed.timed
    def forward(self, x):
        """
        Applies XConv to the input data.
        :type p: FloatTensor (N, N_rep, D)
        :type P: FloatTensor (N, N_rep, N_neighbors, D)
        :type F: FloatTensor (N, N_rep, N_neighbors, C_in)
        :rtype:  FloatTensor (TODO: shape)
        :param p: Representative point
        :param P: Regional point cloud such that F[:,p_idx,:] is the feature associated with P[:,p_idx,:]
        :param F: Regional features such that P[:,p_idx,:] is the feature associated with F[:,p_idx,:]
        :return: Features aggregated into point p.
        """
        p, P, F = x
        assert(p.size()[0] == P.size()[0] == F.size()[0])       # Check N is equal.
        assert(p.size()[1] == P.size()[1] == F.size()[1])       # Check N_rep is equal.
        assert(P.size()[2] == F.size()[2] == self.N_neighbors)  # Check N_neighbors is equal.
        assert(p.size()[2] == P.size()[3] == self.D)            # Check D is equal.
        assert(F.size()[3] == self.C_in)                        # Check C_in is equal.

        N = len(P)
        N_rep = p.size()[1]
        p_center = torch.unsqueeze(p, dim = 2)

        # Move P to local coordinate system of p.
        P_local = P - p_center
        # P_local = self.pts_layernorm(P - p_center)

        # Individually lift each point into C_lifted dim space.
        F_lifted = self.mlp_lift(P_local)

        # Cat F_lifted and F, to size (N, N_rep, N_neighbors, C_lifted + C_in).
        F_cat = torch.cat((F_lifted, F), -1)

        # Learn the (N, K, K) X-transformation matrix.
        X_shape = (N, N_rep, self.N_neighbors, self.N_neighbors)
        X = self.mid_conv(P_local)
        X = X.contiguous().view(*X_shape)
        X = self.mid_dwconv1(X)
        X = X.contiguous().view(*X_shape)
        X = self.mid_dwconv2(X)
        X = X.contiguous().view(*X_shape)

        # Weight and permute F_cat with the learned X.
        F_X = torch.matmul(X, F_cat)
        F_p = self.end_conv(F_X).squeeze(dim = 2)
        return F_p

class PointCNN(nn.Module):
    """
    TODO: Insert documentation
    """

    def __init__(self, C_in, C_out, D, N_neighbors, dilution, N_rep,
                 r_indices_func, C_lifted = None, mlp_width = 2):
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param D: Spatial dimensionality of points.
        :param N_neighbors: Number of neighbors to convolve over.
        :param N_rep: Number of representative points.
        :param dilution: "Spread" of neighboring points.
        :param r_indices_func: Selector function of the type,
            INP
            ======
            ps : (N, N_rep, D) Representative points
            P  : (N, *, D) Point cloud
            N_neighbors : Number of points for each region.
            dilution : "Spread" of neighboring points (analogous to stride).

            OUT
            ======
            P_idx : (N, N_rep, N_neighbors) Array of indices into P such that
            P[P_idx] is the set of points in the "region" around p.

        a representative point p and a point cloud P. From these it returns an
        array of N_neighbors
        :param C_lifted: Dimensionality of lifted point features.
        :param mlp_width: Number of hidden layers in MLPs.
        """
        super(PointCNN, self).__init__()

        if C_lifted == None:
            C_lifted = C_in  # Not optimal?

        self.r_indices_func = r_indices_func
        self.x_conv = XConv(C_in, C_out, D, N_neighbors, N_rep, C_lifted, mlp_width)
        self.dilution = dilution

    def select_region(self, P, P_idx):
        """
        Selects
        :type P: FloatTensor (N, *, D)
        :type P_idx: FloatTensor (N, N_rep, N_neighbors)
        :rtype P_region: FloatTensor (N_rep, N_neighbors, D)
        :param P: Point cloud to select regional points from
        :param P_idx: Indices of points in region to be selected
        :return:
        """
        regions = torch.stack([
            P[n][idx,:] for n, idx in enumerate(torch.unbind(P_idx, dim = 0))
        ], dim = 0)
        return regions

    def forward(self, x):
        """
        Given a set of representative points, a point cloud, and its
        corresponding features, return a new set of representative points with
        features projected from the point cloud.
        :type ps: FloatTensor (N, *, D)
        :type P: FloatTensor (N, N_neighbors, D)
        :type F: FloatTensor (N, N_neighbors, C_in)
        :rtype:  FloatTensor (N, N_rep, D)
        :param ps: Representative points
        :param P: Regional point cloud such that F[:,p_idx,:] is the feature associated with P[:,p_idx,:]
        :param F: Regional features such that P[:,p_idx,:] is the feature associated with F[:,p_idx,:]
        :return:
        """
        ps, P, F = x
        P_idx = self.r_indices_func(ps, P, self.x_conv.N_neighbors, self.dilution)  # This step takes ~97% of the time.
        P_regional = self.select_region(P, P_idx)  # Prime target for optimization: KNN on GPU.
        if False:
            # Draw neighborhood points, for debugging.
            t = 10
            n = 0
            test_point = ps[n,t,:].cpu().data.numpy()
            neighborhood = P_regional[n,t,:,:].cpu().data.numpy()
            plt.scatter(P[n][:,0], P[n][:,1])
            plt.scatter(test_point[0], test_point[1], s = 100, c = 'green')
            plt.scatter(neighborhood[:,0], neighborhood[:,1], s = 100, c = 'red')
            plt.show()
        F_regional = self.select_region(F, P_idx)
        # ps, P, F_P -> ps_F
        F_p = self.x_conv((ps, P_regional, F_regional))
        return F_p

class rPointCNN(nn.Module):
    """ PointCNN with randomly sampled representative points. """

    def __init__(self, *args, **kwargs):
        super(rPointCNN, self).__init__()
        self.pointcnn = PointCNN(*args, **kwargs)
        self.N_rep = args[5]  # Exists because PointCNN requires it.

    def forward(self, x):
        P, F = x
        if self.N_rep < P.size()[1]:
            idx = np.random.choice(P.size()[1], self.N_rep, replace = False).tolist()
            ps = P[:,idx,:]
        else:
            # All input points are representative points.
            ps = P
        ps_F = self.pointcnn((ps, P, F))
        return ps, ps_F

if __name__ == "__main__":
    np.random.seed(0)

    N = 1
    num_points = 500
    N_rep = 20
    D = 2
    C_in = 16
    C_out = 32
    N_neighbors = 30
    dilution = 1

    model = PointCNN(C_in, C_out, D, N_neighbors, dilution, N_rep, knn_indices_func_gpu).cuda()

    test_P  = np.random.rand(N,num_points,D).astype(np.float32)
    test_F  = np.random.rand(N,num_points,C_in).astype(np.float32)
    idx = np.random.choice(test_P.shape[1], N_rep, replace = False)
    test_ps = test_P[:,idx,:]

    test_P = Variable(torch.from_numpy(test_P)).cuda()
    test_F = Variable(torch.from_numpy(test_F)).cuda()
    test_ps = Variable(torch.from_numpy(test_ps)).cuda()

    print(test_F.size())
    for _ in range(1):
        out = model((test_ps, test_P, test_F))
    print(out.size())
