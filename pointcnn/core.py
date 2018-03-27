import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

try:
    from .util import knn_indices_func, endchannels
    from .layers import MLP, BatchNorm, SeparableConv2d, DepthwiseConv2d
    from .context import timed
except SystemError:
    from util import knn_indices_func, endchannels
    from layers import MLP, BatchNorm, SeparableConv2d, DepthwiseConv2d
    from context import timed

class XConv(nn.Module):
    """
    Vectorized pointwise convolution.
    """

    def __init__(self, C_in, C_out, D, N_neighbors, N_rep, C_lifted = None, mlp_width = 2):
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
        self.pts_batchnorm = BatchNorm(2, D, momentum = 0.9)
        # self.pts_batchnorm = BatchNorm(BatchNorm())

        # Main dense linear layers
        self.mlp_lift = MLP([D] + [self.C_lifted] * mlp_width)
        self.mid_conv = endchannels(nn.Conv2d(D, N_neighbors, 1).cuda())
        self.mlp = MLP([N_neighbors] * mlp_width)  # Somehow, original code has K x K.
        self.end_conv = endchannels(nn.Conv2d(C_lifted + C_in, C_out, (N_neighbors, 1), groups = C_out).cuda())

        # Params for kernel initialization.
        self.K = nn.Parameter(torch.FloatTensor(C_out, C_in + self.C_lifted, N_neighbors))
        stdv = 1. / np.sqrt(N_neighbors)
        self.K.data.uniform_(-stdv, stdv)

    def forward(self, p, P, F):
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
        assert(p.size()[0] == P.size()[0] == F.size()[0])                # Check N is equal.
        assert(p.size()[1] == P.size()[1] == F.size()[1] == self.N_rep)  # Check N_rep is equal.
        assert(P.size()[2] == F.size()[2] == self.N_neighbors)           # Check N_neighbors is equal.
        assert(p.size()[2] == P.size()[3] == self.D)                     # Check D is equal.
        assert(F.size()[3] == self.C_in)                                 # Check C_in is equal.

        N = len(P)
        p_center = torch.unsqueeze(p, dim = 2)
        P_local = self.pts_batchnorm(P - p_center)  # Move P to local coordinate system of p.
        F_lifted = self.mlp_lift(P_local)           # Individually lift each point into C_lifted dim space.
        F_cat = torch.cat((F_lifted, F), -1)        # Cat F_lifted and F, to size (N, N_rep, N_neighbors, C_lifted + C_in).
        X_shape = (N, self.N_rep, N_neighbors, N_neighbors)
        X = self.mlp(self.mid_conv(P_local))        # Learn the (N, K, K) X-transformation matrix.
        X = X.contiguous().view(*X_shape)
        F_X = torch.matmul(X, F_cat)                # Weight and permute F_cat with the learned X.
        F_p = self.end_conv(F_X)
        return torch.squeeze(F_p, dim = 2)

class PointCNN(nn.Module):
    """
    TODO: Insert documentation
    """

    def __init__(self, C_in, C_out, D, N_neighbors, N_rep, r_indices_func, C_lifted = None, mlp_width = 4):
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param D: Spatial dimensionality of points.
        :param N_neighbors: Number of neighbors to convolve over.
        :param r_indices_func: Selector function of the type,
            INP
            ======
            ps : (N, N_rep, D) Representative points
            P  : (N, *, D) Point cloud
            N_neighbors : Number of points for each region.

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

    # @timed.timed
    def forward(self, ps, P, F):
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
        P_idx = self.r_indices_func(ps.cpu(), P.cpu(), N_neighbors).cuda()  # This step takes ~97% of the time.
        P_regional = self.select_region(P, P_idx)                           # Prime target for optimization: KNN on GPU.
        if False:
            # Draw neighborhood points, for debugging.
            t = 23
            n = 3
            test_point = ps[n,t,:].cpu().data.numpy()
            neighborhood = P_regional[n,t,:,:].cpu().data.numpy()
            plt.scatter(P[n][:,0], P[n][:,1])
            plt.scatter(test_point[0], test_point[1], s = 100, c = 'green')
            plt.scatter(neighborhood[:,0], neighborhood[:,1], s = 100, c = 'red')
            plt.show()
        F_regional = self.select_region(F, P_idx)
        # ps, P, F_P -> ps_F
        return self.x_conv(ps, P_regional, F_regional)

if __name__ == "__main__":
    np.random.seed(0)

    TESTING = PointCNN

    if TESTING == XConv:
        N = 4
        N_rep = 150
        D = 3
        C_in = 8
        C_out = 32
        N_neighbors = 10

        model = XConv(C_in, C_out, D, N_neighbors, N_rep)
        p = Variable(torch.from_numpy(np.random.rand(N,N_rep,D).astype(np.float32)))
        P = Variable(torch.from_numpy(np.random.rand(N,N_rep,N_neighbors,D).astype(np.float32)))
        F = Variable(torch.from_numpy(np.random.rand(N,N_rep,N_neighbors,C_in).astype(np.float32)))

        out = model(p, P, F)

    elif TESTING == PointCNN:
        N = 4
        num_points = 1000
        N_rep = 50
        D = 2
        C_in = 128
        C_out = 256
        N_neighbors = 5

        model = PointCNN(C_in, C_out, D, N_neighbors, N_rep, knn_indices_func).cuda()

        test_P  = np.random.rand(N,num_points,D).astype(np.float32)
        test_F  = np.random.rand(N,num_points,C_in).astype(np.float32)
        idx = np.random.choice(test_P.shape[1], N_rep, replace = False)
        test_ps = test_P[:,idx,:]

        test_P = Variable(torch.from_numpy(test_P)).cuda()
        test_F = Variable(torch.from_numpy(test_F)).cuda()
        test_ps = Variable(torch.from_numpy(test_ps)).cuda()

        print(test_F.size())
        out = model(test_ps, test_P, test_F)
        print(out.size())
