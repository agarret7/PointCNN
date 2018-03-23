import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

try:
    from .util import knn_indices_func
    from .context import timed
except SystemError:
    from util import knn_indices_func
    from context import timed

class BatchNorm(nn.Module):
    """
    PyTorch Linear layers transform shape in the form (N,*,in_features) ->
    (N,*,out_features). BatchNorm normalizes over axis 1. Thus, BatchNorm
    following a linear layer ONLY has the desired behavior if there are no
    additional (*) dimensions. To get the desired behavior, we first transpose
    the appropriate axis into the channel dim, then tranpsose out.
    """

    def __init__(self, D, num_features, dim = 1, *args, **kwargs):
        super(BatchNorm, self).__init__()
        if D == 1:
            self.bn = nn.BatchNorm1d(num_features, *args, **kwargs)
        elif D == 2:
            self.bn = nn.BatchNorm2d(num_features, *args, **kwargs)
        elif D == 3:
            self.bn = nn.BatchNorm3d(num_features, *args, **kwargs)
        else:
            raise ValueError("Dimensionality %i not supported" % D)

        self.dim = dim

    def forward(self, x):
        x = torch.transpose(x, 1, self.dim).contiguous()  # Must be made contiguous for cudNN.
        self.bn(x)
        x = torch.transpose(x, self.dim, 1)
        return x

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
                          BatchNorm(D = 2, num_features = C_out, dim = -1, momentum = 0.9)
            ) for (C_in, C_out) in zip(layer_sizes, layer_sizes[1:])
        ])
    else:
        return nn.Sequential(*[
            nn.Sequential(nn.Linear(C_in, C_out),
                          activation_layer,
            ) for (C_in, C_out) in zip(layer_sizes, layer_sizes[1:])
        ])

class XConv(nn.Module):
    """
    Vectorized pointwise convolution.
    """

    def __init__(self, C_in, C_out, D, N_neighbors, N_rep, C_lifted = None, mlp_width = 4):
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
        self.pts_batchnorm = nn.BatchNorm2d(N_rep, momentum = 0.9)

        # Main dense linear layers
        self.mlp_lift = MLP(np.around(np.geomspace(D, self.C_lifted, num = mlp_width)).astype(int))
        self.mlp = nn.Sequential(
            # torch.Conv2d(TODO),
            MLP(np.around(np.geomspace(3, N_neighbors)).astype(int))  # Somehow, original code has K x K.
        )

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
        P_local = self.pts_batchnorm(P - p_center)           # Move P to local coordinate system of p.
        F_lifted = self.mlp_lift(P_local)                    # Individually lift each point into C_lifted dim space.
        F_cat = torch.cat((F_lifted, F), -1)                 # Cat F_lifted and F, to size (N, N_rep, N_neighbors, C_lifted + C_in).
        X_shape = (N, self.N_rep, N_neighbors, N_neighbors)
        X = self.mlp(P_local).contiguous().view(*X_shape)    # Learn the (N, K, K) X-transformation matrix.
        F_X = torch.matmul(X, F_cat)                         # Weight and permute F_cat with the learned X.

        # CODE PAST THIS POINT BROKEN.
        # TODO: Implement separable_conv2d
        F_p = nn.functional.conv1d(            # Finally, typical convolution between K and F_X.
            torch.transpose(F_X, 1, 2),
            self.K
        )

        return torch.squeeze(F_p, dim = 2)

class PointCNN(nn.Module):

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
        :type P_idx: FloatTensor (N, N_neighbors)
        :rtype P_region: FloatTensor (N_rep, N_neighbors, D)
        :param P: Point cloud to select regional points from
        :param P_idx: Indices of points in region to be selected
        """
        regions = torch.stack([
            P[n][idx,:] for n, idx in enumerate(torch.unbind(P_idx, dim = 0))
        ], dim = 0)
        return regions

    @timed.timed
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
        P_idx = self.r_indices_func(ps.cpu(), P.cpu(), N_neighbors).cuda()
        inp_regions = torch.stack([
            self.x_conv(p, self.select_region(P, P_idx[:,n]), self.select_region(F, P_idx[:,n]))
            for n, p in enumerate(torch.unbind(ps, dim = 1))
        ], dim = 1)
        return inp_regions

if __name__ == "__main__":
    np.random.seed(0)

    TESTING = XConv

    if TESTING == XConv:
        N = 4
        N_rep = 100
        D = 3
        C_in = 8
        C_out = 32
        N_neighbors = 10

        model = XConv(C_in, C_out, D, N_neighbors, N_rep).cuda()
        p = Variable(torch.from_numpy(np.random.rand(N,N_rep,D).astype(np.float32))).cuda()
        P = Variable(torch.from_numpy(np.random.rand(N,N_rep,N_neighbors,D).astype(np.float32))).cuda()
        F = Variable(torch.from_numpy(np.random.rand(N,N_rep,N_neighbors,C_in).astype(np.float32))).cuda()
        out = model(p, P, F)

    elif TESTING == PointCNN:
        N = 4
        num_points = 15000
        N_rep = 7500
        D = 3
        C_in = 8
        C_out = 32
        N_neighbors = 5

        model = PointCNN(C_in, C_out, D, N_neighbors, N_rep, knn_indices_func).cuda()

        test_P  = np.random.rand(N,num_points,D).astype(np.float32)
        test_F  = np.random.rand(N,num_points,C_in).astype(np.float32)
        idx = np.random.choice(test_P.shape[1], N_rep, replace = False)
        test_ps = test_P[:,idx,:]

        test_P = Variable(torch.from_numpy(test_P)).cuda()
        test_F = Variable(torch.from_numpy(test_F)).cuda()
        test_ps = Variable(torch.from_numpy(test_ps)).cuda()

        out = model(test_ps, test_P, test_F)
