import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def MLP(layer_sizes, activation_func = nn.ReLU()):
    """
    Creates a fully connected MLP of arbitrary depth.
    :param layer_sizes: Sizes of MLP hidden layers.
    :param activation_func: Activation function to be applied in between layers.
    :return: Multilayer perceptron module
    """
    if isinstance(layer_sizes, np.ndarray):
        layer_sizes = layer_sizes.tolist()
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(C_in, C_out), activation_func)
        for (C_in, C_out) in zip(layer_sizes, layer_sizes[1:])
    ])

class XConv(nn.Module):

    def __init__(self, C_in, C_out, D, N_neighbors, C_lifted = None, mlp_width = 4):
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param D: Spatial dimensionality of points.
        :param N_neighbors: Number of neighbors to convolve over.
        :param C_lifted: Dimensionality of lifted point features.
        :param mlp_width: Number of hidden layers in MLPs.
        """

        if C_lifted == None:
            C_lifted = C_in  # Not optimal?

        super(XConv, self).__init__()
        self.N_neighbors = N_neighbors
        self.D = D
        self.mlp_lift = MLP(np.around(np.geomspace(D, C_lifted, num = mlp_width)).astype(int))
        self.mlp = MLP(np.floor(np.geomspace(D, N_neighbors)).astype(int))

        self.K = nn.Parameter(torch.FloatTensor(C_out, C_in + C_lifted, N_neighbors))
        stdv = 1. / np.sqrt(N_neighbors)
        self.K.data.uniform_(-stdv, stdv)

    def forward(self, p, P, F):
        """
        Applies XConv to the input data.
        :type p: FloatTensor (N, D)
        :type P: FloatTensor (N, N_neighbors, D)
        :type F: FloatTensor (N, N_neighbors, C_in)
        :rtype:  FloatTensor (TODO: shape)
        :param p: Representative point
        :param P: Regional point cloud such that F[:,p_idx,:] is the feature associated with P[:,p_idx,:]
        :param F: Regional features such that P[:,p_idx,:] is the feature associated with F[:,p_idx,:]
        :return: Features aggregated into point p.
        """

        assert(p.size()[0] == P.size()[0] == F.size()[0])       # Check N is equal.
        assert(P.size()[1] == F.size()[1] == self.N_neighbors)  # Check N_neighbors is equal.
        assert(p.size()[1] == P.size()[2] == self.D)            # Check D is equal.

        N = len(P)
        P_loc = P - torch.unsqueeze(p, 1)               # Move P to local coordinate system of p.
        F_lifted = self.mlp_lift(P_loc)                 # Individually lift each point into C_lifted dim space.
        F_cat = torch.cat((F_lifted, F), 2)             # Cat F_lifted and F, to size (N, K, C_lifted + C_in).
        X = self.mlp(P_loc)                             # Learn the (N, K, K) X-transformation matrix.
        F_X = torch.stack([                             # Weight and permute F_cat with the learned X.
            torch.mm(X[n], F_cat[n]) for n in range(N)
        ], dim = 0)
        F_p = nn.functional.conv1d(                     # Finally, typical convolution between K and F_X.
            torch.transpose(F_X, 1, 2),
            self.K
        )
        return F_p.view(N, -1)

class PointCNN(nn.Module):

    def __init__(self, C_in, C_out, D, N_neighbors, r_indices_func, C_lifted = None, mlp_width = 4):
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param D: Spatial dimensionality of points.
        :param N_neighbors: Number of neighbors to convolve over.
        :param r_indices_func: Selector function of the type,
            INP 
            ======
            p : (N, D) Representative point
            P : (N, *, D) Point cloud

            OUT
            ======
            P_idx : (N, N_neighbors) Array of indices into P such that
            P[P_idx] is the set of points in the "region" around p.

        a representative point p and a point cloud P. From these it returns an
        array of N_neighbors 
        :param C_lifted: Dimensionality of lifted point features.
        :param mlp_width: Number of hidden layers in MLPs.
        """
        super(PointCNN, self).__init__()
        if C_lifted == None:
            C_lifted = C_in  # Not optimal?

        self.r_filter = r_filter
        self.x_conv = XConv(C_in, C_out, D, N_neighbors, C_lifted, mlp_width)

    def select_region(self, P_idx, P):
        """
        Selects 
        :type P_idx: FloatTensor (N, N_neighbors)
        :type P: FloatTensor (N, *, *)
        :param P_idx: Indices of points in region to be selected
        :param P: Point cloud to select regional points from
        """
        return torch.stack([
            P[n,:,:].index_select(0, idx) for n, idx in torch.unbind(P_idx, dim = 0)
        ], dim = 0)

    def forward(self, ps, P, F):
        """
        Given a set of representative points, a point cloud, and its
        corresponding features, return a new set of representative points with
        features projected from the point cloud.
        :type p: FloatTensor (N, *, D)
        :type P: FloatTensor (N, N_neighbors, D)
        :type F: FloatTensor (N, N_neighbors, C_in)
        :rtype:  FloatTensor (TODO: shape)
        :param p: Representative point
        :param P: Regional point cloud such that F[:,p_idx,:] is the feature associated with P[:,p_idx,:]
        :param F: Regional features such that P[:,p_idx,:] is the feature associated with F[:,p_idx,:]
        :return:
        """
        # (N, *, N_neighbors, D)
        P_idx = self.r_indices_func(p, P)
        inp_regions = torch.stack([
            self.x_conv(p, self.select_region(P, P_idx), self.select_region(F, P_idx))
            for p in torch.unbind(ps, dim = 1)
        ], dim = 1)
        return inp_regions

def knn_indices_func(p, P):
    """
    Indexing function based on K-Nearest Neighbors search.
    :type p: FloatTensor (N, D)
    :type P: FloatTensor (N, *, D)
    :rtype: FloatTensor (N, N_neighbors)
    :param p: Representative point
    :param P: Point cloud to get indices from
    :return: Array of indices, P_idx, into P such that P[P_idx] is the set
    of points in the "region" around p.
    """
    raise NotImplementedError("Implement me!")
