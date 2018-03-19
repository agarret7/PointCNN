import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn.neighbors import NearestNeighbors

from context import pytorch_knn_cuda
KNN = pytorch_knn_cuda.KNearestNeighbor

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
        self.x_conv = XConv(C_in, C_out, D, N_neighbors, C_lifted, mlp_width)

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

    def forward(self, ps, P, F):
        """
        Given a set of representative points, a point cloud, and its
        corresponding features, return a new set of representative points with
        features projected from the point cloud.
        :type ps: FloatTensor (N, *, D)
        :type P: FloatTensor (N, N_neighbors, D)
        :type F: FloatTensor (N, N_neighbors, C_in)
        :rtype:  FloatTensor (TODO: shape)
        :param ps: Representative points
        :param P: Regional point cloud such that F[:,p_idx,:] is the feature associated with P[:,p_idx,:]
        :param F: Regional features such that P[:,p_idx,:] is the feature associated with F[:,p_idx,:]
        :return:
        """
        P_idx = self.r_indices_func(ps, P, N_neighbors)
        inp_regions = torch.stack([
            self.x_conv(p, self.select_region(P, P_idx[:,n]), self.select_region(F, P_idx[:,n]))
            for n, p in enumerate(torch.unbind(ps, dim = 1))
        ], dim = 1)
        return inp_regions

def knn_indices_func(ps, P, k):
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
    ps = ps.data.numpy()
    P = P.data.numpy()

    def single_batch_knn(p, P_particular):
        nbrs = NearestNeighbors(k, algorithm = "ball_tree").fit(p)
        indices = nbrs.kneighbors(P_particular)[1]
        return indices

    region_idx = np.stack([
        single_batch_knn(p, P[n]) for n, p in enumerate(ps)
    ], axis = 0)
    return torch.from_numpy(region_idx)

if __name__ == "__main__":
    np.random.seed(0)

    N = 4
    num_points = 5000
    N_rep = 1000
    D = 3
    C_in = 8
    C_out = 32
    N_neighbors = 100

    model = PointCNN(C_in, C_out, D, N_neighbors, knn_indices_func)

    test_P  = np.random.rand(N,num_points,D).astype(np.float32)
    test_F  = np.random.rand(N,num_points,C_in).astype(np.float32)
    idx = np.random.choice(test_P.shape[1], N_rep, replace = False)
    test_ps = test_P[:,idx,:]

    test_P = Variable(torch.from_numpy(test_P))
    test_F = Variable(torch.from_numpy(test_F))
    test_ps = Variable(torch.from_numpy(test_ps))

    print(test_P)
    out = model(test_ps, test_P, test_F)
    print(out)
