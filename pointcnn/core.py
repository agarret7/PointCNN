import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

try:
    from .util import knn_indices_func, knn_indices_func_gpu
    from .layers import MLP, LayerNorm, Conv, SepConv, Dense, endchannels
    # from .context import timed
except SystemError:
    from util import knn_indices_func, knn_indices_func_gpu
    from layers import MLP, LayerNorm, Conv, SepConv, Dense, endchannels
    # from context import timed

class XConv(nn.Module):
    """
    Vectorized pointwise convolution.
    """

    def __init__(self, C_in, C_out, D, N_neighbors, N_rep, C_lifted, depth_multiplier):
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param D: Spatial dimensionality of points.
        :param N_neighbors: Number of neighbors to convolve over.
        :param C_lifted: Dimensionality of lifted point features.
        :param mlp_width: Number of hidden layers in MLPs.
        """
        super(XConv, self).__init__()

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
        self.dense1 = Dense(D, C_lifted)
        self.dense2 = Dense(C_lifted, C_lifted)

        # Layers to generate X
        self.x_trans = nn.Sequential(
            endchannels(Conv(
                in_channels = D,
                out_channels = N_neighbors**2,
                kernel_size = (1, N_neighbors),
                with_bn = False
            )),
            Dense(N_neighbors**2, N_neighbors**2, with_bn = False),
            Dense(N_neighbors**2, N_neighbors**2, with_bn = False, activation = None)
        )

        """
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
        """

        print(depth_multiplier)
        # Final 
        self.end_conv = endchannels(SepConv(
            in_channels = C_lifted + C_in,
            out_channels = C_out,
            kernel_size = (1, N_neighbors),
            depth_multiplier = depth_multiplier
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
        if F is not None:
            assert(p.size()[0] == P.size()[0] == F.size()[0])       # Check N is equal.
            assert(p.size()[1] == P.size()[1] == F.size()[1])       # Check N_rep is equal.
            assert(P.size()[2] == F.size()[2] == self.N_neighbors)  # Check N_neighbors is equal.
            assert(F.size()[3] == self.C_in)                        # Check C_in is equal.
        else:
            assert(p.size()[0] == P.size()[0])                      # Check N is equal.
            assert(p.size()[1] == P.size()[1])                      # Check N_rep is equal.
            assert(P.size()[2] == self.N_neighbors)                 # Check N_neighbors is equal.
        assert(p.size()[2] == P.size()[3] == self.D)                # Check D is equal.

        N = len(P)
        N_rep = p.size()[1]
        p_center = torch.unsqueeze(p, dim = 2)

        # Move P to local coordinate system of p.
        P_local = P - p_center
        # P_local = self.pts_layernorm(P - p_center)

        # Individually lift each point into C_lifted dim space.
        F_lifted0 = self.dense1(P_local)
        F_lifted  = self.dense2(F_lifted0)

        # Cat F_lifted and F,None to size (N, N_rep, N_neighbors, C_lifted + C_in).
        if F is None:
            F_cat = F_lifted
        else:
            F_cat = torch.cat((F_lifted, F), -1)

        # Learn the (N, K, K) X-transformation matrix.
        X_shape = (N, N_rep, self.N_neighbors, self.N_neighbors)
        X = self.x_trans(P_local)
        X = X.view(*X_shape)

        """
        X = self.mid_conv(P_local)
        X = X.contiguous().view(*X_shape)
        X = self.mid_dwconv1(X)
        X = X.contiguous().view(*X_shape)
        X = self.mid_dwconv2(X)
        X = X.contiguous().view(*X_shape)
        """

        # Weight and permute F_cat with the learned X.
        F_X = torch.matmul(X, F_cat)
        F_p = self.end_conv(F_X).squeeze(dim = 2)
        return F_p

class PointCNN(nn.Module):
    """
    TODO: Insert documentation
    """

    def __init__(self, C_in, C_out, D, N_neighbors, dilution, N_rep, r_indices_func):
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

        C_lifted = C_out // 2 if C_in == 0 else C_out // 4
        depth_multiplier = min(int(np.ceil(C_out / C_in)), 4)

        self.r_indices_func = r_indices_func
        self.dense = Dense(C_in, C_out // 2) if C_in != 0 else None
        self.x_conv = XConv(C_out // 2 if C_in != 0 else C_in, C_out, D, N_neighbors, N_rep, C_lifted, depth_multiplier)
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
        t0 = time.time()
        F = self.dense(F) if F is not None else F
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
        t1 = time.time()
        F_regional = self.select_region(F, P_idx) if F is not None else F
        test_time = time.time() - t1
        F_p = self.x_conv((ps, P_regional, F_regional))
        total_time = time.time() - t0
        # print("frac of time:", test_time / total_time)
        return F_p

class rPointCNN(nn.Module):
    """ PointCNN with randomly sampled representative points. """

    def __init__(self, *args, **kwargs):
        super(rPointCNN, self).__init__()
        self.pointcnn = PointCNN(*args, **kwargs)
        self.N_rep = args[5]  # Exists because PointCNN requires it.

    def forward(self, x):
        P, F = x
        if 0 < self.N_rep < P.size()[1]:
            idx = np.random.choice(P.size()[1], self.N_rep, replace = False).tolist()
            ps = P[:,idx,:]
        else:
            # All input points are representative points.
            ps = P
        ps_F = self.pointcnn((ps, P, F))
        return ps, ps_F

def plot(P, F):
    num_F = F.size()[2]
    pts = P[0].data.cpu().numpy()
    plt.scatter(pts[:,0], pts[:,1], s = num_F, c = "k")
    plt.savefig("./%i.png" % num_F)
    plt.cla()

if __name__ == "__main__":
    np.random.seed(0)

    N = 1
    num_points = 1000
    D = 2
    C_in = 4
    N_neighbors = 10
    dilution = 1

    layer1 = rPointCNN(C_in,   8, D, N_neighbors, dilution, 1000, knn_indices_func).cuda()
    layer2 = rPointCNN(   8,  16, D, N_neighbors, dilution,  500, knn_indices_func).cuda()
    layer3 = rPointCNN(  16,  32, D, N_neighbors, dilution,  250, knn_indices_func).cuda()
    layer4 = rPointCNN(  32,  64, D, N_neighbors, dilution,  125, knn_indices_func).cuda()
    layer5 = rPointCNN(  64, 128, D, N_neighbors, dilution,   50, knn_indices_func).cuda()

    P  = np.random.rand(N,num_points,D).astype(np.float32)
    F  = np.random.rand(N,num_points,C_in).astype(np.float32)
    P = Variable(torch.from_numpy(P)).cuda()
    F = Variable(torch.from_numpy(F)).cuda()

    if True:
        P, F = layer1((P, F))
    else:
        plot(P, F)
        P, F = layer1((P, F))
        plot(P, F)
        P, F = layer2((P, F))
        plot(P, F)
        P, F = layer3((P, F))
        plot(P, F)
        P, F = layer4((P, F))
        plot(P, F)
        P, F = layer5((P, F))
        plot(P, F)
