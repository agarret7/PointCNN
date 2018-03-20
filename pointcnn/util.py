import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

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
        # p, P_particular = P_particular, p
        nbrs = NearestNeighbors(k + 1, algorithm = "ball_tree").fit(P_particular)
        indices = nbrs.kneighbors(p)[1]
        return indices[:,1:]

    region_idx = np.stack([
        single_batch_knn(p, P[n]) for n, p in enumerate(ps)
    ], axis = 0)
    return torch.from_numpy(region_idx)

if __name__ == "__main__":
    from torch.autograd import Variable

