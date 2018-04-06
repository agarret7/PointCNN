import sys, os

import torch

import numpy as np
from sklearn.neighbors import NearestNeighbors

torch.CUDA_LAUNCH_BLOCKING = 1

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 

sys.path.append(os.path.join(CURRENT_DIR, "..", "lib"))

import pytorch_knn_cuda

try:
    # from .context import pytorch_knn_cuda
    pass
except SystemError:
    # from context import pytorch_knn_cuda
    pass

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

def knn_indices_func(ps, P, k, d):
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
        nbrs = NearestNeighbors(d*k + 1, algorithm = "ball_tree").fit(P_particular)
        indices = nbrs.kneighbors(p)[1]
        return indices[:,1::d]

    region_idx = np.stack([
        single_batch_knn(p, P[n]) for n, p in enumerate(ps)
    ], axis = 0)
    return torch.from_numpy(region_idx)

# def knn_indices_func_gpu(ps, P, k, d):
# 
#     def single_batch_knn(p, P_particular):
#         nbrs_f = pytorch_knn_cuda.KNearestNeighbor(d*k + 1)
#         indices = nbrs_f(P_particular, p)[0]
#         return indices[:,1::d]
# 
#     region_idx = torch.stack([
#         single_batch_knn(p, P[n]) for n, p in enumerate(ps)
#     ], dim = 0)
#     return region_idx

def knn_indices_func_gpu(ps, P, k, d):

    def single_batch_knn(qry, ref):
        n, d = ref.size()
        m, d = qry.size()
        mref = ref.expand(m, n, d)
        mqry = qry.expand(n, m, d).transpose(0, 1)
        dist2 = torch.sum((mqry - mref)**2, 2).squeeze()
        _, inds = torch.topk(dist2, k*d + 1, dim = 1, largest = False)
        return inds[:,1::d]

    region_idx = torch.stack([
        single_batch_knn(p, P[n]) for n, p in enumerate(ps)
    ], dim = 0)
    return region_idx

def plot(pts, fts):
    num_F = fts.size()[2]
    pts = pts[0].data.cpu().numpy()
    plt.scatter(pts[:,0], pts[:,1], s = num_F, c = "k")
    plt.savefig("./%i.png" % num_F)
    plt.cla()

if __name__ == "__main__":
    from torch.autograd import Variable
    N_rep = 1000
    N = 2
    num_points = 10000
    D = 3
    test_P  = np.random.rand(N,num_points,D).astype(np.float32)
    idx = np.random.choice(test_P.shape[1], N_rep, replace = False)
    test_ps = test_P[:,idx,:]
    test_P = Variable(torch.from_numpy(test_P)).cuda()
    test_ps = Variable(torch.from_numpy(test_ps)).cuda()
    out = knn_indices_func_gpu(test_ps, test_P, 5, 3)

    print(out)
