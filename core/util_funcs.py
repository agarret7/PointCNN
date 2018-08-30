# Standard Modules
import time

# External Modules
import torch
from torch import cuda, FloatTensor, LongTensor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LSHForest, NearestNeighbors
from typing import Union

# Types to allow for both CPU and GPU models.
UFloatTensor = Union[FloatTensor, cuda.FloatTensor]
ULongTensor = Union[LongTensor, cuda.LongTensor]

def knn_indices_func_cpu(rep_pts : FloatTensor,  # (N, pts, dim)
                         pts : FloatTensor,      # (N, x, dim)
                         K : int, D : int
                        ) -> LongTensor:         # (N, pts, K)
    """
    CPU-based Indexing function based on K-Nearest Neighbors search.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: "Spread" of neighboring points.
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    if rep_pts.is_cuda:
        rep_pts = rep_pts.cpu()
    if pts.is_cuda:
        pts = pts.cpu()
    rep_pts = rep_pts.data.numpy()
    pts = pts.data.numpy()

    region_idx = []

    for n, p in enumerate(rep_pts):
        P_particular = pts[n]
        nbrs = NearestNeighbors(D*K + 1, algorithm = "auto").fit(P_particular)
        indices = nbrs.kneighbors(p)[1]
        region_idx.append(indices[:,1::D])

    region_idx = torch.from_numpy(np.stack(region_idx, axis = 0))
    return region_idx

def knn_indices_func_approx(rep_pts : FloatTensor,  # (N, pts, dim)
                            pts : FloatTensor,      # (N, x, dim)
                            K : int, D : int
                           ) -> LongTensor:         # (N, pts, K)
    """
    Approximate CPU-based Indexing function based on K-Nearest Neighbors search.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: "Spread" of neighboring points.
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    if rep_pts.is_cuda:
        rep_pts = rep_pts.cpu()
    if pts.is_cuda:
        pts = pts.cpu()
    rep_pts = rep_pts.data.numpy()
    pts = pts.data.numpy()

    region_idx = []

    for n, p in enumerate(rep_pts):
        P_particular = pts[n]
        lshf = LSHForest(n_estimators = 20, n_candidates = 100, n_neighbors = D*K + 1)
        lshf.fit(P_particular)
        indices = lshf.kneighbors(p, return_distance = False)
        region_idx.append(indices[:,1::D])

def knn_indices_func_gpu(rep_pts : cuda.FloatTensor,  # (N, pts, dim)
                         pts : cuda.FloatTensor,      # (N, x, dim)
                         K : int, D : int
                        ) -> cuda.LongTensor:         # (N, pts, K)
    """
    GPU-based Indexing function based on K-Nearest Neighbors search.
    Very memory intensive, and thus unoptimal for large numbers of points.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: "Spread" of neighboring points.
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    region_idx = []

    for n, qry in enumerate(rep_pts):
        qry = qry.half()
        ref = pts[n].half()
        r_A = torch.sum(qry * qry, dim = 1, keepdim = True)
        r_B = torch.sum(ref * ref, dim = 1, keepdim = True)
        dist2 = r_A - 2 * torch.matmul(qry, torch.t(ref)) + torch.t(r_B)
        _, inds = torch.topk(dist2, D*K + 1, dim = 1, largest = False)
        region_idx.append(inds[:,1::D])

    region_idx = torch.stack(region_idx, dim = 0)

    return region_idx
