# External Modules
import numpy as np
import matplotlib.pyplot as plt

# Internal Modules
from PointCNN.core import UFloatTensor

def plot_pts_and_fts(pts : UFloatTensor,  # (N, x, dims)
                     fts : UFloatTensor,  # (N, x, y)
                     fts_scale : int
                    ) -> None:
    """
    Visualization function. Shows points and number of features, represented by
    the size of the point.
    :param pts: Point cloud such that fts[:,p_idx,:] is the feature associated
    with pts[:,p_idx,:].
    :param fts: Features such that pts[:,p_idx,:] is the feature associated
    with fts[:,p_idx,:].
    :param fts_scale: Scale of size for fts.
    """
    if pts.is_cuda:
        pts = pts.cpu()
    num_F = fts.size()[2]
    pts = pts[0].data.numpy()
    plt.scatter(pts[:,0], pts[:,1], s = fts_scale * num_F, c = "k")
    plt.show()
    input("Press ENTER to continue...")
    plt.cla()

def plot_neighborhood(pts : UFloatTensor,          # (N, x, dims)
                      rep_pts : UFloatTensor,      # (N, P, dims)
                      pts_regional : UFloatTensor  # (N, P, dims)
                     ) -> None:
    """
    Visualization function. Shows neighborhood points around a randomly
    selected representative.
    :param pts: Point cloud.
    :param rep_pts: Representative points.
    :param pts_regional: Regional neighborhoods around representative points.
    """
    if rep_pts.is_cuda:
        rep_pts = rep_pts.cpu()
        pts_regional = pts_regional.cpu()
    n = np.randint(0, rep_pts.shape[0])
    t = np.randint(0, rep_pts.shape[1])
    test_point = rep_pts[n,t,:].data.numpy()
    neighborhood = pts_regional[n,t,:,:].data.numpy()
    plt.scatter(pts[n][:,0], pts[n][:,1])
    plt.scatter(test_point[0], test_point[1], s = 100, c = 'green')
    plt.scatter(neighborhood[:,0], neighborhood[:,1], s = 100, c = 'red')
    plt.show()
    plt.cla()
