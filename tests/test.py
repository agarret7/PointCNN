import unittest

import torch
from torch.autograd import Variable
import numpy as np

from PointCNN import XConv, RandPointCNN, knn_indices_func_cpu
from PointCNN.tests.util_funcs import plot_pts_and_fts

np.random.seed(0)

class BasicTests(unittest.TestCase):
    """ Basic test cases """

    def test_xconv_shape(self):
        self.assertTrue(True)

        N = 4
        D = 3
        C_in = 8
        C_out = 32
        N_neighbors = 100

        model = XConv(C_in, C_out, D, N_neighbors).cuda()
        p = Variable(torch.from_numpy(np.random.rand(N,D).astype(np.float32))).cuda()
        P = Variable(torch.from_numpy(np.random.rand(N,N_neighbors,D).astype(np.float32))).cuda()
        F = Variable(torch.from_numpy(np.random.rand(N,N_neighbors,C_in).astype(np.float32))).cuda()
        out = model(p, P, F)
        self.assertEqual(out.size(), (N, C_out))

    def test_knn(self):
        P = np.array([[[0,0],
                       [0,0.95],
                       [1,0],
                       [1,1]]])
        ps = P[:,[0,3],:]

        P = Variable(torch.from_numpy(P))
        ps = Variable(torch.from_numpy(ps))

        out = knn_indices_func_cpu(ps, P, 2).numpy()
        target = np.array([[[1,2],
                            [2,1]]])

        self.assertTrue(np.array_equal(target, out))

    def test_pointcnn_shape(self):
        N = 1
        num_points = 1000
        dims = 2
        C_in = 4
        K = 10
        D = 1

        layer1 = RandPointCNN(C_in,   8, dims, K, D, 1000, knn_indices_func_cpu).cuda()
        layer2 = RandPointCNN(   8,  16, dims, K, D,  500, knn_indices_func_cpu).cuda()
        layer3 = RandPointCNN(  16,  32, dims, K, D,  250, knn_indices_func_cpu).cuda()
        layer4 = RandPointCNN(  32,  64, dims, K, D,  125, knn_indices_func_cpu).cuda()
        layer5 = RandPointCNN(  64, 128, dims, K, D,   50, knn_indices_func_cpu).cuda()

        pts  = np.random.rand(N,num_points,dims).astype(np.float32)
        fts  = np.random.rand(N,num_points,C_in).astype(np.float32)
        pts = Variable(torch.from_numpy(pts)).cuda()
        fts = Variable(torch.from_numpy(fts)).cuda()

        if True:
            pts, fts = layer1((pts, fts))
        else:
            plot_pts_and_fts(pts, fts)
            pts, fts = layer1((pts, fts))
            plot_pts_and_fts(pts, fts)
            pts, fts = layer2((pts, fts))
            plot_pts_and_fts(pts, fts)
            pts, fts = layer3((pts, fts))
            plot_pts_and_fts(pts, fts)
            pts, fts = layer4((pts, fts))
            plot_pts_and_fts(pts, fts)
            pts, fts = layer5((pts, fts))
            plot_pts_and_fts(pts, fts)

if __name__ == "__main__":
    unittest.main()
