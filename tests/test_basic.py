import unittest

import torch
from torch.autograd import Variable
import numpy as np

from pointcnn.core import XConv, knn_indices_func

class BasicTests(unittest.TestCase):
    """ Basic test cases """

    def test_xconv_shape(self):
        self.assertTrue(True)
        np.random.seed(0)

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

        out = knn_indices_func(ps, P, 2).numpy()
        target = np.array([[[1,2],
                            [2,1]]])

        self.assertTrue(np.array_equal(target, out))


if __name__ == "__main__":
    unittest.main()
