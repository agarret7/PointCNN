import unittest

import torch
from torch.autograd import Variable
import numpy as np

from pointcnn.core import XConv

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

        model = XConv(C_in, C_out, D, N_neighbors)
        test_p = Variable(torch.from_numpy(np.random.rand(N,D).astype(np.float32)))
        test_P = Variable(torch.from_numpy(np.random.rand(N,N_neighbors,D).astype(np.float32)))
        test_F = Variable(torch.from_numpy(np.random.rand(N,N_neighbors,C_in).astype(np.float32)))
        test_out = model(test_p, test_P, test_F)
        self.assertEqual(test_out.size(), (N, C_out))

if __name__ == "__main__":
    unittest.main()
