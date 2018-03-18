import torch

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

foo = torch.FloatTensor(10, 5, 6)
bar = torch.FloatTensor(10, 6, 7)
baz = zipwith_matmul(foo, bar)
