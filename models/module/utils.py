import torch

VERY_SMALL_NUMBER = 1e-12
INF = 1e20


def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x


def normalize_adj(mx):
    """Row-normalize matrix: symmetric normalized Laplacian"""
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag_embed(r_inv_sqrt.view(mx.size(0), -1), dim1=-2, dim2=-1)
    return torch.matmul(torch.matmul(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)
