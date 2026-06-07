"""
Node permutation strategies (none, degree, RCM, etc.).
"""

import torch
from torch import Tensor


def rcm_gather_perm(edge_index: Tensor, num_nodes: int, device: torch.device) -> Tensor:
    """RCM ordering via SciPy on symmetrized adjacency (CPU)."""
    try:
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee
    except ImportError as e:
        raise ImportError(
            "order='rcm' requires SciPy (pip install scipy). "
        ) from e

    row = edge_index[0].detach().cpu().numpy()
    col = edge_index[1].detach().cpu().numpy()
    row_sym = np.concatenate([row, col])
    col_sym = np.concatenate([col, row])
    data = np.ones(row_sym.shape[0], dtype=np.float64)
    adj = csr_matrix((data, (row_sym, col_sym)), shape=(num_nodes, num_nodes))
    perm_scipy = reverse_cuthill_mckee(adj, symmetric_mode=True)
    inv = np.empty(num_nodes, dtype=np.int64)
    inv[perm_scipy] = np.arange(num_nodes, dtype=np.int64)
    return torch.as_tensor(inv, device=device, dtype=torch.long)


def apply_permutation(data, perm):
    """Apply node permutation: remap x, masks, and edge_index."""
    num_nodes = data.num_nodes
    row, col = data.edge_index

    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(num_nodes, device=perm.device)

    data.x = data.x[perm]
    if data.y is not None:
        data.y = data.y[perm]
    if hasattr(data, "train_mask"):
        data.train_mask = data.train_mask[perm]
    if hasattr(data, "val_mask"):
        data.val_mask = data.val_mask[perm]
    if hasattr(data, "test_mask"):
        data.test_mask = data.test_mask[perm]

    data.edge_index = torch.stack([inv[row], inv[col]], dim=0)
    return data, perm, inv


def reorder_graph(data, order, seed=42):
    """Build a permutation from strategy and apply it to data."""
    num_nodes = data.num_nodes
    row, col = data.edge_index
    degree = torch.bincount(torch.cat([row, col]), minlength=num_nodes)

    if order == "none":
        perm = torch.arange(num_nodes)
    elif order == "degree_desc":
        perm = torch.argsort(degree, descending=True)
    elif order == "degree_asc":
        perm = torch.argsort(degree, descending=False)
    elif order == "reverse":
        perm = torch.arange(num_nodes - 1, -1, -1)
    elif order == "random":
        generator = torch.Generator()
        generator.manual_seed(seed)
        perm = torch.randperm(num_nodes, generator=generator)
    elif order == "rcm":
        perm = rcm_gather_perm(torch.stack([row, col], dim=0), num_nodes, row.device)
    else:
        raise ValueError(
            "Unsupported order. Choose from: none, degree_desc, degree_asc, reverse, random, rcm."
        )

    return apply_permutation(data, perm)
