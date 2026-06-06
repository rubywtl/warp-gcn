"""
GCN-style models for each SpMM backend.
"""

import torch
from torch import Tensor
from torch_geometric.nn import GCNConv

from spmm_cuda import load_spmm_extension


class GCN(torch.nn.Module):
    """PyG GCNConv; expects edge_index or SparseTensor adj_t."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_input: Tensor) -> Tensor:
        x = self.conv1(x, edge_input).relu()
        x = self.conv2(x, edge_input)
        return x


class CuSparseGCN(torch.nn.Module):
    """torch.sparse.mm adjacency + linear layers (cusparse backend)."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels, bias=True)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels, bias=True)

    def forward(self, x: Tensor, adj_csr: Tensor) -> Tensor:
        x = torch.sparse.mm(adj_csr, x)
        x = self.lin1(x).relu()
        x = torch.sparse.mm(adj_csr, x)
        x = self.lin2(x)
        return x


class CustomSpMMGCN(torch.nn.Module):
    """Custom CUDA CSR SpMM + linear layers."""

    def __init__(self, in_channels, hidden_channels, out_channels, mode="simple"):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels, bias=True)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels, bias=True)
        self.mode = mode
        self.ext = None

    def _spmm(self, cache, x):
        if self.ext is None:
            self.ext = load_spmm_extension()
        if self.mode == "simple":
            return self.ext.spmm_simple(
                cache["row_ptr"], cache["col_idx"], cache["values"], x
            )[0]
        if self.mode == "degree":
            return self.ext.spmm_degree_aware(
                cache["row_ptr"],
                cache["col_idx"],
                cache["values"],
                x,
                cache["small_rows"],
                cache["medium_rows"],
                cache["large_rows"],
            )[0]
        if self.mode == "split_atomic":
            return self.ext.spmm_row_split_atomic(
                cache["row_ptr"],
                cache["col_idx"],
                cache["values"],
                x,
                cache["task_row_id"],
                cache["task_start_edge"],
                cache["task_end_edge"],
            )[0]
        raise ValueError(f"Unsupported custom SpMM mode: {self.mode}")

    def forward(self, x: Tensor, cache: dict) -> Tensor:
        x = self._spmm(cache, x)
        x = self.lin1(x).relu()
        x = self._spmm(cache, x)
        x = self.lin2(x)
        return x
