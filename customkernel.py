import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class WarpCentricGCNConv(MessagePassing):
    """
    Drop-in replacement for PyG's GCNConv.
    Swap in your custom CUDA kernel inside forward().
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add')
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # --- normalization (same as vanilla GCNConv) ---
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        deg            = degree(edge_index[0], x.size(0), dtype=x.dtype)
        deg_inv_sqrt   = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm           = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]

        # --- linear transform ---
        x = self.lin(x)

        # --- aggregation: replace this with your warp-centric kernel ---
        # TODO: call your custom CUDA kernel here instead of propagate()
        # e.g. x = warp_centric_aggregate(x, edge_index, norm)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j