"""
Dataset loading and per-backend graph preparation.
"""

import torch
from torch_geometric.datasets import Amazon, Planetoid, Reddit
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from csr import build_row_split_tasks, csr_from_sparse_csr, degree_buckets

try:
    from torch_sparse import SparseTensor
except ImportError:
    SparseTensor = None


def load_dataset(name, root):
    """Load a supported PyG / OGB dataset."""
    dataset_name = name.lower()
    if dataset_name == "cora":
        return Planetoid(root=root, name="Cora")
    if dataset_name == "reddit":
        return Reddit(root=root)
    if dataset_name == "amazon_photo":
        return Amazon(root=root, name="Photo")
    if dataset_name == "amazon_computers":
        return Amazon(root=root, name="Computers")
    if dataset_name == "ogbn_products":
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
        except ImportError as e:
            raise ImportError(
                "dataset ogbn_products requires the ogb package: pip install ogb"
            ) from e
        return PygNodePropPredDataset(name="ogbn-products", root=root)
    raise ValueError(
        "Unsupported dataset. Choose: cora, reddit, amazon_photo, amazon_computers, "
        "ogbn_products."
    )


def prepare_backend_data(data, backend):
    """Build adjacency representation for the chosen backend."""
    if backend == "edge_index":
        return data
    if backend == "spmm":
        if SparseTensor is None:
            raise ImportError(
                "spmm backend requires torch_sparse. Install torch_sparse first."
            )
        row, col = data.edge_index
        num_nodes = data.num_nodes
        data.adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
        return data
    if backend == "cusparse":
        num_nodes = data.num_nodes
        norm_edge_index, norm_edge_weight = gcn_norm(
            data.edge_index,
            edge_weight=None,
            num_nodes=num_nodes,
            add_self_loops=True,
        )
        adj = torch.sparse_coo_tensor(
            norm_edge_index,
            norm_edge_weight,
            size=(num_nodes, num_nodes),
        ).coalesce()
        data.adj_csr = adj.to_sparse_csr()
        return data
    if backend in {"custom_simple", "custom_degree", "custom_split_atomic"}:
        num_nodes = data.num_nodes
        norm_edge_index, norm_edge_weight = gcn_norm(
            data.edge_index,
            edge_weight=None,
            num_nodes=num_nodes,
            add_self_loops=True,
        )
        adj = torch.sparse_coo_tensor(
            norm_edge_index,
            norm_edge_weight,
            size=(num_nodes, num_nodes),
        ).coalesce()
        adj_csr = adj.to_sparse_csr()
        row_ptr, col_idx, values = csr_from_sparse_csr(adj_csr)
        cache = {
            "row_ptr": row_ptr,
            "col_idx": col_idx,
            "values": values,
        }

        _, small_rows, medium_rows, large_rows = degree_buckets(row_ptr)
        cache["small_rows"] = small_rows
        cache["medium_rows"] = medium_rows
        cache["large_rows"] = large_rows

        task_row_id, task_start_edge, task_end_edge = build_row_split_tasks(
            row_ptr, chunk_size=256
        )
        cache["task_row_id"] = task_row_id
        cache["task_start_edge"] = task_start_edge
        cache["task_end_edge"] = task_end_edge
        data.custom_cache = cache
        data.adj_csr = adj_csr
        return data
    raise ValueError(
        "Unsupported backend. Choose one of: edge_index, spmm, cusparse, "
        "custom_simple, custom_degree, custom_split_atomic."
    )


def move_prepared_data_to_device(data, backend, device):
    """Move prepared tensors (and custom_cache) to device."""
    data = data.to(device)
    if backend.startswith("custom_"):
        data.custom_cache = {
            k: v.to(device) for k, v in data.custom_cache.items()
        }
    return data
