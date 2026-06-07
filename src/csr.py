"""
CSR helpers and row-bucketing for custom SpMM kernels.
"""

import torch


def csr_from_sparse_csr(adj_csr):
    """Turn PyTorch CSR into row_ptr / col_idx / values for custom kernels."""
    row_ptr = adj_csr.crow_indices().to(torch.int64).contiguous()
    col_idx = adj_csr.col_indices().to(torch.int64).contiguous()
    values = adj_csr.values().contiguous()
    return row_ptr, col_idx, values


def degree_buckets(row_ptr):
    """Bucket rows by CSR nnz: <=32, 33-256, >256 (for custom_degree)."""
    degree = row_ptr[1:] - row_ptr[:-1]
    small_rows = torch.nonzero(degree <= 32, as_tuple=False).flatten().to(torch.int64).contiguous()
    medium_rows = torch.nonzero((degree > 32) & (degree <= 256), as_tuple=False).flatten().to(torch.int64).contiguous()
    large_rows = torch.nonzero(degree > 256, as_tuple=False).flatten().to(torch.int64).contiguous()
    return degree, small_rows, medium_rows, large_rows


def build_row_split_tasks(row_ptr, chunk_size=256):
    """Split each nonempty CSR row into edge chunks (for custom_split_atomic)."""
    device = row_ptr.device
    dtype = torch.int64
    num_rows = row_ptr.size(0) - 1
    if num_rows <= 0:
        empty = torch.empty(0, device=device, dtype=dtype)
        return empty, empty, empty

    row_ptr = row_ptr.reshape(-1).to(dtype)
    starts = row_ptr[:-1]
    ends = row_ptr[1:]
    degree = ends - starts
    cs = torch.as_tensor(chunk_size, device=device, dtype=dtype)
    chunk_counts = torch.where(
        degree > 0,
        (degree + cs - 1) // cs,
        torch.zeros_like(degree),
    )
    total = int(chunk_counts.sum().item())
    if total == 0:
        empty = torch.empty(0, device=device, dtype=dtype)
        return empty, empty, empty

    rows = torch.arange(num_rows, device=device, dtype=dtype)
    task_row_id = torch.repeat_interleave(rows, chunk_counts)
    exclusive_prefix = torch.cat(
        [
            torch.zeros(1, device=device, dtype=dtype),
            chunk_counts.cumsum(0)[:-1],
        ]
    )
    task_idx = torch.arange(total, device=device, dtype=dtype)
    pos_in_row = task_idx - exclusive_prefix[task_row_id]

    task_start_edge = starts[task_row_id] + pos_in_row * cs
    task_end_edge = torch.minimum(task_start_edge + cs, ends[task_row_id])

    return (
        task_row_id.contiguous(),
        task_start_edge.contiguous(),
        task_end_edge.contiguous(),
    )
