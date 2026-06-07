"""
The benchmark focuses on timing, correctness checks, and model factory
of our custom SpMM backends, as well as the built-in edge_index and spmm backends.
"""

import time

import torch

from models import CuSparseGCN, CustomSpMMGCN, GCN
from kernels.spmm_cuda import load_spmm_extension

CUSTOM_ALL_BACKENDS = (
    "custom_simple",
    "custom_degree",
    "custom_split_atomic",
)


def model_for_backend(backend, in_channels, hidden_channels, out_channels):
    """Construct the GCN-style model for a backend name."""
    if backend == "cusparse":
        return CuSparseGCN(in_channels, hidden_channels, out_channels)
    if backend == "custom_simple":
        return CustomSpMMGCN(in_channels, hidden_channels, out_channels, mode="simple")
    if backend == "custom_degree":
        return CustomSpMMGCN(in_channels, hidden_channels, out_channels, mode="degree")
    if backend == "custom_split_atomic":
        return CustomSpMMGCN(
            in_channels, hidden_channels, out_channels, mode="split_atomic"
        )
    if backend in {"edge_index", "spmm"}:
        return GCN(in_channels, hidden_channels, out_channels)
    raise ValueError(f"Unsupported backend for model construction: {backend}")


def benchmark_forward(model, data, device, backend, warmup_steps=30, timed_steps=100):
    """Time average forward pass (warmup excluded)."""
    if backend == "edge_index":
        edge_input = data.edge_index
    elif backend == "spmm":
        edge_input = data.adj_t
    elif backend == "cusparse":
        edge_input = data.adj_csr
    elif backend in {"custom_simple", "custom_degree", "custom_split_atomic"}:
        edge_input = data.custom_cache
    else:
        raise ValueError("Unsupported backend in benchmark_forward.")
    model.eval()
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(data.x, edge_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(timed_steps):
            _ = model(data.x, edge_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return elapsed / timed_steps


def percentile(values, pct):
    """Percentile with linear interpolation."""
    if not values:
        raise ValueError("Cannot compute percentile of empty values.")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def validate_spmm_correctness(data, backend):
    """Compare custom SpMM output to torch.sparse.mm."""
    if backend not in {"custom_simple", "custom_degree", "custom_split_atomic"}:
        return
    ext = load_spmm_extension(verbose=False)
    cache = data.custom_cache
    x = data.x
    ref = torch.sparse.mm(data.adj_csr, x)
    if backend == "custom_simple":
        out = ext.spmm_simple(
            cache["row_ptr"], cache["col_idx"], cache["values"], x
        )[0]
    elif backend == "custom_degree":
        out = ext.spmm_degree_aware(
            cache["row_ptr"],
            cache["col_idx"],
            cache["values"],
            x,
            cache["small_rows"],
            cache["medium_rows"],
            cache["large_rows"],
        )[0]
    else:
        out = ext.spmm_row_split_atomic(
            cache["row_ptr"],
            cache["col_idx"],
            cache["values"],
            x,
            cache["task_row_id"],
            cache["task_start_edge"],
            cache["task_end_edge"],
        )[0]
    max_abs_diff = (out - ref).abs().max().item()
    print(f"Correctness check ({backend}) | max abs diff vs torch.sparse.mm: {max_abs_diff:.6e}")
