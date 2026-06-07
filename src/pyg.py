"""
GCN SpMM benchmark entry point.

Module layout (same directory):
  pyg.py        — CLI + main (this file)
  spmm_cuda.py  — JIT CUDA kernels (simple / degree / split_atomic)
  csr.py        — CSR extraction, degree buckets, row-split tasks
  reorder.py    — Node permutations (none, degree_*, rcm, …)
  models.py     — GCN, CuSparseGCN, CustomSpMMGCN
  data.py       — load_dataset, prepare_backend_data
  benchmark2.py — timing, correctness, model factory
"""

import inspect
import statistics
import argparse
from copy import deepcopy

import torch

# PyTorch 2.6+ defaults torch.load(weights_only=True). OGB/PyG processed files
# unpickle full Data objects; default must be False for trusted local caches.
if "weights_only" in inspect.signature(torch.load).parameters:
    _torch_load_orig = torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _torch_load_orig(*args, **kwargs)

    torch.load = _torch_load_compat

from benchmark2 import (
    CUSTOM_ALL_BACKENDS,
    benchmark_forward,
    model_for_backend,
    percentile,
    validate_spmm_correctness,
)
from data import load_dataset, move_prepared_data_to_device, prepare_backend_data
from reorder import reorder_graph
from spmm_cuda import load_spmm_extension

try:
    from torch_sparse import SparseTensor
except ImportError:
    SparseTensor = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[
            "cora",
            "reddit",
            "amazon_photo",
            "amazon_computers",
            "ogbn_products",
        ],
        default="cora",
    )
    parser.add_argument("--root", default="./pyg_data")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--timed-steps", type=int, default=100)
    parser.add_argument(
        "--backend",
        choices=[
            "edge_index",
            "spmm",
            "cusparse",
            "custom_simple",
            "custom_degree",
            "custom_split_atomic",
            "custom_all",
        ],
        default="cusparse",
    )
    parser.add_argument(
        "--orders",
        nargs="+",
        default=["none", "degree_desc", "degree_asc", "reverse", "random"],
        help="Node permutations to try. Include 'none' for reorder speedups vs identity.",
    )
    parser.add_argument(
        "--kernel-compare-orders",
        nargs="+",
        default=["none"],
        help=(
            "For custom_* / custom_all: which orders get a timed custom-kernel run "
            "and comparison to spmm (default: none). Must be a subset of --orders. "
            "Each token may be comma-separated, e.g. none,degree_desc or two args: none degree_desc."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hidden",
        type=int,
        default=16,
        help="GCN hidden width (use 64 to match common paper configs; affects SpMM feature dim).",
    )
    args = parser.parse_args()

    kco = []
    for item in args.kernel_compare_orders:
        for part in str(item).split(","):
            p = part.strip()
            if p:
                kco.append(p)
    args.kernel_compare_orders = kco
    return args


def validate_args(args):
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1.")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0.")
    if args.timed_steps < 1:
        raise ValueError("--timed-steps must be >= 1.")
    if args.hidden < 1:
        raise ValueError("--hidden must be >= 1.")

    if SparseTensor is None:
        raise ImportError(
            "This script needs torch_sparse for the spmm baseline / reordering benchmark."
        )
    if "none" not in args.orders:
        raise ValueError(
            "Include 'none' in --orders so reordering speedups can use spmm@none as reference."
        )
    for o in args.kernel_compare_orders:
        if o not in args.orders:
            raise ValueError(
                f"--kernel-compare-orders: '{o}' is not in --orders {args.orders}."
            )

    if (args.backend.startswith("custom_") or args.backend == "custom_all") and not torch.cuda.is_available():
        raise RuntimeError("Custom CUDA backends require a CUDA-capable device.")


def prepare_all_orders(args, dataset):
    """Reorder + backend-prep for every --orders entry."""
    _prep_backend = "custom_simple" if args.backend == "custom_all" else args.backend
    base_data = dataset[0]
    spmm_data = {}
    main_data = {}
    for order in args.orders:
        reordered_data, _, _ = reorder_graph(deepcopy(base_data), order, args.seed)
        spmm_data[order] = prepare_backend_data(deepcopy(reordered_data), "spmm")
        if args.backend == "spmm":
            main_data[order] = spmm_data[order]
        else:
            main_data[order] = prepare_backend_data(reordered_data, _prep_backend)
    return spmm_data, main_data, _prep_backend


def build_models(args, dataset, hidden, device):
    spmm_model = model_for_backend(
        "spmm", dataset.num_features, hidden, dataset.num_classes
    ).to(device)
    if args.backend == "custom_all":
        main_models = {
            b: model_for_backend(b, dataset.num_features, hidden, dataset.num_classes).to(
                device
            )
            for b in CUSTOM_ALL_BACKENDS
        }
    else:
        main_models = {
            args.backend: model_for_backend(
                args.backend, dataset.num_features, hidden, dataset.num_classes
            ).to(device)
        }
    return spmm_model, main_models


def bench_stats(model, data, backend_name, device, args):
    runs_ms = [
        benchmark_forward(
            model, data, device, backend_name, args.warmup, args.timed_steps
        )
        * 1000
        for _ in range(args.repeats)
    ]
    return {
        "mean": statistics.mean(runs_ms),
        "std": statistics.pstdev(runs_ms),
        "p50": percentile(runs_ms, 50),
        "p95": percentile(runs_ms, 95),
        "p99": percentile(runs_ms, 99),
    }


def print_reorder_results(spmm_results, orders):
    none_spmm_mean = spmm_results["none"]["mean"]
    print()
    print("=== Reordering benchmark (spmm: GCN + torch_sparse SparseTensor) ===")
    for order in orders:
        r = spmm_results[order]
        speedup = none_spmm_mean / r["mean"] if r["mean"] > 0 else float("inf")
        print(
            f"  {order:>11}: mean {r['mean']:.4f} ms (std: {r['std']:.4f}) | "
            f"p50 {r['p50']:.4f} | p95 {r['p95']:.4f} | p99 {r['p99']:.4f} | "
            f"speedup vs spmm@none: {speedup:.4f}x"
        )


def print_custom_all_results(args, spmm_results, main_models, main_data, device):
    print()
    print(
        "=== Custom CUDA kernels vs spmm (same order; subset: --kernel-compare-orders) ==="
    )
    print(
        "  Timings are sequential (one kernel at a time); fair comparison vs same library spmm."
    )
    for order in args.kernel_compare_orders:
        sm = spmm_results[order]["mean"]
        print(f"  {order:>11}: spmm baseline mean {sm:.4f} ms")
        for b in CUSTOM_ALL_BACKENDS:
            custom_r = bench_stats(main_models[b], main_data[order], b, device, args)
            cm = custom_r["mean"]
            speedup = sm / cm if cm > 0 else float("inf")
            short = b.replace("custom_", "")
            print(
                f"             {short:16} mean {cm:.4f} ms (std: {custom_r['std']:.4f}) | "
                f"p50 {custom_r['p50']:.4f} | p95 {custom_r['p95']:.4f} | "
                f"p99 {custom_r['p99']:.4f} | speedup vs spmm: {speedup:.4f}x"
            )


def print_custom_single_results(args, spmm_results, main_models, main_data, device):
    print()
    print(
        "=== Custom CUDA kernel vs spmm (same order; subset: --kernel-compare-orders) ==="
    )
    for order in args.kernel_compare_orders:
        custom_r = bench_stats(
            main_models[args.backend], main_data[order], args.backend, device, args
        )
        sm = spmm_results[order]["mean"]
        cm = custom_r["mean"]
        speedup = sm / cm if cm > 0 else float("inf")
        print(
            f"  {order:>11}: custom mean {cm:.4f} ms (std: {custom_r['std']:.4f}) | "
            f"p50 {custom_r['p50']:.4f} | p95 {custom_r['p95']:.4f} | "
            f"p99 {custom_r['p99']:.4f} | "
            f"speedup vs spmm: {speedup:.4f}x (spmm mean {sm:.4f} ms)"
        )


def print_other_backend_results(args, spmm_results, main_models, main_data, device):
    print()
    print(f"=== Backend {args.backend} timings (speedup vs spmm on same order) ===")
    for order in args.orders:
        other_r = bench_stats(
            main_models[args.backend], main_data[order], args.backend, device, args
        )
        sm = spmm_results[order]["mean"]
        om = other_r["mean"]
        speedup = sm / om if om > 0 else float("inf")
        print(
            f"  {order:>11}: mean {om:.4f} ms (std: {other_r['std']:.4f}) | "
            f"p50 {other_r['p50']:.4f} | p95 {other_r['p95']:.4f} | "
            f"p99 {other_r['p99']:.4f} | "
            f"speedup vs spmm: {speedup:.4f}x (spmm mean {sm:.4f} ms)"
        )


def main():
    args = parse_args()
    validate_args(args)

    if args.backend.startswith("custom_") or args.backend == "custom_all":
        load_spmm_extension(verbose=False)

    dataset = load_dataset(args.dataset, args.root)
    spmm_data, main_data, move_backend = prepare_all_orders(args, dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = args.hidden
    spmm_model, main_models = build_models(args, dataset, hidden, device)

    for order in args.orders:
        spmm_data[order] = move_prepared_data_to_device(spmm_data[order], "spmm", device)
        main_data[order] = move_prepared_data_to_device(
            main_data[order], move_backend, device
        )

    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Hidden dim: {hidden}")
    print(f"Backend (secondary benchmark): {args.backend}")
    print(f"Runs: {args.repeats}")
    print(f"Orders: {', '.join(args.orders)}")

    if args.backend == "custom_all":
        _d0 = main_data[args.kernel_compare_orders[0]]
        for b in CUSTOM_ALL_BACKENDS:
            validate_spmm_correctness(_d0, b)
    elif args.backend.startswith("custom_"):
        validate_spmm_correctness(main_data[args.kernel_compare_orders[0]], args.backend)

    spmm_results = {
        order: bench_stats(spmm_model, spmm_data[order], "spmm", device, args)
        for order in args.orders
    }

    print_reorder_results(spmm_results, args.orders)

    if args.backend == "custom_all":
        print_custom_all_results(args, spmm_results, main_models, main_data, device)
    elif args.backend.startswith("custom_"):
        print_custom_single_results(args, spmm_results, main_models, main_data, device)
    elif args.backend != "spmm":
        print_other_backend_results(args, spmm_results, main_models, main_data, device)


if __name__ == "__main__":
    main()
