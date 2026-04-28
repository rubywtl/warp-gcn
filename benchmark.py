import torch
import torch.nn.functional as F
import numpy as np
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='reddit',
                    choices=['cora', 'citeseer', 'pubmed', 'arxiv', 'reddit'],
                    help='Dataset to benchmark')
parser.add_argument('--backend', type=str, default='spmm',
                    choices=['scatter', 'spmm', 'cusparse', 'cugraph'],
                    help='Aggregation backend to benchmark. '
                         'NOTE: cugraph requires the cugraph-bench conda env — '
                         'do NOT run in gnn-bench or it will break your install')
parser.add_argument('--num-runs', type=int, default=200)
parser.add_argument('--hidden', type=int, default=64,
                    help='Hidden layer dimension')
parser.add_argument('--reorder', type=str, default='none',
                    choices=['none', 'degree_asc', 'degree_desc', 'metis', 'rcm'],
                    help='Node reordering strategy applied before benchmarking.\n'
                         '  none        — original ordering (baseline)\n'
                         '  degree_asc  — sort nodes by degree ascending (low-degree first)\n'
                         '  degree_desc — sort nodes by degree descending (hubs first)\n'
                         '  metis       — METIS graph partitioning-based reorder\n'
                         '  rcm         — Reverse Cuthill-McKee bandwidth reduction\n')
parser.add_argument('--correctness', action='store_true',
                    help='Run correctness check comparing reordered vs original output')
args = parser.parse_args()

# ── cugraph env guard ─────────────────────────────────────────────────────────
if args.backend == 'cugraph':
    import os, sys
    env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if env != 'cugraph-bench':
        print("ERROR: cugraph backend must be run in the cugraph-bench conda env")
        print(f"       current env: '{env}'")
        print()
        print("       to set up the cugraph-bench env:")
        print("         conda create -n cugraph-bench python=3.11 -y")
        print("         conda activate cugraph-bench")
        print("         pip install torch==2.3.0 torchvision torchaudio \\")
        print("             --index-url https://download.pytorch.org/whl/cu121")
        print("         pip install torch_geometric ogb numpy")
        print("         pip install cugraph-pyg-cu12 \\")
        print("             --extra-index-url https://pypi.nvidia.com")
        print()
        print("       then run:")
        print("         conda activate cugraph-bench")
        print(f"        python benchmark.py --dataset {args.dataset} --backend cugraph")
        sys.exit(1)

# ── backend setup (must happen before any pyg imports) ────────────────────────
if args.backend == 'scatter':
    import torch_geometric.typing
    torch_geometric.typing.WITH_TORCH_SPARSE = False
    torch_geometric.typing.WITH_TORCH_SCATTER = False
    print("Backend: scatter_add (naive)")

elif args.backend == 'spmm':
    import torch_geometric.typing
    torch_geometric.typing.WITH_TORCH_SPARSE = True
    print("Backend: SpMM via torch_sparse")

elif args.backend == 'cusparse':
    import torch_geometric.typing
    torch_geometric.typing.WITH_TORCH_SPARSE = False
    torch_geometric.typing.WITH_TORCH_SCATTER = False
    print("Backend: cuSPARSE (torch native sparse CSR)")

elif args.backend == 'cugraph':
    import torch_geometric.typing
    torch_geometric.typing.WITH_TORCH_SPARSE = False
    torch_geometric.typing.WITH_TORCH_SCATTER = False
    print("Backend: cuGraph")

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, Reddit

# ── dataset ───────────────────────────────────────────────────────────────────
if args.dataset == 'arxiv':
    from ogb.nodeproppred import PygNodePropPredDataset
    ds   = PygNodePropPredDataset('ogbn-arxiv', root='/tmp/arxiv')
    data = ds[0].cuda()
    num_features, num_classes = 128, 40

elif args.dataset == 'reddit':
    ds   = Reddit(root='/tmp/reddit')
    data = ds[0].cuda()
    num_features, num_classes = ds.num_features, ds.num_classes

else:
    name = args.dataset.capitalize()
    ds   = Planetoid(root=f'/tmp/{args.dataset}', name=name)
    data = ds[0].cuda()
    num_features, num_classes = ds.num_features, ds.num_classes

# ── save original data for correctness check (must be before reordering) ──────
# we keep COO edge_index here — correctness check runs on raw format
# before any backend-specific conversion (SparseTensor / CSR)
data_original_for_check = None
perm_for_check = None

if args.correctness and args.reorder != 'none':
    data_original_for_check = copy.deepcopy(data)
    print("Saved original data snapshot for correctness check")

# ── data reordering ───────────────────────────────────────────────────────────
def reorder_nodes(data, strategy):
    """
    Reorder graph nodes and return (reordered_data, perm) where perm[i] is the
    original node id that moved to position i. Pass perm to test_correctness.

    Strategies
    ----------
    degree_asc / degree_desc
        Sort by degree. degree_desc puts hubs first — if load imbalance is the
        bottleneck, scheduling heavy rows early should help.
    metis
        Cluster-based reorder for cache/coalescing locality.
    rcm
        Reverse Cuthill-McKee — minimises adjacency matrix bandwidth.
    """
    from torch_geometric.utils import degree

    num_nodes  = data.num_nodes
    edge_index = data.edge_index
    perm       = None

    if strategy == 'none':
        print("Reorder: none (baseline)")
        return data, None

    elif strategy in ('degree_asc', 'degree_desc'):
        deg  = degree(edge_index[0], num_nodes=num_nodes).cpu()
        perm = torch.argsort(deg, descending=(strategy == 'degree_desc'))
        print(f"Reorder: {strategy}  |  "
              f"degree range after reorder: "
              f"{deg[perm[0]].item():.0f} → {deg[perm[-1]].item():.0f}")

    elif strategy == 'metis':
        try:
            from torch_geometric.utils import to_undirected
            from torch_geometric.transforms import METIS
            undirected_ei = to_undirected(edge_index, num_nodes=num_nodes)
            num_parts = max(2, num_nodes // 1000)
            transform  = METIS(num_parts=num_parts, recursive=False)
            cpu_data   = data.cpu()
            cpu_data   = transform(cpu_data)
            perm       = torch.argsort(cpu_data.metis_partition)
            print(f"Reorder: metis  |  {num_parts} partitions")
            data = cpu_data.cuda()
            edge_index = data.edge_index
        except Exception as e:
            print(f"WARNING: metis reorder failed ({e}), falling back to none")
            return data, None

    elif strategy == 'rcm':
        try:
            from torch_geometric.utils import to_scipy_sparse_matrix
            from scipy.sparse.csgraph import reverse_cuthill_mckee
            cpu_ei = edge_index.cpu()
            sp     = to_scipy_sparse_matrix(cpu_ei, num_nodes=num_nodes)
            order  = reverse_cuthill_mckee(sp.tocsr(), symmetric_mode=False)
            perm   = torch.from_numpy(order).long()
            print(f"Reorder: rcm  |  "
                  f"bandwidth before: {sp.tocsr().indices.max() - sp.tocsr().indices.min()}")
        except ImportError:
            print("WARNING: scipy not found (pip install scipy)")
            return data, None

    # ── apply permutation ────────────────────────────────────────────────────
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(num_nodes, device=perm.device)

    if data.x is not None:
        data.x = data.x[perm.to(data.x.device)]
    if data.y is not None:
        data.y = data.y[perm.to(data.y.device)]

    ei = data.edge_index.cpu()
    ei = inv_perm[ei[0]], inv_perm[ei[1]]
    data.edge_index = torch.stack(ei, dim=0).to(data.x.device)

    return data, perm


if args.reorder != 'none':
    print(f"\n── Applying node reorder: {args.reorder} ──")
    data, perm_for_check = reorder_nodes(data, args.reorder)
    print()
else:
    print("Reorder: none (baseline)")

# ── correctness check (runs before backend conversion, on raw COO) ────────────
def test_correctness(data_orig, data_reordered, perm, num_features, num_classes):
    """
    Compare GCN outputs on original vs reordered graph.
    Uses a fresh model with fixed seed so weights are identical both runs.
    Unshuffles the reordered output back to original node order before comparing.
    Tolerates small float32 numerical differences (< 1e-4).
    """
    print("\n── Correctness check ──")

    # fresh model with fixed seed — same weights for both runs
    torch.manual_seed(0)

    class _GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(num_features, 64)
            self.conv2 = GCNConv(64, num_classes)
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            return self.conv2(x, edge_index)

    check_model = _GCN().cuda().eval()

    with torch.no_grad():
        out_orig      = check_model(data_orig.x, data_orig.edge_index)
        out_reordered = check_model(data_reordered.x, data_reordered.edge_index)

    # unshuffle reordered output → original node order
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(len(perm), device=perm.device)
    out_unshuffled = out_reordered[inv_perm.to(out_reordered.device)]

    max_diff  = (out_orig - out_unshuffled).abs().max().item()
    mean_diff = (out_orig - out_unshuffled).abs().mean().item()

    print(f"Max diff  : {max_diff:.2e}")
    print(f"Mean diff : {mean_diff:.2e}")

    # sanity checks
    assert data_orig.num_nodes == data_reordered.num_nodes, \
        "FAILED — node count changed after reorder"
    assert data_orig.edge_index.shape[1] == data_reordered.edge_index.shape[1], \
        "FAILED — edge count changed after reorder"

    if max_diff < 1e-4:
        print("Correctness check PASSED ✓")
    else:
        print(f"Correctness check FAILED ✗ — max diff {max_diff:.2e} exceeds 1e-4")
        print("  Possible causes: edge remapping bug, feature permutation bug")

    print()


if args.correctness and args.reorder != 'none' and data_original_for_check is not None:
    test_correctness(
        data_original_for_check,
        data,                   # already reordered, still COO at this point
        perm_for_check,
        num_features,
        num_classes
    )
elif args.correctness and args.reorder == 'none':
    print("Correctness check skipped — no reordering applied (baseline == baseline)")

# ── memory estimate + OOM warning ─────────────────────────────────────────────
num_edges    = data.edge_index.shape[1] if hasattr(data.edge_index, 'shape') \
               else data.edge_index.nnz()
estimated_gb = (num_edges * num_features * 4) / (1024 ** 3)
gpu_gb       = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"Estimated peak memory: {estimated_gb:.1f} GB  |  GPU memory: {gpu_gb:.0f} GB")

if args.backend == 'scatter' and estimated_gb > gpu_gb * 0.7:
    print(f"WARNING: scatter will likely OOM on {args.dataset} "
          f"({estimated_gb:.1f} GB needed, {gpu_gb:.0f} GB available)")
    print(f"         try --backend spmm or --backend cusparse instead")
    import sys; sys.exit(1)

# ── convert edge_index to backend format ──────────────────────────────────────
# NOTE: this must happen AFTER correctness check — check needs raw COO
if args.backend == 'spmm':
    from torch_sparse import SparseTensor
    row = data.edge_index[0]
    col = data.edge_index[1]
    data.edge_index = SparseTensor(
        row=col, col=row,
        sparse_sizes=(data.num_nodes, data.num_nodes)
    )
    print(f"edge_index converted to SparseTensor — {data.edge_index.nnz()} edges")

elif args.backend == 'cusparse':
    n    = data.num_nodes
    ei   = data.edge_index
    vals = torch.ones(num_edges, dtype=torch.float32, device=data.x.device)
    data.edge_index = torch.sparse_coo_tensor(
        ei, vals, (n, n)
    ).to_sparse_csr().cuda()
    print(f"edge_index converted to CSR — {num_edges} edges")

elif args.backend == 'cugraph':
    try:
        from cugraph_pyg.nn import GCNConv as CuGraphGCNConv
        use_cugraph = True
        print("cuGraph GCNConv loaded")
    except ImportError:
        print("ERROR: cugraph_pyg not found — is cugraph-bench active?")
        import sys; sys.exit(1)

# ── model ─────────────────────────────────────────────────────────────────────
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        if args.backend == 'cugraph':
            self.conv1 = CuGraphGCNConv(num_features, args.hidden)
            self.conv2 = CuGraphGCNConv(args.hidden, num_classes)
        else:
            self.conv1 = GCNConv(num_features, args.hidden)
            self.conv2 = GCNConv(args.hidden, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

model = GCN().cuda().eval()

# ── warmup ────────────────────────────────────────────────────────────────────
with torch.no_grad():
    for _ in range(args.num_runs // 4):
        model(data.x, data.edge_index)

# ── benchmark ─────────────────────────────────────────────────────────────────
starter = torch.cuda.Event(enable_timing=True)
ender   = torch.cuda.Event(enable_timing=True)
times   = []

with torch.no_grad():
    for _ in range(args.num_runs):
        starter.record()
        model(data.x, data.edge_index)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))

print(f"\nDataset : {args.dataset} — {data.num_nodes} nodes, {num_edges} edges")
print(f"Backend : {args.backend}")
print(f"Reorder : {args.reorder}")
print(f"Hidden  : {args.hidden}")
print(f"p50     : {np.percentile(times, 50):.3f} ms")
print(f"p95     : {np.percentile(times, 95):.3f} ms")
print(f"p99     : {np.percentile(times, 99):.3f} ms")