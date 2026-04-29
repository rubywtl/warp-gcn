import torch
import torch.nn.functional as F
import numpy as np
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='products',
                    choices=['cora', 'citeseer', 'pubmed', 'arxiv', 'reddit',
                             'products', 'papers100m',
                             'synthetic-small',   # 1M nodes,  10M edges
                             'synthetic-medium',  # 5M nodes,  50M edges
                             'synthetic-large',   # 20M nodes, 200M edges
                             ],
                    help='Dataset to benchmark. '
                         'synthetic-* generate random power-law graphs at controlled scale.')
parser.add_argument('--backend', type=str, default='spmm',
                    choices=['scatter', 'spmm', 'cusparse', 'cugraph'],
                    help='Aggregation backend to benchmark. '
                         'NOTE: cugraph requires the cugraph-bench conda env — '
                         'do NOT run in gnn-bench or it will break your install')
parser.add_argument('--num-runs', type=int, default=200)
parser.add_argument('--hidden', type=int, default=64,
                    help='Hidden layer dimension')
parser.add_argument('--reorder', type=str, default='degree_desc',
                    choices=['none', 'degree_asc', 'degree_desc', 'metis', 'rcm'],
                    help='Node reordering strategy applied before benchmarking.\n'
                         '  none        — original ordering (baseline)\n'
                         '  degree_asc  — sort nodes by degree ascending (low-degree first)\n'
                         '  degree_desc — sort nodes by degree descending (hubs first)\n'
                         '  metis       — METIS graph partitioning-based reorder\n'
                         '  rcm         — Reverse Cuthill-McKee bandwidth reduction\n')
parser.add_argument('--correctness', action='store_true',
                    help='Run correctness check comparing reordered vs original output')
parser.add_argument('--synthetic-feat-dim', type=int, default=128,
                    help='Feature dimension for synthetic graphs (default: 128)')
parser.add_argument('--synthetic-avg-degree', type=int, default=10,
                    help='Target average degree for synthetic graphs (default: 10)')
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


# ── synthetic graph generator ─────────────────────────────────────────────────

def make_synthetic_graph(n_nodes, avg_degree, feat_dim, device='cuda'):
    """
    Generate a random directed graph with power-law-ish degree distribution
    using preferential attachment (Barabasi-Albert style), fast approximation.

    For large graphs we just sample edges uniformly — it's fast, gets the
    right sparsity, and is sufficient to stress-test aggregation kernels.
    The degree distribution won't be power-law but the memory access pattern
    (random sparse) is actually a harder benchmark than BA graphs.

    Returns a Data-like object with .x, .edge_index, .num_nodes attributes.
    """
    import torch
    from torch_geometric.data import Data

    n_edges = n_nodes * avg_degree
    print(f"[Synthetic] Generating {n_nodes:,} nodes, {n_edges:,} edges, "
          f"feat_dim={feat_dim} ...")

    # random edges — uniform sampling, no self-loops
    src = torch.randint(0, n_nodes, (n_edges,))
    dst = torch.randint(0, n_nodes, (n_edges,))
    # remove self-loops
    mask       = src != dst
    src, dst   = src[mask], dst[mask]
    edge_index = torch.stack([src, dst], dim=0)

    # small random features — keep memory manageable
    x = torch.randn(n_nodes, feat_dim, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, num_nodes=n_nodes)
    data = data.cuda()
    print(f"[Synthetic] Done. edge_index: {data.edge_index.shape}  "
          f"x: {data.x.shape}")
    return data


SYNTHETIC_CONFIGS = {
    #                  nodes       avg_degree
    'synthetic-small':  (1_000_000,  10),
    'synthetic-medium': (5_000_000,  10),
    'synthetic-large':  (20_000_000, 10),
}


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

elif args.dataset == 'products':
    from ogb.nodeproppred import PygNodePropPredDataset
    print("Loading ogbn-products (~2.4M nodes, 62M edges) — downloading if needed ...")
    ds   = PygNodePropPredDataset('ogbn-products', root='/tmp/products')
    data = ds[0].cuda()
    num_features, num_classes = ds.num_features, ds.num_classes
    print(f"  nodes={data.num_nodes:,}  edges={data.edge_index.shape[1]:,}  "
          f"feat={num_features}  classes={num_classes}")

elif args.dataset == 'papers100m':
    from ogb.nodeproppred import PygNodePropPredDataset
    print("Loading ogbn-papers100M (~111M nodes, 1.6B edges).")
    print("WARNING: this dataset is ~57GB on disk and ~120GB in RAM.")
    print("         Make sure you have enough memory before proceeding.")
    print("         Download may take 30+ minutes on first run.")
    ds   = PygNodePropPredDataset('ogbn-papers100M', root='/tmp/papers100m')
    data = ds[0].cuda()
    num_features, num_classes = ds.num_features, ds.num_classes
    print(f"  nodes={data.num_nodes:,}  edges={data.edge_index.shape[1]:,}  "
          f"feat={num_features}  classes={num_classes}")

elif args.dataset in SYNTHETIC_CONFIGS:
    n_nodes, avg_deg = SYNTHETIC_CONFIGS[args.dataset]
    # allow CLI overrides
    avg_deg      = args.synthetic_avg_degree
    num_features = args.synthetic_feat_dim
    num_classes  = 64   # arbitrary — only aggregation kernel is benchmarked

    # memory sanity check before allocating
    feat_mem_gb  = (n_nodes * num_features * 4) / (1024 ** 3)
    edge_mem_gb  = (n_nodes * avg_deg * 2 * 8) / (1024 ** 3)
    gpu_gb       = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[Synthetic] Memory estimate: features={feat_mem_gb:.1f}GB  "
          f"edges={edge_mem_gb:.1f}GB  GPU={gpu_gb:.0f}GB")
    if feat_mem_gb + edge_mem_gb > gpu_gb * 0.85:
        print(f"WARNING: {args.dataset} may OOM on your {gpu_gb:.0f}GB GPU.")
        print(f"         Try reducing --synthetic-feat-dim or --synthetic-avg-degree.")
        import sys; sys.exit(1)

    data = make_synthetic_graph(n_nodes, avg_deg, num_features)

else:
    name = args.dataset.capitalize()
    ds   = Planetoid(root=f'/tmp/{args.dataset}', name=name)
    data = ds[0].cuda()
    num_features, num_classes = ds.num_features, ds.num_classes

# ── save original data for correctness check (must be before reordering) ──────
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
            from torch_geometric.utils import metis_graph_cut
            perm = metis_graph_cut(
                edge_index,
                num_nodes=num_nodes,
                num_parts=max(2, num_nodes // 1000)
            )
            perm = torch.argsort(perm)
            print(f"Reorder: metis via metis_graph_cut")
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
            perm = torch.from_numpy(order.copy()).long()
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

# ── correctness check ─────────────────────────────────────────────────────────
def test_correctness(data_orig, data_reordered, perm, num_features, num_classes):
    """
    Compare GCN outputs on original vs reordered graph.
    Skipped for synthetic graphs (no ground truth, just a stress test).
    """
    print("\n── Correctness check ──")

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

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(len(perm), device=perm.device)
    out_unshuffled = out_reordered[inv_perm.to(out_reordered.device)]

    max_diff  = (out_orig - out_unshuffled).abs().max().item()
    mean_diff = (out_orig - out_unshuffled).abs().mean().item()

    print(f"Max diff  : {max_diff:.2e}")
    print(f"Mean diff : {mean_diff:.2e}")

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
    if args.dataset.startswith('synthetic'):
        print("Correctness check skipped for synthetic graphs "
              "(no canonical output to compare against)")
    else:
        test_correctness(
            data_original_for_check,
            data,
            perm_for_check,
            num_features,
            num_classes
        )
elif args.correctness and args.reorder == 'none':
    print("Correctness check skipped — no reordering applied (baseline == baseline)")

# ── memory estimate + OOM warning ─────────────────────────────────────────────
num_edges = (data.edge_index.shape[1]
             if hasattr(data.edge_index, 'shape')
             else data.edge_index.nnz())
estimated_gb = (num_edges * num_features * 4) / (1024 ** 3)
gpu_gb       = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"Estimated peak memory: {estimated_gb:.1f} GB  |  GPU memory: {gpu_gb:.0f} GB")

if args.backend == 'scatter' and estimated_gb > gpu_gb * 0.7:
    print(f"WARNING: scatter will likely OOM on {args.dataset} "
          f"({estimated_gb:.1f} GB needed, {gpu_gb:.0f} GB available)")
    print(f"         try --backend spmm or --backend cusparse instead")
    import sys; sys.exit(1)

# ── convert edge_index to backend format ──────────────────────────────────────
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
print(f"Warming up ({args.num_runs // 4} iters) ...")
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

print(f"\nDataset : {args.dataset} — {data.num_nodes:,} nodes, {num_edges:,} edges")
print(f"Backend : {args.backend}")
print(f"Reorder : {args.reorder}")
print(f"Hidden  : {args.hidden}")
print(f"p50     : {np.percentile(times, 50):.3f} ms")
print(f"p95     : {np.percentile(times, 95):.3f} ms")
print(f"p99     : {np.percentile(times, 99):.3f} ms")