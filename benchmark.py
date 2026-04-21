# benchmark.py
import torch
import torch.nn.functional as F
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'citeseer', 'pubmed', 'arxiv', 'reddit'],
                    help='Dataset to benchmark')
parser.add_argument('--backend', type=str, default='scatter',
                    choices=['scatter', 'spmm', 'cusparse', 'cugraph'],
                    help='Aggregation backend to benchmark. '
                         'NOTE: cugraph requires the cugraph-bench conda env — '
                         'do NOT run in gnn-bench or it will break your install')
parser.add_argument('--num-runs', type=int, default=200)
parser.add_argument('--hidden', type=int, default=64,
                    help='Hidden layer dimension')
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
        print("         pip install cugraph-cu12 pytorch-cugraph \\")
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

# ── memory estimate + OOM warning ─────────────────────────────────────────────
num_edges     = data.edge_index.shape[1]
estimated_gb  = (num_edges * num_features * 4) / (1024 ** 3)
gpu_gb        = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
print(f"Estimated peak memory: {estimated_gb:.1f} GB  |  GPU memory: {gpu_gb:.0f} GB")

if args.backend == 'scatter' and estimated_gb > gpu_gb * 0.7:
    print(f"WARNING: scatter will likely OOM on {args.dataset} "
          f"({estimated_gb:.1f}GB needed, {gpu_gb:.0f}GB available)")
    print(f"         scatter materializes all {num_edges:,} neighbor "
          f"features ({num_features}-dim) at once")
    print(f"         try --backend spmm or --backend cusparse instead")
    import sys; sys.exit(1)

# ── convert edge_index format based on backend ────────────────────────────────
if args.backend == 'spmm':
    from torch_sparse import SparseTensor
    row       = data.edge_index[0]
    col       = data.edge_index[1]
    data.edge_index = SparseTensor(
        row=col, col=row,
        sparse_sizes=(data.num_nodes, data.num_nodes)
    )
    print(f"edge_index converted to SparseTensor — {data.edge_index.nnz()} edges")

elif args.backend == 'cusparse':
    n    = data.num_nodes
    row  = data.edge_index[0]
    col  = data.edge_index[1]
    vals = torch.ones(num_edges, dtype=torch.float32, device=data.x.device)
    data.edge_index = torch.sparse_coo_tensor(
        data.edge_index, vals, (n, n)
    ).to_sparse_csr().cuda()
    print(f"edge_index converted to CSR — {num_edges} edges")

elif args.backend == 'cugraph':
    try:
        from cugraph_pyg.nn import GCNConv as CuGraphGCNConv
        use_cugraph = True
        print("cuGraph GCNConv loaded")
    except ImportError:
        print("ERROR: cugraph_pyg not found — is cugraph-bench env active?")
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

print(f"Dataset : {args.dataset} — {data.num_nodes} nodes, {num_edges} edges")
print(f"Hidden  : {args.hidden}")
print(f"p50     : {np.percentile(times, 50):.3f} ms")
print(f"p95     : {np.percentile(times, 95):.3f} ms")
print(f"p99     : {np.percentile(times, 99):.3f} ms")