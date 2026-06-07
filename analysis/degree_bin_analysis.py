# degree_bin_analysis.py
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='reddit',
                    choices=['cora', 'citeseer', 'pubmed', 'arxiv', 'reddit'])
parser.add_argument('--backend', type=str, default='cusparse',
                    choices=['scatter', 'spmm', 'cusparse'])
args = parser.parse_args()

# ── backend setup ─────────────────────────────────────────────────────────────
if args.backend == 'scatter':
    import torch_geometric.typing
    torch_geometric.typing.WITH_TORCH_SPARSE = False
    torch_geometric.typing.WITH_TORCH_SCATTER = False
elif args.backend == 'spmm':
    import torch_geometric.typing
    torch_geometric.typing.WITH_TORCH_SPARSE = True
elif args.backend == 'cusparse':
    import torch_geometric.typing
    torch_geometric.typing.WITH_TORCH_SPARSE = False
    torch_geometric.typing.WITH_TORCH_SCATTER = False

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.utils import subgraph

# ── dataset ───────────────────────────────────────────────────────────────────
if args.dataset == 'arxiv':
    from ogb.nodeproppred import PygNodePropPredDataset
    ds   = PygNodePropPredDataset('ogbn-arxiv', root='/tmp/arxiv')
    data = ds[0]
    num_features, num_classes = 128, 40
elif args.dataset == 'reddit':
    ds   = Reddit(root='/tmp/reddit')
    data = ds[0]
    num_features, num_classes = ds.num_features, ds.num_classes
else:
    name = args.dataset.capitalize()
    ds   = Planetoid(root=f'/tmp/{args.dataset}', name=name)
    data = ds[0]
    num_features, num_classes = ds.num_features, ds.num_classes

# ── degree distribution ───────────────────────────────────────────────────────
degree = torch.bincount(data.edge_index[0], minlength=data.num_nodes)

print(f"\nDataset : {args.dataset}")
print(f"Backend : {args.backend}")
print(f"Nodes   : {data.num_nodes:,}")
print(f"Edges   : {data.num_edges:,}")
print(f"Avg deg : {degree.float().mean():.1f}")
print(f"Max deg : {degree.max().item()}")
print()

# ── bin definitions ───────────────────────────────────────────────────────────
bin_defs = [
    ('low    (0-5)',    (degree >= 0)  & (degree <= 5)),
    ('medium (6-20)',   (degree > 5)   & (degree <= 20)),
    ('high   (21-100)', (degree > 20)  & (degree <= 100)),
    ('very high (100+)',(degree > 100)),
]

# ── print degree stats ────────────────────────────────────────────────────────
print(f"{'Bin':<22} {'Nodes':>8} {'%':>6} {'Avg deg':>8} {'Max deg':>8}")
print("-" * 58)

bin_stats = []
for bin_name, mask in bin_defs:
    node_ids = mask.nonzero(as_tuple=True)[0]
    count    = node_ids.shape[0]
    bin_deg  = degree[node_ids]
    avg_d    = bin_deg.float().mean().item() if count > 0 else 0
    max_d    = bin_deg.max().item() if count > 0 else 0
    pct      = count / data.num_nodes * 100
    print(f"{bin_name:<22} {count:>8,} {pct:>5.1f}% {avg_d:>8.1f} {max_d:>8}")
    bin_stats.append((bin_name, node_ids, count))

# ── per-bin latency ───────────────────────────────────────────────────────────
print(f"\n{'Bin':<22} {'Nodes':>8} {'p50 (ms)':>10} {'p95 (ms)':>10} {'p99 (ms)':>10} {'us/node':>10}")
print("-" * 78)

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, num_classes)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

for bin_name, node_ids, count in bin_stats:
    torch.cuda.empty_cache()
    gc.collect()

    if count == 0:
        print(f"{bin_name:<22} {0:>8,} {'—':>10} {'—':>10} {'—':>10} {'—':>10}")
        continue

    # build subgraph
    sub_edge_index, _ = subgraph(
        node_ids, data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes
    )

    sub_x = data.x[node_ids].cuda()
    n     = node_ids.shape[0]

    if args.backend == 'cusparse':
        vals     = torch.ones(sub_edge_index.shape[1], dtype=torch.float32)
        sub_edge = torch.sparse_coo_tensor(
            sub_edge_index, vals, (n, n)
        ).to_sparse_csr().cuda()
    elif args.backend == 'spmm':
        from torch_sparse import SparseTensor
        sub_edge = SparseTensor(
            row=sub_edge_index[1],
            col=sub_edge_index[0],
            sparse_sizes=(n, n)
        )
    else:
        sub_edge = sub_edge_index.cuda()

    model = GCN().cuda().eval()

    # warmup
    with torch.no_grad():
        for _ in range(20):
            try:
                model(sub_x, sub_edge)
            except Exception:
                break

    # timed runs
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    times   = []

    with torch.no_grad():
        for _ in range(100):
            try:
                starter.record()
                model(sub_x, sub_edge)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))
            except RuntimeError:
                break

    del model, sub_x, sub_edge
    torch.cuda.empty_cache()
    gc.collect()

    if len(times) == 0:
        print(f"{bin_name:<22} {count:>8,} {'OOM':>10} {'OOM':>10} {'OOM':>10} {'OOM':>10}")
    else:
        p50    = np.percentile(times, 50)
        p95    = np.percentile(times, 95)
        p99    = np.percentile(times, 99)
        us_per = p50 * 1000 / count
        print(f"{bin_name:<22} {count:>8,} {p50:>10.3f} {p95:>10.3f} {p99:>10.3f} {us_per:>10.4f}")

print()
print("us/node = p50 latency per node — should explode for very high bin if divergence dominates")