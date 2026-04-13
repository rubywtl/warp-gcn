# benchmark.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'citeseer', 'pubmed', 'arxiv'],
                    help='Dataset to benchmark')
args = parser.parse_args()

# load dataset
if args.dataset == 'arxiv':
    from ogb.nodeproppred import PygNodePropPredDataset
    ds = PygNodePropPredDataset('ogbn-arxiv', root='/tmp/arxiv')
    data = ds[0].cuda()
    num_features, num_classes = 128, 40
else:
    name = args.dataset.capitalize()
    ds = Planetoid(root=f'/tmp/{args.dataset}', name=name)
    data = ds[0].cuda()
    num_features, num_classes = ds.num_features, ds.num_classes

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

model = GCN().cuda().eval()

with torch.no_grad():
    for _ in range(50):
        model(data.x, data.edge_index)

starter = torch.cuda.Event(enable_timing=True)
ender   = torch.cuda.Event(enable_timing=True)
times   = []

with torch.no_grad():
    for _ in range(200):
        starter.record()
        model(data.x, data.edge_index)
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))

print(f"Dataset : {args.dataset} — {data.num_nodes} nodes, {data.num_edges} edges")
print(f"p50     : {np.percentile(times, 50):.3f} ms")
print(f"p95     : {np.percentile(times, 95):.3f} ms")
print(f"p99     : {np.percentile(times, 99):.3f} ms")