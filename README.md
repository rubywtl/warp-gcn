# Adaptive Warp-Centric Execution for Graph Neural Networks on GPUs

A degree-aware reordering framework for accelerating Sparse Matrix Multiplication (SpMM) in Graph Convolutional Networks (GCNs). Real-world graphs follow a power-law degree distribution, causing severe load imbalance across GPU warps during neighbor aggregation. Adapti addresses this by sorting nodes by degree and distributing them evenly across thread blocks as a one-time preprocessing step, with no changes to the graph's mathematical structure.

Evaluated on Reddit and ogbn-products, Adapti achieves up to **1.21x speedup** on the GCN forward pass with negligible preprocessing overhead.

## How it works

GCN inference is bottlenecked by SpMM in the neighbor aggregation step. Under a naive row-parallel assignment, warps assigned to high-degree nodes do orders of magnitude more work than those assigned to low-degree nodes, stalling the kernel. Adapti reorders the input graph once before inference, grouping nodes into degree-sorted buckets that are evenly distributed across GPU thread blocks. This eliminates warp divergence and improves effective GPU utilization without modifying the SpMM kernel's math or the model weights.


## Environment setup

Tested on H200, CUDA 12.1, Python 3.11.

### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- CUDA 12.1 driver (`nvidia-smi` should show CUDA 12.x)

### Steps

**1. Create and activate the environment**
```bash
conda create -n bench python=3.11 -y
conda activate bench
```

**2. Install PyTorch**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**3. Install PyG and sparse kernels**
```bash
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

**4. Install OGB and utilities**
```bash
pip install ogb urllib3 numpy
```

**5. Verify**
```bash
python -c "
import torch, torch_geometric
print('PyTorch :', torch.__version__)
print('PyG     :', torch_geometric.__version__)
print('CUDA    :', torch.cuda.is_available())
print('GPU     :', torch.cuda.get_device_name(0))
"
```

## Running benchmarks

### E2E Implementation
coming soon...

### Single kernel bench
Benchmarks use CUDA events for precise kernel timing.

### Analyze previous kernels
```bash
python benchmark.py --dataset reddit
python benchmark.py --dataset products
python benchmark.py --dataset arxiv
```
Then run benchmarks with the same `benchmark.py` script under the new environment.

### Test our kernels


