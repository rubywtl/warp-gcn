# Adaptive Warp-Centric Execution for Graph Neural Networks on GPUs

A degree-aware reordering framework for accelerating Sparse Matrix Multiplication (SpMM) in Graph Convolutional Networks (GCNs). Real-world graphs follow a power-law degree distribution, causing severe load imbalance across GPU warps during neighbor aggregation. Adapti addresses this by sorting nodes by degree and distributing them evenly across thread blocks as a one-time preprocessing step, with no changes to the graph's mathematical structure.

Evaluated on Reddit and ogbn-products, Adapti achieves up to **1.21x speedup** on the GCN forward pass with negligible preprocessing overhead.

## Quick start — run benchmarks

Entry point: **`pyg.py`**. Run from the repo root on a **CUDA GPU** (custom backends require GPU).

Required Python modules in the same directory:

```text
pyg.py  spmm_cuda.py  csr.py  reorder.py  models.py  data.py  benchmark2.py
```

### Reddit (reordering + custom kernels)

```bash
python -u pyg.py --dataset reddit --backend custom_all \
  --orders none degree_desc rcm --kernel-compare-orders none,degree_desc,rcm \
  --hidden 64 --repeats 200 --warmup 50 --timed-steps 20
```

### Other datasets

```bash
# OGB ogbn-products (pip install ogb; downloads to --root)
python -u pyg.py --dataset ogbn_products --root ./pyg_data --backend custom_all \
  --orders none degree_desc rcm --kernel-compare-orders none,degree_desc,rcm \
  --hidden 64 --repeats 200 --warmup 50 --timed-steps 20

# Amazon co-purchase
python -u pyg.py --dataset amazon_photo --root ./pyg_data --backend custom_all \
  --orders none degree_desc --kernel-compare-orders none,degree_desc \
  --hidden 64 --repeats 200

python -u pyg.py --dataset amazon_computers --root ./pyg_data --backend custom_all \
  --orders none degree_desc --kernel-compare-orders none,degree_desc \
  --hidden 64 --repeats 200
```

### Library SpMM baseline only (reordering sweep)

```bash
python -u pyg.py --dataset reddit --backend spmm \
  --orders none degree_desc degree_asc reverse random \
  --hidden 64 --repeats 200 --warmup 30 --timed-steps 100
```

### Quick smoke test

```bash
python -u pyg.py --dataset reddit --backend custom_all \
  --orders none degree_desc --kernel-compare-orders none,degree_desc \
  --hidden 64 --repeats 20 --warmup 5 --timed-steps 20
```

### Useful flags

| Flag | Meaning |
|------|---------|
| `--backend custom_all` | Run `custom_simple`, `custom_degree`, `custom_split_atomic` |
| `--backend spmm` | Library baseline only (torch_sparse) |
| `--orders` | Reorder strategies (`none`, `degree_desc`, `rcm`, …) |
| `--kernel-compare-orders` | Which orders get custom-kernel timing vs spmm |
| `--hidden` | GCN hidden width (64 matches common paper configs) |

**Note:** Reported ms timings are **forward-pass only** (after graph prep). Reorder + CSR build time is not included in those numbers. For `rcm`, install **scipy** in the conda env.

Design details: see **`KERNEL_DESIGN_AND_FLOW.md`**.

---

## How it works

GCN inference is bottlenecked by SpMM in the neighbor aggregation step. Under a naive row-parallel assignment, warps assigned to high-degree nodes do orders of magnitude more work than those assigned to low-degree nodes, stalling the kernel. Adapti reorders the input graph once before inference, grouping nodes into degree-sorted buckets that are evenly distributed across GPU thread blocks. This eliminates warp divergence and improves effective GPU utilization without modifying the SpMM kernel's math or the model weights.

![Solution overview](assets/img/solution.png)

---

## Environment setup

Tested on **NVIDIA H200**, **CUDA 12.x**, **Python 3.11**.

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- CUDA-capable GPU (`nvidia-smi` shows CUDA 12.x)

### Local / generic conda

**1. Create and activate the environment**

```bash
conda create -n pyg-gpu python=3.11 -y
conda activate pyg-gpu
```

**2. Install PyTorch** (match your CUDA version)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**3. Install PyG and sparse kernels**

```bash
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

**4. Install OGB, SciPy, utilities**

```bash
pip install ogb scipy numpy
```

**5. Verify**

```bash
python -c "
import torch, torch_geometric
print('PyTorch :', torch.__version__)
print('PyG     :', torch_geometric.__version__)
print('CUDA    :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU     :', torch.cuda.get_device_name(0))
"
```

On shared clusters, point **`TMPDIR`** and PyTorch extension cache to scratch if `/tmp` is small (custom CUDA kernels JIT-compile on first run):

```bash
export TMPDIR=/path/to/scratch/tmp
export TORCH_EXTENSIONS_DIR=/path/to/scratch/torch_extensions
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR"
```

---

## Running benchmarks (details)

### End-to-end GCN forward (`pyg.py`)

Measures full 2-layer GCN forward time (warmup + timed repeats). Two printed blocks:

1. **Reordering benchmark** — `torch_sparse` SpMM for every `--orders` value (speedup vs `none`).
2. **Custom CUDA vs spmm** — for each `--kernel-compare-orders` entry, times `custom_simple`, `custom_degree`, `custom_split_atomic` vs library SpMM on the **same** reordered graph.

### Module layout

| File | Contents |
|------|----------|
| `pyg.py` | CLI and main benchmark loop |
| `reorder.py` | Node permutations (`none`, `degree_desc`, `rcm`, …) |
| `data.py` | Dataset load + backend prep (CSR, SparseTensor) |
| `spmm_cuda.py` | Custom CUDA SpMM kernels |
| `models.py` | GCN / CuSparse / CustomSpMM models |
| `benchmark2.py` | Timing, correctness checks, model factory |
