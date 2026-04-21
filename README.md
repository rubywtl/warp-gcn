## Environment setup

Tested on H100, CUDA 12.1, Python 3.11.

### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- CUDA 12.1 driver (`nvidia-smi` should show CUDA 12.x)

### Steps

**1. Create and activate the environment**
```bash
conda create -n gnn-bench python=3.11 -y
conda activate gnn-bench
```

**2. Install PyTorch via pip (avoids MKL conflict with conda)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install cugraph-cu12
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

### Benchmarks
cuda timer
```bash
python benchmark.py --dataset cora
python benchmark.py --dataset arxiv
python benchmark.py --dataset citeseer
python benchmark.py --dataset pubmed
```

for CuGraph
we have to use a different env to run this 
```bash
conda create -n cugraph-bench python=3.11 -y
conda activate cugraph-bench
```