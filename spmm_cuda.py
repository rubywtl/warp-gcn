"""
JIT-loaded custom CUDA CSR SpMM extension (simple, degree-aware, split-atomic).
"""

import torch
from torch.utils.cpp_extension import load_inline

_SPMM_EXT = None

CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> spmm_simple_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor x);

std::vector<torch::Tensor> spmm_degree_aware_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor x,
    torch::Tensor small_rows,
    torch::Tensor medium_rows,
    torch::Tensor large_rows);

std::vector<torch::Tensor> spmm_row_split_atomic_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor x,
    torch::Tensor task_row_id,
    torch::Tensor task_start_edge,
    torch::Tensor task_end_edge);

std::vector<torch::Tensor> spmm_simple(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor x) {
  return spmm_simple_cuda(row_ptr, col_idx, values, x);
}

std::vector<torch::Tensor> spmm_degree_aware(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor x,
    torch::Tensor small_rows,
    torch::Tensor medium_rows,
    torch::Tensor large_rows) {
  return spmm_degree_aware_cuda(
      row_ptr, col_idx, values, x, small_rows, medium_rows, large_rows);
}

std::vector<torch::Tensor> spmm_row_split_atomic(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor x,
    torch::Tensor task_row_id,
    torch::Tensor task_start_edge,
    torch::Tensor task_end_edge) {
  return spmm_row_split_atomic_cuda(
      row_ptr, col_idx, values, x, task_row_id, task_start_edge, task_end_edge);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spmm_simple", &spmm_simple, "Simple CSR SpMM (CUDA)");
  m.def("spmm_degree_aware", &spmm_degree_aware, "Degree-aware CSR SpMM (CUDA)");
  m.def(
      "spmm_row_split_atomic",
      &spmm_row_split_atomic,
      "Row-split atomic CSR SpMM (CUDA)");
}
"""

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__global__ void spmm_row_per_block_kernel(
    const int64_t* __restrict__ row_ptr,
    const int64_t* __restrict__ col_idx,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    int64_t num_rows,
    int64_t feat_dim) {
  int64_t row = blockIdx.x;
  if (row >= num_rows) {
    return;
  }
  int64_t start = row_ptr[row];
  int64_t end = row_ptr[row + 1];
  for (int64_t f = threadIdx.x; f < feat_dim; f += blockDim.x) {
    scalar_t acc = static_cast<scalar_t>(0);
    for (int64_t e = start; e < end; ++e) {
      int64_t col = col_idx[e];
      acc += values[e] * x[col * feat_dim + f];
    }
    y[row * feat_dim + f] = acc;
  }
}

template <typename scalar_t>
__global__ void spmm_small_kernel(
    const int64_t* __restrict__ row_ptr,
    const int64_t* __restrict__ col_idx,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ rows,
    int64_t num_rows_in_bucket,
    scalar_t* __restrict__ y,
    int64_t feat_dim) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_rows_in_bucket) {
    return;
  }
  int64_t row = rows[idx];
  int64_t start = row_ptr[row];
  int64_t end = row_ptr[row + 1];
  for (int64_t f = 0; f < feat_dim; ++f) {
    scalar_t acc = static_cast<scalar_t>(0);
    for (int64_t e = start; e < end; ++e) {
      int64_t col = col_idx[e];
      acc += values[e] * x[col * feat_dim + f];
    }
    y[row * feat_dim + f] = acc;
  }
}

template <typename scalar_t>
__global__ void spmm_warp_per_row_kernel(
    const int64_t* __restrict__ row_ptr,
    const int64_t* __restrict__ col_idx,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ rows,
    int64_t num_rows_in_bucket,
    scalar_t* __restrict__ y,
    int64_t feat_dim) {
  int lane = threadIdx.x & 31;
  int warp_id_in_block = threadIdx.x >> 5;
  int warps_per_block = blockDim.x >> 5;
  int64_t global_warp_id = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id_in_block;
  if (global_warp_id >= num_rows_in_bucket) {
    return;
  }
  int64_t row = rows[global_warp_id];
  int64_t start = row_ptr[row];
  int64_t end = row_ptr[row + 1];
  for (int64_t f = lane; f < feat_dim; f += 32) {
    scalar_t acc = static_cast<scalar_t>(0);
    for (int64_t e = start; e < end; ++e) {
      int64_t col = col_idx[e];
      acc += values[e] * x[col * feat_dim + f];
    }
    y[row * feat_dim + f] = acc;
  }
}

template <typename scalar_t>
__global__ void spmm_row_split_atomic_kernel(
    const int64_t* __restrict__ col_idx,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ task_row_id,
    const int64_t* __restrict__ task_start_edge,
    const int64_t* __restrict__ task_end_edge,
    int64_t num_tasks,
    scalar_t* __restrict__ y,
    int64_t feat_dim) {
  int lane = threadIdx.x & 31;
  int warp_id_in_block = threadIdx.x >> 5;
  int warps_per_block = blockDim.x >> 5;
  int64_t global_warp_id = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id_in_block;
  if (global_warp_id >= num_tasks) {
    return;
  }

  int64_t row = task_row_id[global_warp_id];
  int64_t start = task_start_edge[global_warp_id];
  int64_t end = task_end_edge[global_warp_id];
  for (int64_t f = lane; f < feat_dim; f += 32) {
    scalar_t acc = static_cast<scalar_t>(0);
    for (int64_t e = start; e < end; ++e) {
      int64_t col = col_idx[e];
      acc += values[e] * x[col * feat_dim + f];
    }
    atomicAdd(&y[row * feat_dim + f], acc);
  }
}

std::vector<torch::Tensor> spmm_simple_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor x) {
  CHECK_INPUT(row_ptr);
  CHECK_INPUT(col_idx);
  CHECK_INPUT(values);
  CHECK_INPUT(x);
  TORCH_CHECK(row_ptr.dtype() == torch::kInt64, "row_ptr must be int64");
  TORCH_CHECK(col_idx.dtype() == torch::kInt64, "col_idx must be int64");

  auto num_rows = row_ptr.size(0) - 1;
  auto feat_dim = x.size(1);
  auto y = torch::zeros({num_rows, feat_dim}, x.options());
  constexpr int threads = 128;
  const int blocks = static_cast<int>(num_rows);

  AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "spmm_simple_cuda", ([&] {
    spmm_row_per_block_kernel<scalar_t>
        <<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
        row_ptr.data_ptr<int64_t>(),
        col_idx.data_ptr<int64_t>(),
        values.data_ptr<scalar_t>(),
        x.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        num_rows,
        feat_dim);
  }));
  return {y};
}

std::vector<torch::Tensor> spmm_degree_aware_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor x,
    torch::Tensor small_rows,
    torch::Tensor medium_rows,
    torch::Tensor large_rows) {
  CHECK_INPUT(row_ptr);
  CHECK_INPUT(col_idx);
  CHECK_INPUT(values);
  CHECK_INPUT(x);
  CHECK_INPUT(small_rows);
  CHECK_INPUT(medium_rows);
  CHECK_INPUT(large_rows);
  TORCH_CHECK(small_rows.dtype() == torch::kInt64, "small_rows must be int64");
  TORCH_CHECK(medium_rows.dtype() == torch::kInt64, "medium_rows must be int64");
  TORCH_CHECK(large_rows.dtype() == torch::kInt64, "large_rows must be int64");

  auto num_rows = row_ptr.size(0) - 1;
  auto feat_dim = x.size(1);
  auto y = torch::zeros({num_rows, feat_dim}, x.options());

  constexpr int small_threads = 128;
  constexpr int warp_threads = 128;  // 4 warps / block
  const int small_blocks = static_cast<int>((small_rows.size(0) + small_threads - 1) / small_threads);
  const int medium_blocks = static_cast<int>((medium_rows.size(0) + 3) / 4);
  const int large_blocks = static_cast<int>((large_rows.size(0) + 3) / 4);

  AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "spmm_degree_aware_cuda", ([&] {
    if (small_rows.size(0) > 0) {
      spmm_small_kernel<scalar_t>
          <<<small_blocks, small_threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
          row_ptr.data_ptr<int64_t>(),
          col_idx.data_ptr<int64_t>(),
          values.data_ptr<scalar_t>(),
          x.data_ptr<scalar_t>(),
          small_rows.data_ptr<int64_t>(),
          small_rows.size(0),
          y.data_ptr<scalar_t>(),
          feat_dim);
    }
    if (medium_rows.size(0) > 0) {
      spmm_warp_per_row_kernel<scalar_t>
          <<<medium_blocks, warp_threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
          row_ptr.data_ptr<int64_t>(),
          col_idx.data_ptr<int64_t>(),
          values.data_ptr<scalar_t>(),
          x.data_ptr<scalar_t>(),
          medium_rows.data_ptr<int64_t>(),
          medium_rows.size(0),
          y.data_ptr<scalar_t>(),
          feat_dim);
    }
    if (large_rows.size(0) > 0) {
      spmm_warp_per_row_kernel<scalar_t>
          <<<large_blocks, warp_threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
          row_ptr.data_ptr<int64_t>(),
          col_idx.data_ptr<int64_t>(),
          values.data_ptr<scalar_t>(),
          x.data_ptr<scalar_t>(),
          large_rows.data_ptr<int64_t>(),
          large_rows.size(0),
          y.data_ptr<scalar_t>(),
          feat_dim);
    }
  }));
  return {y};
}

std::vector<torch::Tensor> spmm_row_split_atomic_cuda(
    torch::Tensor row_ptr,
    torch::Tensor col_idx,
    torch::Tensor values,
    torch::Tensor x,
    torch::Tensor task_row_id,
    torch::Tensor task_start_edge,
    torch::Tensor task_end_edge) {
  CHECK_INPUT(row_ptr);
  CHECK_INPUT(col_idx);
  CHECK_INPUT(values);
  CHECK_INPUT(x);
  CHECK_INPUT(task_row_id);
  CHECK_INPUT(task_start_edge);
  CHECK_INPUT(task_end_edge);
  TORCH_CHECK(task_row_id.dtype() == torch::kInt64, "task_row_id must be int64");
  TORCH_CHECK(task_start_edge.dtype() == torch::kInt64, "task_start_edge must be int64");
  TORCH_CHECK(task_end_edge.dtype() == torch::kInt64, "task_end_edge must be int64");

  auto num_rows = row_ptr.size(0) - 1;
  auto feat_dim = x.size(1);
  auto num_tasks = task_row_id.size(0);
  auto y = torch::zeros({num_rows, feat_dim}, x.options());

  constexpr int warp_threads = 128;  // 4 warps / block
  int blocks = static_cast<int>((num_tasks + 3) / 4);
  AT_DISPATCH_FLOATING_TYPES(values.scalar_type(), "spmm_row_split_atomic_cuda", ([&] {
    if (num_tasks > 0) {
      spmm_row_split_atomic_kernel<scalar_t>
          <<<blocks, warp_threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
          col_idx.data_ptr<int64_t>(),
          values.data_ptr<scalar_t>(),
          x.data_ptr<scalar_t>(),
          task_row_id.data_ptr<int64_t>(),
          task_start_edge.data_ptr<int64_t>(),
          task_end_edge.data_ptr<int64_t>(),
          num_tasks,
          y.data_ptr<scalar_t>(),
          feat_dim);
    }
  }));
  return {y};
}
"""


def load_spmm_extension(verbose=False):
    """Build and load custom CUDA SpMM extension once per process."""
    global _SPMM_EXT
    if _SPMM_EXT is not None:
        return _SPMM_EXT

    if not torch.cuda.is_available():
        raise RuntimeError("Custom CUDA SpMM backend requires CUDA.")

    _SPMM_EXT = load_inline(
        name="csr_spmm_ext_atomic",
        cpp_sources=[CPP_SOURCE],
        cuda_sources=[CUDA_SOURCE],
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_cflags=["-O3"],
        verbose=verbose,
    )
    return _SPMM_EXT
