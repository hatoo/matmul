#include "cute/numeric/integral_constant.hpp"
#include "cute/tensor.hpp"
#include <cstdint>

template <class M, class N, class K>
__global__ void cute_matmul_kernel01(float *a, float *b, float *c, M m, N n,
                                     K k) {
  cute::Tensor A =
      cute::make_tensor(cute::make_gmem_ptr(a), cute::make_shape(m, k),
                        cute::make_stride(k, cute::_1{}));
  cute::Tensor B =
      cute::make_tensor(cute::make_gmem_ptr(b), cute::make_shape(n, k),
                        cute::make_stride(cute::_1{}, n));
  cute::Tensor C =
      cute::make_tensor(cute::make_gmem_ptr(c), cute::make_shape(m, n),
                        cute::make_stride(n, cute::_1{}));

  auto block = cute::make_shape(cute::_128{}, cute::_128{}, cute::_8{});

  auto block_coord = cute::make_coord(blockIdx.x, blockIdx.y, cute::_);
  cute::Layout thread_layout =
      cute::make_layout(cute::make_shape(cute::_16{}, cute::_16{}));

  cute::Tensor A_block =
      cute::local_tile(A, block, block_coord,
                       cute::make_step(cute::_1{}, cute::X{}, cute::_1{}));
  cute::Tensor B_block =
      cute::local_tile(B, block, block_coord,
                       cute::make_step(cute::X{}, cute::_1{}, cute::_1{}));
  cute::Tensor C_block =
      cute::local_tile(C, block, block_coord,
                       cute::make_step(cute::_1{}, cute::_1{}, cute::X{}));

  cute::Tensor A_block_copy = cute::local_partition(
      A_block, cute::make_layout(cute::make_shape(cute::_32{}, cute::_8{})),
      threadIdx.x);

  cute::Tensor B_block_copy = cute::local_partition(
      B_block, cute::make_layout(cute::make_shape(cute::_32{}, cute::_8{})),
      threadIdx.x);

  cute::Tensor C_thread =
      cute::local_partition(C_block, thread_layout, threadIdx.x);

  __shared__ float a_shared[128 * 8];
  __shared__ float b_shared[128 * 8];

  cute::Tensor A_shared =
      cute::make_tensor(cute::make_smem_ptr(a_shared),
                        cute::make_shape(cute::_128{}, cute::_8{}));
  cute::Tensor B_shared =
      cute::make_tensor(cute::make_smem_ptr(b_shared),
                        cute::make_shape(cute::_128{}, cute::_8{}));

  cute::Tensor A_shared_copy = cute::local_partition(
      A_shared, cute::make_layout(cute::make_shape(cute::_32{}, cute::_8{})),
      threadIdx.x);
  cute::Tensor B_shared_copy = cute::local_partition(
      B_shared, cute::make_layout(cute::make_shape(cute::_32{}, cute::_8{})),
      threadIdx.x);

  cute::Tensor A_shared_local =
      cute::local_partition(A_shared, thread_layout, threadIdx.x,
                            cute::make_step(cute::_1{}, cute::X{}));
  cute::Tensor B_shared_local =
      cute::local_partition(B_shared, thread_layout, threadIdx.x,
                            cute::make_step(cute::X{}, cute::_1{}));

  cute::Tensor C_reg =
      cute::make_tensor<float>(cute::make_shape(cute::_8{}, cute::_8{}));

  cute::clear(C_reg);

  for (int i = 0; i < k / 8; ++i) {
    cute::copy(A_block_copy(cute::_, cute::_, i), A_shared_copy);
    cute::copy(B_block_copy(cute::_, cute::_, i), B_shared_copy);

    cute::cp_async_fence();

    cute::cp_async_wait<0>();
    __syncthreads();

    cute::gemm(A_shared_local, B_shared_local, C_reg);
    __syncthreads();
  }

  cute::copy(C_reg, C_thread);
}

void cute01(uintptr_t a, uintptr_t b, uintptr_t c, int m, int n, int k) {
  int blockSize = 128;
  dim3 gridSize((m + blockSize - 1) / blockSize,
                (n + blockSize - 1) / blockSize);

  if (m == 4096 && n == 4096 && k == 4096) {
    auto m = cute::_4096{};
    auto n = cute::_4096{};
    auto k = cute::_4096{};

    cute_matmul_kernel01<<<gridSize, 256>>>((float *)a, (float *)b, (float *)c,
                                            m, n, k);

  } else {
    cute_matmul_kernel01<<<gridSize, 256>>>((float *)a, (float *)b, (float *)c,
                                            m, n, k);
  }
}