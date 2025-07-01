#include "cute/tensor.hpp"
#include <cstdint>

template <class M, class N, class K>
__global__ void cute_matmul_kernel02(float *a, float *b, float *c, M m, N n,
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

  cute::TiledCopy copy_a = cute::make_tiled_copy(
      cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, float>{},
      cute::make_layout(cute::make_shape(cute::_128{}, cute::_2{})),
      cute::make_layout(cute::make_shape(cute::_1{}, cute::_4{})));
  cute::ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  cute::Tensor A_block_copy = thr_copy_a.partition_S(A_block);

  cute::TiledCopy copy_b = cute::make_tiled_copy(
      cute::Copy_Atom<cute::UniversalCopy<cute::uint128_t>, float>{},
      cute::make_layout(cute::make_shape(cute::_32{}, cute::_8{})),
      cute::make_layout(cute::make_shape(cute::_4{}, cute::_1{})));
  cute::ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  cute::Tensor B_block_copy = thr_copy_b.partition_S(B_block);

  __shared__ float a_shared[128 * 8];
  __shared__ float b_shared[128 * 8];

  cute::Tensor A_shared = cute::make_tensor(
      cute::make_smem_ptr(a_shared), cute::make_shape(cute::_128{}, cute::_8{}),
      cute::make_stride(cute::_8{}, cute::_1{}));
  cute::Tensor B_shared =
      cute::make_tensor(cute::make_smem_ptr(b_shared),
                        cute::make_shape(cute::_128{}, cute::_8{}));

  cute::Tensor A_shared_copy = thr_copy_a.partition_D(A_shared);
  cute::Tensor B_shared_copy = thr_copy_b.partition_D(B_shared);

  cute::TiledMMA mma = cute::make_tiled_mma(
      cute::UniversalFMA<float>{}, cute::make_layout(cute::make_shape(
                                       cute::_16{}, cute::_16{}, cute::_1{})));

  cute::ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  cute::Tensor A_shared_local = thr_mma.partition_A(A_shared);
  cute::Tensor B_shared_local = thr_mma.partition_B(B_shared);
  cute::Tensor C_thread_local = thr_mma.partition_C(C_block);

  cute::Tensor C_reg = cute::make_fragment_like(C_thread_local);
  cute::clear(C_reg);

  for (int i = 0; i < k / 8; ++i) {
    cute::copy(copy_a, A_block_copy(cute::_, cute::_, cute::_, i),
               A_shared_copy);
    cute::copy(copy_b, B_block_copy(cute::_, cute::_, cute::_, i),
               B_shared_copy);

    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    cute::gemm(mma, A_shared_local, B_shared_local, C_reg);
    __syncthreads();
  }

  cute::copy(C_reg, C_thread_local);
}

void cute02(uintptr_t a, uintptr_t b, uintptr_t c, int m, int n, int k) {
  int blockSize = 128;
  dim3 gridSize((m + blockSize - 1) / blockSize,
                (n + blockSize - 1) / blockSize);

  if (m == 4096 && n == 4096 && k == 4096) {
    auto m = cute::_4096{};
    auto n = cute::_4096{};
    auto k = cute::_4096{};

    cute_matmul_kernel02<<<gridSize, 256>>>((float *)a, (float *)b, (float *)c,
                                            m, n, k);

  } else {
    cute_matmul_kernel02<<<gridSize, 256>>>((float *)a, (float *)b, (float *)c,
                                            m, n, k);
  }
}