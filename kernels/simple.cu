#include <cstdint>

__global__ void simple_matmul_kernel(float *a, float *b, float *c, int m, int n,
                                     int k) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m && col < n) {
    float value = 0.0f;
    for (int i = 0; i < k; ++i) {
      value += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = value;
  }
}

void simple(uintptr_t a, uintptr_t b, uintptr_t c, int m, int n, int k) {
  dim3 blockSize(16, 16);
  dim3 gridSize((m + blockSize.x - 1) / blockSize.x,
                (n + blockSize.y - 1) / blockSize.y);

  simple_matmul_kernel<<<gridSize, blockSize>>>((float *)a, (float *)b,
                                                (float *)c, m, n, k);
}