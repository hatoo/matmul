#include <cstdint>

const int TILE_SIZE = 16;

__global__ void tile_matmul_kernel(float *a, float *b, float *c, int m, int n,
                                   int k) {

  // Shared memory for tiles
  __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

  // Thread indices
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Global thread indices
  int row = blockIdx.y * TILE_SIZE + ty;
  int col = blockIdx.x * TILE_SIZE + tx;

  float sum = 0.0f;

  // Loop over tiles
  for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
    // Load tile from matrix A
    int a_row = row;
    int a_col = tile * TILE_SIZE + tx;
    if (a_row < m && a_col < k) {
      tile_a[ty][tx] = a[a_row * k + a_col];
    } else {
      tile_a[ty][tx] = 0.0f;
    }

    // Load tile from matrix B
    int b_row = tile * TILE_SIZE + ty;
    int b_col = col;
    if (b_row < k && b_col < n) {
      tile_b[ty][tx] = b[b_row * n + b_col];
    } else {
      tile_b[ty][tx] = 0.0f;
    }

    // Synchronize threads to ensure tiles are loaded
    __syncthreads();

    // Compute partial dot product for this tile
    for (int i = 0; i < TILE_SIZE; ++i) {
      sum += tile_a[ty][i] * tile_b[i][tx];
    }

    // Synchronize before loading next tile
    __syncthreads();
  }

  // Write result to global memory
  if (row < m && col < n) {
    c[row * n + col] = sum;
  }
}

void tile(uintptr_t a, uintptr_t b, uintptr_t c, int m, int n, int k) {
  dim3 blockSize(TILE_SIZE, TILE_SIZE);
  dim3 gridSize((m + blockSize.x - 1) / blockSize.x,
                (n + blockSize.y - 1) / blockSize.y);

  tile_matmul_kernel<<<gridSize, blockSize>>>((float *)a, (float *)b,
                                              (float *)c, m, n, k);
}