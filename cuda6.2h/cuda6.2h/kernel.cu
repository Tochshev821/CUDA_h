#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#define N 10


__global__ void matrixAdd(const int* A, const
    int* B, int* C)
{
    int i = blockIdx.x;
    int j = threadIdx.x;
    C[i * N + j] = A[i * N + j] + B[i * N + j];
}

int main()
{
    int a[N][N];
    int b[N][N];
    int c[N][N];
    int* ca;
    int* cb;
    int* cc;
    for (int i = 0; i < N * N; ++i) {
        *(*a + i) = 3;
    }

    for (int i = 0; i < N * N; ++i) {
        *(*b + i) = 3;
    }
    cudaMalloc((void**)&ca, N * N * sizeof(int));
    cudaMalloc((void**)&cb, N * N * sizeof(int));
    cudaMalloc((void**)&cc, N * N * sizeof(int));
    cudaMemcpy(ca, &a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cb, &b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    matrixAdd << <N, N >> > (ca, cb, cc);
    cudaMemcpy(&c, cc, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    cudaFree(ca);
    cudaFree(cb);
    cudaFree(cc);
    return 0;
}