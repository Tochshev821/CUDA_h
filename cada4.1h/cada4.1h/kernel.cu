#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>

#define N 1000


__device__ bool IsApropriate(double* x, double* y)
{
    bool result = (*x) * (*x) + (*y) * (*y) <= 1;
    return result;
}

__global__ void CalculatePI(int * dev_a)
{
    double x = (double)blockIdx.x / N;
    double y = (double)threadIdx.x / N;
    IsApropriate(&x, &y) ? atomicAdd(dev_a, 1) : 0;
}

int main()
{
    int di = N * N;
    int a = 0;
    int* dev_a;

    cudaMalloc((void**)&dev_a, sizeof(int));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("%s ",
        cudaGetErrorString(err));
    cudaMemcpy(dev_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("%s ",
        cudaGetErrorString(err));
    CalculatePI << < N,  N>> > (dev_a);

    cudaMemcpy(&a, dev_a, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) printf("%s ",
        cudaGetErrorString(err));
    printf("pi = %f\n", (double)a * 4 / di);

    cudaFree(dev_a);
    return 0;
}