#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include <stdio.h>
#define N 1000 

__global__ void ZFunction(float* a, float* b)
{
    int i = threadIdx.x; 
    a[i] = 1.f / powf(float(i + 1), *b);
}


int main()
{
    float b = 2; //степень
    float a[N]; //массив членов ряда
    float* dev_b = 0;
    float* dev_a = 0;
    float sum = 0; //частная сумма ряда

    cudaMalloc((void**)&dev_a, N * sizeof(float));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("%s ",
        cudaGetErrorString(err));
    cudaMalloc((void**)&dev_b, sizeof(float));
    if (err != cudaSuccess) printf("%s ",
        cudaGetErrorString(err));
    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("%s ",
        cudaGetErrorString(err));
    cudaMemcpy(dev_b, &b, sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) printf("%s ",
        cudaGetErrorString(err));
    ZFunction << <1, N >> > (dev_a, dev_b);

    cudaMemcpy(a, dev_a, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) printf("%s ",
        cudaGetErrorString(err));
    for (int i = 0; i < N; ++i)
    {
        sum += a[i];
    }
    printf("%f\n", sum);

    cudaFree(dev_a);
    cudaFree(dev_b);
    return 0;
}