
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define N 1000
#include <cmath>


__global__ void kernel(double* a)
{
    int i = threadIdx.x;
    a[i] = std::sqrtf(1.0 - double(i) * double(i) / double(N) / double(N));
}

int main()
{
    double a[N] = { 0 };
    double* dev_a;

    cudaError_t err= cudaGetLastError();

    cudaMalloc((void**)&dev_a, N * sizeof(double));


    kernel << <2, N >> > (dev_a);

    cudaMemcpy(a, dev_a, N * sizeof(double), cudaMemcpyDeviceToHost);


    if (err != cudaSuccess) printf("%s ",
        cudaGetErrorString(err));

    double q = 0;
    for (int i = 0; i < N; ++i) {
        q += a[i];
    }

    printf("Pi is %f\n", q * 4 / N);

    cudaFree(dev_a);
    return 0;
}