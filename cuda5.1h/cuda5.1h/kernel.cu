#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cmath"
#include <stdio.h>

#define N 5 

__global__ void ProdD(double* a, double* b, double* c)
{
	int i = threadIdx.x; 
	if (i > N - 1) return; 	
	c[i] = __dmul_rn(a[i], b[i]);
}

__global__ void ProdF(float* a, float* b, float* c)
{
	int i = threadIdx.x; 
	if (i > N - 1) return; 
	c[i] = __fmul_rn(a[i], b[i]);
}

int main()
{
	cudaEvent_t start_f, stop_f, start_d, stop_d;
	cudaEventCreate(&start_f);
	cudaEventCreate(&stop_f);
	cudaEventCreate(&start_d);
	cudaEventCreate(&stop_d);

	float a_f[N] = { 1,2,3,4,5 }, b_f[N]= { 1,2,3,4,5 }, c_f[N];
	double a_d[N]= { 1,2,3,4,5 }, b_d[N]= { 1,2,3,4,5 }, c_d[N];

	float* dev_a_f, * dev_b_f, * dev_c_f;
	double* dev_a_d, * dev_b_d, * dev_c_d;



	cudaMalloc((void**)&dev_a_f, N * sizeof(float));
	cudaMalloc((void**)&dev_b_f, N * sizeof(float));
	cudaMalloc((void**)&dev_c_f, N * sizeof(float));
	cudaMalloc((void**)&dev_a_d, N * sizeof(double));
	cudaMalloc((void**)&dev_b_d, N * sizeof(double));
	cudaMalloc((void**)&dev_c_d, N * sizeof(double));

	cudaMemcpy(dev_a_f, a_f, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b_f, b_f, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_a_d, a_d, N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b_d, b_d, N * sizeof(double), cudaMemcpyHostToDevice);

	cudaEventRecord(start_f, 0); 
	ProdF << <1, N >> > (dev_a_f, dev_b_f, dev_c_f);
	cudaEventRecord(stop_f, 0); 
	cudaEventSynchronize(stop_f);

	float kernelTime_f;
	cudaEventElapsedTime(&kernelTime_f, start_f, stop_f);
	printf("Float kernel time = %f ms\n", kernelTime_f);

	cudaEventRecord(start_d, 0); 
	ProdD << <1, N >> > (dev_a_d, dev_b_d, dev_c_d);
	cudaEventRecord(stop_d, 0); 
	cudaEventSynchronize(stop_d);

	float kernelTime_d;
	cudaEventElapsedTime(&kernelTime_d, start_d, stop_d);
	printf("Double kernel time = %f ms\n", kernelTime_d);

	cudaMemcpy(c_f, dev_c_f, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(c_d, dev_c_d, N * sizeof(double), cudaMemcpyDeviceToHost);

	float prod_f = 0;
	double prod_d = 0;

	for (int i = 0; i < N; i++)
	{
		prod_f += c_f[i];
		prod_d += c_d[i];
	}

	printf("prod_f = %f\nprod_d = %f\n", prod_f, prod_d);

	cudaFree(dev_a_f);
	cudaFree(dev_b_f);
	cudaFree(dev_c_f);
	cudaFree(dev_a_d);
	cudaFree(dev_b_d);
	cudaFree(dev_c_d);
	return 0;
}