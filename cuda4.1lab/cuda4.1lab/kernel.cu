#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>
#include <math.h>
#include <vector>



__global__ void MonteKarloPi(int* niter, int* count, double* x, double* y)
{

    for (int i = 0; i < *niter; ++i)
    {

        //get random points


        double z = (x[i] * x[i]) + (y[i] * y[i]);

        //check to see if point is in unit circle
        if (z <= 1)
        {
            ++* count;
        }
    }


}



int main()
{


    const int niter = 10000;
    int i;
    int count = 0;
    double pi;
    int* dev_niter, * dev_count;
    int size = sizeof(int);
    double x[niter], * dev_x;
    double y[niter], * dev_y;
    const size_t x_size = sizeof(double) * size_t(niter);

    srand(time(NULL));

    for (int i = 0; i < niter; i++)
    {
        x[i] = (double)rand() / RAND_MAX;
        y[i] = (double)rand() / RAND_MAX;

    }

    cudaMalloc((void**)&dev_niter, size);
    cudaMalloc((void**)&dev_count, size);

    cudaMalloc((void**)&dev_x, x_size);
    cudaMalloc((void**)&dev_y, x_size);

    cudaMemcpy(dev_niter, &niter, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_count, &count, size, cudaMemcpyHostToDevice);

    cudaMemcpy(dev_x, x, x_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, x_size, cudaMemcpyHostToDevice);

    MonteKarloPi << < 1, 2 >> > (dev_niter, dev_count, dev_x, dev_y);

    cudaMemcpy(&count, dev_count, size, cudaMemcpyDeviceToHost);



    pi = ((double)count / (double)niter) * 4.0;
    printf("Pi: %f\n", pi);


    return 0;
}
