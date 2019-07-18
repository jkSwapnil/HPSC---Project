#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>

using namespace std;

#define BLOCK_SIZE 30

__global__ void nodiag_normalize(float *A, float *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
	if (x == i && x!=y){
		I[x*n + y] /= A[i*n + i];
		A[x*n + y] /= A[i*n + i];
	}
	
}

__global__ void diag_normalize(float *A, float *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
	if (x == y && x == i){
		I[x*n + y] /= A[i*n + i];
		A[x*n + y] /= A[i*n + i];
	}

}

__global__ void gaussjordan(float *A, float *I, int n, int i)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			I[x*n + y] -= I[i*n + y] * A[x*n + i];
			if (y != i){
				A[x*n + y] -= A[i*n + y] * A[x*n + i];
			}	 
		}
	}

}

__global__ void set_zero(float *A, float *I, int n, int i){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n){
		if (x != i){
			if (y == i){
				A[x*n + y] = 0;
			}
		}
	}
}

int main(int argc, char const *argv[])
{
    int n = 1000;

    // allocate memory in host RAM, h_cc is used to store CPU result
    float *h_a, *h_ia, *I; // CPU
    float *d_a, *dI; // GPU
    int ddsize = n*n*sizeof(float);
    h_a = (float *)malloc(sizeof(float)*n*n);
    h_ia = (float *)malloc(sizeof(float)*n*n);
    I = (float *)malloc(sizeof(float)*n*n);

    // random initialize matrix A
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = i*j*0.01;
        }
    }

    //initialise unit matrix
    cudaMalloc((void**)&dI, ddsize);
    for (int i = 0; i<n; i++){
		for (int j = 0; j<n; j++){
			if (i == j) I[i*n + i] = 1.0;
			else I[i*n + j] = 0.0;
		}
	}

    cudaMalloc(&d_a, sizeof(float)*n*n);
    cudaMalloc(&I, sizeof(float)*n*n);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(int)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(dI, I, sizeof(int)*n*n, cudaMemcpyHostToDevice);
   
    // Launch kernel 
	dim3 numBlocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);  
    for (int i = 0; i<n; i++){
		nodiag_normalize << <numBlocks, threadsPerBlock >> >(d_a, dI, n, i);
		diag_normalize << <numBlocks, threadsPerBlock >> >(d_a, dI, n, i);
		gaussjordan << <numBlocks, threadsPerBlock >> >(d_a, dI, n, i);
		set_zero << <numBlocks, threadsPerBlock >> >(d_a, dI, n, i);
	}
    // Transefr results from device to host 
    cudaMemcpy(h_ia, dI, ddsize, cudaMemcpyDeviceToHost);
    //cudaMemcpy(I, d_a, ddsize, cudaMemcpyDeviceToHost);

    for(int i=0; i<n; i++){
        printf("\n");
        for(int j=0; j<n; j++){
            printf("\t%f",I[i*n + j]);

        }
    }
    // free memory
    cudaFree(d_a);
    cudaFree(dI);
    free(h_a);
    free(h_ia);
    free(I);
    return 0;
}