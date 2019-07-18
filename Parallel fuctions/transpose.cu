#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 32

__global__ void gpu_matrix_transpose(float* mat_in, float* mat_out, unsigned int n, unsigned int m) 
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < m && row < n) 
    {
        unsigned int pos = row*m  + col;
        unsigned int trans_pos = col * n + row;
        mat_out[trans_pos] = mat_in[pos];
    }
}

int main(int argc, char const *argv[])
{
    int m, n;
    m = 10000, n = 10000;

    // allocate memory in host RAM, h_cc is used to store CPU result
    float *h_a, *h_b;
    h_a = (float *)malloc(sizeof(float)*m*n);
    h_b = (float *)malloc(sizeof(float)*n*m);;

    // random initialize matrix A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = i*j*0.01;
        }
    }

    // Allocate memory space on the device 
    float *d_a, *d_b;
    cudaMalloc(&d_a, sizeof(float)*m*n);
    cudaMalloc(&d_b, sizeof(float)*n*m);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
   
    // Launch kernel 
	dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_matrix_transpose<<<dimGrid, dimBlock>>>(d_a, d_b, m, n);    
    // Transefr results from device to host 
    cudaMemcpy(h_b, d_b, sizeof(int)*n*m, cudaMemcpyDeviceToHost);
    //cudaThreadSynchronize();

    for(int i=0; i<n; i++){
        printf("\n");
        for(int j=0; j<m; j++){
            printf("\t%f",h_b[i*m + j]);

        }
    }
    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
    return 0;
}