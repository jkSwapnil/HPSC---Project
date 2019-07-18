#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<fstream>
#include<vector>
#include<string>
#include<sstream>
#include<iterator>
#include<algorithm>
#include<iostream>

using namespace std; 
#define BLOCK_SIZE 32

__global__ void gpu_matrix_mult(float *a,float *b, float *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}


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

int main() 
{
    int m, n;
    m = 1000, n = 5;
    float *data;
    data = new float[m*n];
    ifstream fin("a.csv");
    string line, word;
    vector<string> row;
    int i = 0, flag=0;

    //***************************************Reading Data from input file*********************************//
    while(!fin.eof())
    {   

        line.clear();
        getline(fin,line);
        row.clear();
        stringstream s(line);
        while (getline(s, word, ',')) { 
            row.push_back(word); 
        }
        if(i==0 && flag ==0){
            flag = 1;
            continue;
        }
        for(int j=0; j<n; ++j){
            data[i*n + j] = stof(row[j]);
        }
        i+=1;
    } 

    fin.close();

    //*******************************Making of Input output matrix from data is complete**************************//
    float *x, *y;
    y = new float[m*1];
    x = new float[m*n];

    for(int i=0; i<m; i++){
        y[i] = data[i*n + 0];
        for(int j=0; j<n-1; j++){
            x[i*n + j] = data[i*n + j + 1];
        }
        x[i*n + n-1] = 1;
    } 

    // for(int i=0; i<m; i++){
    //     printf("\n");
    //     for(int j=0; j<n; j++){
    //         printf("\t%f",x[i*n + j]);
    //     }
    // } 

    // for(int i=0; i<m; i++){
    //     printf("\t%f",y[i]);
    // } 
    delete [] data;
    //*******************************Using Cuda Functions to calculate the parameters**************************//

    float *xt = new float[n*m];  // transpose of the x matrix
    float * d_x, *d_xt;
    cudaMalloc((void **) &d_x, sizeof(float)*m*n);
    cudaMalloc((void **) &d_xt, sizeof(float)*n*m);
    cudaMemcpy(d_x, x, sizeof(float)*m*n, cudaMemcpyHostToDevice);

    dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    gpu_matrix_transpose<<<dimGrid, dimBlock>>>(d_x, d_xt, m, n);

    cudaMemcpy(xt, d_xt, sizeof(float)*m*n, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_xt);

    float * xtx = new float[m*m]; // matrix is the multiplication of xt and x
    float * d_xtx;
    cudaMalloc((void **) &d_x, sizeof(float)*m*n);
    cudaMalloc((void **) &d_xt, sizeof(float)*n*m);
    cudaMalloc((void **) &d_xtx, sizeof(float)*m*m);
    cudaMemcpy(d_xt, xt, sizeof(float)*n*m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, sizeof(float)*m*n, cudaMemcpyHostToDevice);


    dim3 dimGrid1((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 dimBlock1(BLOCK_SIZE, BLOCK_SIZE);
    gpu_matrix_mult<<<dimGrid1, dimBlock1>>>(d_xt, d_x, d_xtx, m, n, m); 

    cudaMemcpy(xtx, d_xtx, sizeof(float)*m*m, cudaMemcpyDeviceToHost);

    //code to be written for inverting the xtx matrix


    return 0;
}