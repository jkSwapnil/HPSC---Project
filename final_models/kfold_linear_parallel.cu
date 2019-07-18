#include<stdio.h>
#include<stdlib.h>
#include <time.h>
#include <cuda.h>
#include <math.h>


// Kernel for GPU cuda 
__global__ void rand_dev(float* x, float* y, float* w_all, int k_value, int no_of_parameters, int n){
	int myid = threadIdx.x; 
	int block_id = blockIdx.x;
	float gradient[2]= {0};		
	double alpha[2] = {0.0000001,0.01};	//learning rate
	float fx, error =0.0;


	alpha[0] = 0.0000001*(block_id+1);
	alpha[1] = 0.01*(block_id+1);
	
	//Training Loop
	for(int epoch=0; epoch<5000; epoch++){
		for(int k=0; k<n; k++){
			if (k< myid*(int)(n/k_value) || k >=((myid+1)*(int)(n/k_value))){
				fx=w_all[myid*no_of_parameters + 0]*x[k] + w_all[myid*no_of_parameters + 1];
				gradient[0]+= (fx-y[k])*x[k];
				gradient[1]+= (fx-y[k]);
			}
		}
		w_all[myid*no_of_parameters + 0] -= alpha[0]*(gradient[0]/900);
		w_all[myid*no_of_parameters + 1] -= alpha[1]*(gradient[1]/900);
		gradient[0] = gradient[1] = 0.0;
	}

	//Testing Loop
	for(int k=myid*(int)(n/k_value); k<(myid+1)*(int)(n/k_value); k++){
		fx = w_all[myid*no_of_parameters + 0]*x[k] + w_all[myid*no_of_parameters + 1];
		error += (fx-y[k]);
	}
	printf("Learning rates: %.7f and %.3f---Weights of batch %d are: %.3f & %.3f---error is: %.3f\n", alpha[0], alpha[1], myid+1, w_all[myid*no_of_parameters + 0], w_all[myid*no_of_parameters + 1] , error);
}


int main(int argc, char *argv[]){
	int n=2000;// n is number of sample points
	int k_value =10, no_of_parameters=2;  
	float host_x[n], host_y[n], host_w_all[k_value][no_of_parameters]={0}, *x, *y, *w_all;
	float size = n * sizeof(float);	
	float size1 = k_value*no_of_parameters*sizeof(float);

	//Data Generation with noise
	for(int i=0; i<n; i++){
		host_x[i] = i;
		host_y[i] = host_x[i]*2 + 10; // m=2 and c=10
		host_y[i] += (float)rand()/RAND_MAX;
	}

	//Initializing the weights
	for(int i=0; i<k_value; i++)
		host_w_all[i][0]=1.5;

	// allocating memory on cuda device
	cudaMalloc(&x, size);
	cudaMemcpy(x, host_x, size, cudaMemcpyHostToDevice);
	cudaMalloc(&y, size);
	cudaMemcpy(y, host_y, size, cudaMemcpyHostToDevice);
	cudaMalloc(&w_all, size1);
	cudaMemcpy(w_all,host_w_all, size1, cudaMemcpyHostToDevice);		


	dim3   DimGrid(5,1);     
	dim3   DimBlock(k_value,1);   
	rand_dev<<< DimGrid,DimBlock >>>(x, y, w_all, k_value, no_of_parameters, n);

	cudaMemcpy(host_w_all, w_all, k_value, cudaMemcpyDeviceToHost);
	cudaFree(x);
	cudaFree(y);
	cudaFree(w_all);
}