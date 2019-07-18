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
	double alpha[2];	//learning rate
	float fx, error =0.0;

	w_all[myid*no_of_parameters + 0] = 1;
	w_all[myid*no_of_parameters + 1] =-1; 

	alpha[0] = 0.035+0.01*(float)block_id;
	alpha[1] = 0.035+0.01*(float)block_id;

	//Train Loop
	for(int i=0; i<7000; i++){
	 	for(int k=0; k<n; k++){
			if (k< myid*(int)(n/k_value) || k >=((myid+1)*(int)(n/k_value))){
			fx=w_all[myid*no_of_parameters + 0]*x[k] + w_all[myid*no_of_parameters + 1];
			fx= 1/(1+exp(-fx));
			gradient[0]+= (y[k]-fx)*fx*(1-fx)*x[k];
			gradient[1]+= (y[k]-fx)*fx*(1-fx);
			}

		}
	w_all[myid*no_of_parameters + 0] += alpha[0]*(gradient[0]/900);
	w_all[myid*no_of_parameters + 1] += alpha[1]*(gradient[1]/900);
	gradient[0] = gradient[1] = 0.0;
	}

	//Testing loop
	for(int k=myid*(int)(n/k_value); k<(myid+1)*(int)(n/k_value); k++){
		fx = w_all[myid*no_of_parameters + 0]*x[k] + w_all[myid*no_of_parameters + 1];
		fx= 1/(1+exp(-fx));			
		error += (fx-y[k]);
	}
	printf("Learning rates: %.3f and %.3f---Weights of batch %d are: %.3f & %.3f---error is: %.3f\n", alpha[0], alpha[1], myid+1, w_all[myid*no_of_parameters + 0], w_all[myid*no_of_parameters + 1] , error);
}


int main(int argc, char *argv[]){
	int n=1000;// n is number of sample points
	int k_value =10, no_of_parameters=2;  
	float host_x[n], host_y[n], host_w_all[k_value][no_of_parameters]={0}, *x, *y, *w_all;
	float size = n * sizeof(float);	
	float size1 = k_value*no_of_parameters*sizeof(float);
	float a=0;

	//Data Generation with noise
	for(int i=0; i<n; i++){
		host_x[i] = a;
		a = a+0.05;
		float h = 0.18*host_x[i] -5.5; // m = 0.18 and c = -5.5
		host_y[i]=round(1/(1+exp(-h))); 
	}

	//Initializing the weights
	for(int i=0; i<k_value; i++){
		host_w_all[i][0]=1;
		host_w_all[i][1]=-1;
	}

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