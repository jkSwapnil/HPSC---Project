#include<stdio.h>
#include<stdlib.h>
#include <time.h>
//#include <cuda.h>
#include <math.h>


void rand_dev(float* x, float* y, float* w_all, int k_value, int no_of_parameters, int n){

	float gradient[2]= {0};		
	double alpha[2] = {0.00000001,0.01};	//learning rate
	float fx, error =0.0;

	for(int i=0; i<5; ++i){
		alpha[0] = 0.0000001*(i+1);
		alpha[1] = 0.01*(i+1);
		if(i==4){
		alpha[0] = alpha[0]*(i+1)*10;
		alpha[1] = alpha[1]*(i+1)*10;
		}
		for(int batch_id=0; batch_id<10; batch_id++){
			for(int i=0; i<5000; i++){
				for(int k=0; k<n; k++){
					if (k< batch_id*(int)(n/k_value) || k >=((batch_id+1)*(int)(n/k_value))){
						fx=w_all[batch_id*no_of_parameters + 0]*x[k] + w_all[batch_id*no_of_parameters + 1];
						gradient[0]+= (fx-y[k])*x[k];
						gradient[1]+= (fx-y[k]);
					}
				}
				w_all[batch_id*no_of_parameters + 0] -= alpha[0]*(gradient[0]/900);
				w_all[batch_id*no_of_parameters + 1] -= alpha[1]*(gradient[1]/900);
				gradient[0] = gradient[1] = 0.0;
			}

			for(int k=batch_id*(int)(n/k_value); k<(batch_id+1)*(int)(n/k_value); k++){
				fx = w_all[batch_id*no_of_parameters + 0]*x[k] + w_all[batch_id*no_of_parameters + 1];
				error += (fx-y[k]);
			}
			printf("Learning rates: %.7f and %.3f---Weights of batch %d are: %.3f & %.3f---error is: %.3f\n", alpha[0], alpha[1], batch_id+1, w_all[batch_id*no_of_parameters + 0], w_all[batch_id*no_of_parameters + 1] , error);
		}
	}
}


int main(int argc, char *argv[]){

	clock_t t1, t2;
	int n=2000;// n is number of sample points
	int k_value =10, no_of_parameters=2;  
	float host_x[n], host_y[n]; 
	float host_w_all[k_value][no_of_parameters];
	for(int i=0; i<n; i++){
		host_x[i] = i;
		host_y[i] = host_x[i]*2 + 10; // m=2 and c=10
		host_y[i] += (float)rand()/RAND_MAX;
	}

	for(int i=0; i<k_value; i++)
		host_w_all[i][0]=1.5;

	t1=clock();

	rand_dev(host_x, host_y, host_w_all, k_value, no_of_parameters, n);

	t2=clock();

	printf("#................................................................#");
	printf("\n Time taken for multiplication is %1fsec\n",(t2-t1)/(double) CLOCKS_PER_SEC);
	printf("#................................................................#");
}