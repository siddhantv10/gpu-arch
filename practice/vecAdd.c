%%cu

#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>

__global__ void kernel_vecAdd (float* A, float*B, float*C, int n){
    int i = threadIdx.x + blockDim.x*blockIdx.x;

    if(i<n)
        C[i] = A[i] + B[i];
}

void host_vecADD (float* h_A, float* h_B, float* h_C, int n){
    int size = n*sizeof(float);

    float* d_A = NULL;
    float* d_B = NULL;
    float* d_C = NULL;

    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void**) &d_A, size);
    err = cudaMalloc((void**) &d_B, size);
    cudaMalloc((void**) &d_C, size);


    printf("copying host data to kernel\n");

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);


    //kernel call

    int threadsPerBlock = 256;
    int blocksPerGrid = 1 + (n-1)/threadsPerBlock;

    kernel_vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C,n);

    err = cudaGetLastError();

    if(err != cudaSuccess){
        fprintf(stderr, "failed to launch kernel (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("copy output data from output dev to host\n");

    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    //verify

    for(int i=0; i<n; i++){
        if(fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5){
            fprintf(stderr, "result verification failed at elemetn %d\n", i);
            exit(EXIT_FAILURE);
        }
        printf("Sucess\n");
    }
}


int main(){
    int n = 4990;
    float A[n], B[n], C[n];

    for(int i=0; i<n; i++){
        A[i] = B[i] = 2;
    }

    host_vecADD(A,B,C,n);
}