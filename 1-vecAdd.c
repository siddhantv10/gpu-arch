%%cu
#include<stdio.h>
#include<stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

//device kernels never return a value - always global void

__global__ void vectorAdd(float* A, float*B, float*C, int n){   //CUDA kernel definition
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i<n)
        C[i] = A[i] + B[i];
}

void vecAdd(float*h_A, float*h_B, float*h_C, int n){            //host program
    
    int size = n*sizeof(float);     //memory size of vector to be added

    float*d_A = NULL, *d_B=NULL, *d_C = NULL;

    //Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;


    //allocate device(GPU) memory to vectors
    err = cudaMalloc((void**)&d_A, size);
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&d_B, size);
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }    

    err = cudaMalloc((void**)&d_C, size);
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //copy vector values from host to device
    printf("Copying input data from the host memory to the CUDA device\n");

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector C from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    //kernel call
    
    int threadsPerBlock = 256;
    int blocksPerGrid = 1 + (n-1)/threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    err = cudaGetLastError();
    //device function / CUDA Kernel called from host does not have a return type
    //CUDA runtime functions which execute in host side can have return type

    if(err != cudaSuccess){
        fprintf(stderr, "Failer to launch vectorAdd kernel (error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy output data from the output device to the host memory\n");
    
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector C from device to host(error code %s)\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    //verify
    for(int i =0; i<n; i++){
        if(fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5){
            fprintf(stderr, "Result verification failed at element %d\n", i);
            exit(EXIT_FAILURE);
        }
        printf("Test PASSED\n");
    }

}

int main(){
    int n = 4900;
    float a[n], b[n], c[n];
    
    printf("[vector addition of %d elements\n]", n);
    for(int i =0; i<n; i++){
        a[i] = b[i] = 1;
    }

    vecAdd(a,b,c,n);

    
}