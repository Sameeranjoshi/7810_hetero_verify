// To compile and run:
// nvcc test.cu -arch=sm_80 -o test
// ./test

#include <cuda/atomic>
#include <cstdio>

using namespace cuda;
#include <stdio.h>

// Kernel function to access data on GPU by two threads
__global__ void accessData(atomic<int>* d_flag, int *d_data, int *d_result) {
    // int threadId = threadIdx.x;
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ int buffer1[1024];

    if (threadId == 0) {    // t0 = writer
        *d_data = 42;
        d_flag->store(1, memory_order_relaxed);
    } 
    if (threadId == 1) { // t1 = reader
        d_result[0] = d_flag->load(memory_order_relaxed);
        d_result[1] = *d_data;
    }
    else{   // stressing threads
        for(int i=0; i< 10000; i++){
            buffer1[threadId] = *d_data;
        }
    }

}

struct Result{
int seq1, seq2, interleave, weak;
};



void run(Result *count_local){
    
    atomic<int> h_flag=0;
    int h_data=0;
    atomic<int>* d_flag;
    int *d_data;
    int h_result[2] = {99,99};
    int* d_result;


    // Allocate memory on GPU
    cudaMalloc(&d_flag,  sizeof(atomic<int>));
    cudaMalloc(&d_data,  sizeof(int));
    cudaMalloc(&d_result, 2*sizeof(int));
    
    // Copy data from host to device
    cudaMemcpy(d_flag, &h_flag, sizeof(atomic<int>), cudaMemcpyHostToDevice);   // init
    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);   // init
    cudaMemcpy(d_result, &h_result, 2*sizeof(int), cudaMemcpyHostToDevice);   // init
    
    accessData<<<1, 1024>>>(d_flag, d_data, d_result);

    // Synchronize to ensure kernel finishes before accessing data
    cudaDeviceSynchronize();

    // Copy data back from device to host
    cudaMemcpy(h_result, d_result, 2*sizeof(int), cudaMemcpyDeviceToHost);

    // // Print modified data
    // printf("flag: %d, data %d\n", h_result[0], h_result[1]);

    if (h_result[0]== 99 && h_result[1] == 99){
        printf("\n Bug in CUDA implementation! exiting");
        exit(1);
    }
     //r0=flag, r1=data
    if (h_result[0] == 0 && h_result[1] == 0){
        count_local->seq1 += 1 ;  //# t1->t2
    }
    else if(h_result[0] == 1 && h_result[1] == 42){
        count_local->seq2 += 1 ;  //# t2-t1
    }
    else if(h_result[0] == 0 && h_result[1] == 42){
        count_local->interleave += 1;
    }
    else if(h_result[0] == 1 && h_result[1] == 0){
        count_local->weak += 1;
    }

    // Free device memory
    cudaFree(d_flag);
    cudaFree(d_data);

}


int main(int argc, char* argv[]) {
        if (argc !=2 ){
            printf("\n ./a.out <number of tests>");
            exit(1);
        }
     int loop_size = atoi(argv[1]);
     Result count_local{0};

    for (int i=0; i< loop_size; i++){
        // printf("i=%d\n", i);
        run(&count_local);
        if (i == loop_size/4){
            printf("\n 25%%");
        } else if (i == loop_size/2){
            printf("\n 50%%");
        }
    }

    printf("\n Histogram after %d runs\n", loop_size);
    printf("seq1 (flag)=0; (data)=0;  = %d\n", count_local.seq1);
    printf("seq2 (flag)=1; (data)=42; = %d\n", count_local.seq2);
    printf("intlv(flag)=0; (data)=42; = %d\n", count_local.interleave);
    printf("weak (flag)=1; (data)=0;  = %d\n", count_local.weak);

    return 0;
}