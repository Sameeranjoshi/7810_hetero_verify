// WRITE INTO SCRATCHPAD
// LOOP FOR EACH THREAD
// <<<1,1024>>>
// STRESSING VS TESTING THREADS
// PREFETCH ON GPU AND CPU
//

#include <cuda/atomic>
#include <cstdio>

using namespace cuda;
#include <stdio.h>

// Kernel function to access data on GPU by two threads
__global__ void accessData(atomic<int>* d_flag, int *d_data, int *d_result, int *d_buffer) {
    // int threadId = threadIdx.x;
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    // __shared__ int buffer1[1024];

    if (threadId == 0) {    // t0 = writer
        for (int i=0; i< 1000; i++){
            *d_data = 42;
            d_flag->store(1, memory_order_relaxed);
        }
    } 
    else if (threadId == 1) { // t1 = reader
        for (int i =1000; i!=0; i--){
            d_result[0] = d_flag->load(memory_order_relaxed);
            d_result[1] = *d_data;
        }
    }   
    else{   // stressing threads
        for(int i=0; i< 1000; i++){
            d_buffer[threadId] = *d_data;
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
    int *d_buffer;
    int h_result[2] = {99,99};
    int* d_result;


    // Allocate memory on GPU
    cudaMalloc(&d_flag,  sizeof(atomic<int>));
    cudaMalloc(&d_data,  sizeof(int));
    cudaMalloc(&d_buffer,  1024*sizeof(int));
    cudaMalloc(&d_result, 2*sizeof(int));
    
    // Copy data from host to device
    cudaMemcpy(d_flag, &h_flag, sizeof(atomic<int>), cudaMemcpyHostToDevice);   // init
    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);   // init
    cudaMemcpy(d_result, &h_result, 2*sizeof(int), cudaMemcpyHostToDevice);   // init

   
        cudaMemPrefetchAsync(d_flag, sizeof(atomic<int>), 0, NULL);         
        cudaMemPrefetchAsync(d_data, sizeof(int), 0, NULL); 
 
    accessData<<<1, 1024>>>(d_flag, d_data, d_result, d_buffer);

        
        

    // Synchronize to ensure kernel finishes before accessing data
    cudaDeviceSynchronize();

        cudaMemPrefetchAsync(d_flag, sizeof(atomic<int>), cudaCpuDeviceId, NULL); 
        cudaMemPrefetchAsync(d_data, sizeof(int), cudaCpuDeviceId, NULL); 
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
    cudaFree(d_buffer);

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