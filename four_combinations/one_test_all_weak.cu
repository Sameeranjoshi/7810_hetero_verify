// To compile and run:
// nvcc test.cu -arch=sm_80 -o test
// ./test

#include <cuda/atomic>
#include <cstdio>

using namespace cuda;

__global__ void consumer(atomic<int>* flag, int* data, int* result0, int*result1) {
    // while (flag->load(memory_order_acquire) == 0) {}
    *result1 = flag->load(memory_order_relaxed);
    *result0 = *data;
}

__global__ void consumer_caching_dummy(atomic<int>* flag, int* data, int* result0, int*result1) {
    // cahce in GPU.
    flag->load(memory_order_relaxed);
    // cache in results0 first.
    *result0 = *data;
}

#define SAFE(x) if (0 != x) { abort(); }

int main(int argc, char* argv[]) {
    atomic<int>* flag;
    int* data;
    int* result0, *result1;

    int data_in_unified_memory = 1;

    ////////////////////////////////////////////////////////////////////////////

    // Flag in unified memory
    SAFE(cudaMallocManaged(&flag, sizeof(atomic<int>)));

    // Data placed as specified
    if (data_in_unified_memory) {
        SAFE(cudaMallocManaged(&data, sizeof(int)));
    } else {
        SAFE(cudaMalloc(&data, sizeof(int)));
    }

    // Result array pinned in CPU memory
    SAFE(cudaMallocHost(&result0, sizeof(int)));
    SAFE(cudaMallocHost(&result1, sizeof(int)));

    // Initial values: data = <unknown>, flag = 0
    flag->store(0, memory_order_relaxed);
    *data = 0;

    ////////////////////////////////////////////////////////////////////////////

    // Add a caching on GPU 
    consumer_caching_dummy<<<1,1>>>(flag, data, result0, result1);
    SAFE(cudaDeviceSynchronize());
    consumer<<<1,1>>>(flag, data, result0, result1);
    
    // Producer sequence
    if (data_in_unified_memory) {
        *data = 42;
    } else {   
        int h_data = 42;
        SAFE(cudaMemcpy(data, &h_data, sizeof(int), cudaMemcpyHostToDevice));
    }
    flag->store(1, memory_order_relaxed);


    // Wait for consumer to finish
    SAFE(cudaDeviceSynchronize());

    // Print the result
    // printf("data = %d (expected 42) flag = %d \n", *data, flag->load(memory_order_acquire));
    printf("data = %d flag = %d (not expected flag=1 and data=0)\n", *result0, *result1);

    return 0;
}