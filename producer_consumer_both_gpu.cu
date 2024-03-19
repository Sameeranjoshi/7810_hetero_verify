// To compile and run:
// nvcc test.cu -arch=sm_80 -o test
// ./test

// Total 3 threads 
// Cpu, GPU-thread-1 , GPU-thread-2
// Intialization thread - CPU
// Producer - GPU-thread-1
// Consumer - GPU-thread-2

#include <cuda/atomic>
#include <cstdio>

using namespace cuda;

__global__ void consumer_gpu(atomic<int>* flag, int* data, int* result) {
    while (flag->load(memory_order_acquire) == 0) {}
    *result = *data;
}

__global__ void producer_gpu(atomic<int> *flag, int* data, int * result){
    // Producer sequences
    *data = 42;
    flag->store(1, memory_order_release);
}

#define SAFE(x) if (0 != x) { abort(); }

int main(int argc, char* argv[]) {
    atomic<int>* flag;
    int* data;
    int* result;

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
    SAFE(cudaMallocHost(&result, sizeof(int)));

    // Initial values: data = <unknown>, flag = 0
    flag->store(0, memory_order_relaxed);

    ////////////////////////////////////////////////////////////////////////////

    // Transfer data is not in unified memory
    // Can comment if necessary
    if (!data_in_unified_memory){   // data is not in unified memory
        int h_data = 42;
        SAFE(cudaMemcpy(data, &h_data, sizeof(int), cudaMemcpyHostToDevice));
    }
    producer_gpu<<<1,1>>>(flag, data, result);

    // Launch the consumer asynchronously
    consumer_gpu<<<1,1>>>(flag, data, result);


    // Wait for consumer to finish
    SAFE(cudaDeviceSynchronize());

    // Print the result
    printf("%d (expected 42)\n", *result);

    return 0;
}