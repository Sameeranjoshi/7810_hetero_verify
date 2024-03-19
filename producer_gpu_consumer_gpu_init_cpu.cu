// To compile and run:
// nvcc test.cu -arch=sm_80 -o test
// ./test

// Total 3 threads 
// Cpu, GPU-thread-1 , GPU-thread-2
// Intialization thread - CPU
// Producer - GPU-thread-1
// Consumer - GPU-thread-2

// much cleaner removed the non in unified memory checks.
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

    ////////////////////////////////////////////////////////////////////////////

    // Flag in unified memory
    SAFE(cudaMallocManaged(&flag, sizeof(atomic<int>)));
    SAFE(cudaMallocManaged(&data, sizeof(int)));
    // Result array pinned in CPU memory
    SAFE(cudaMallocHost(&result, sizeof(int)));

    // Initial values: data = <unknown>, flag = 0
    flag->store(0, memory_order_relaxed);

    ////////////////////////////////////////////////////////////////////////////
    producer_gpu<<<1,1>>>(flag, data, result);
    // Launch the consumer asynchronously
    consumer_gpu<<<1,1>>>(flag, data, result);
    // Wait for consumer to finish
    SAFE(cudaDeviceSynchronize());
    // Print the result
    printf("%d (expected 42)\n", *result);

    return 0;
}