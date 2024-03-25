// To compile and run:
// nvcc test.cu -arch=sm_80 -o test
// ./test

#include <cuda/atomic>
#include <cstdio>

using namespace cuda;

__global__ void consumer(atomic<int>* flag, int* data, int* result0, int *result1) {
    *result1 = flag->load(memory_order_acquire);
    *result0 = *data;
}


#define SAFE(x) if (0 != x) { abort(); }

int main(int argc, char* argv[]) {
    atomic<int>* flag;
    int* data;
    int* result0;
    int* result1;

    // int data_in_unified_memory = 1;

    ////////////////////////////////////////////////////////////////////////////
    // Result array pinned in CPU memory
    SAFE(cudaMallocHost(&result0, sizeof(int)));
    SAFE(cudaMallocHost(&result1, sizeof(int)));

    // Flag in unified memory
    SAFE(cudaMallocManaged(&flag, sizeof(atomic<int>)));
    // Data placed as specified
    SAFE(cudaMallocManaged(&data, sizeof(int)));
    // Initial values: data = 0, flag = 0
    flag->store(0, memory_order_relaxed);
    data = 0;

    ////////////////////////////////////////////////////////////////////////////

    // Launch the consumer asynchronously
    consumer<<<1,1>>>(flag, data, result0, result1);

    // Producer sequence
    *data = 42;
    flag->store(1, memory_order_release);

    // Wait for consumer to finish
    SAFE(cudaDeviceSynchronize());

    // Print the result
    printf("R0 = %d, R1 = %d\n", *result0, *result1);

    return 0;
}