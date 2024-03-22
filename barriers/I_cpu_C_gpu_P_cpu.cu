// To compile and run:
// nvcc test.cu -arch=sm_80 -o test
// ./test

#include <cuda/atomic>
#include <cstdio>

using namespace cuda;

__global__ void consumer(atomic<int>* flag, int* data, int* result) {
    while (flag->load(memory_order_acquire) == 0) {}
    *result = *data;
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


        int device;
    cudaGetDevice(&device);

    struct cudaDeviceAttr attr;
    cudaDeviceGetAttribute(&attr, device);

    printf("\n Printing the attri of device (cudaDevAttrConcurrentManagedAccess) = %d", attr.cudaDevAttrConcurrentManagedAccess);
    ////////////////////////////////////////////////////////////////////////////

    // Launch the consumer asynchronously
    consumer<<<1,1>>>(flag, data, result);

    // Producer sequence
    if (data_in_unified_memory) {
        *data = 42;
    } else {
        int h_data = 42;
        SAFE(cudaMemcpy(data, &h_data, sizeof(int), cudaMemcpyHostToDevice));
    }

    // barrier comes here.
    // how many barriers ? Take from user code how many barriers to add.
    // Barrier or no barrier?

    flag->store(1, memory_order_release);

    // Wait for consumer to finish
    SAFE(cudaDeviceSynchronize());

    // Print the result
    printf("%d (expected 42)\n", *result);

    return 0;
}