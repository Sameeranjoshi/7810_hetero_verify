// To compile and run:
// nvcc test.cu -arch=sm_80 -o test
// ./test

#include <cuda/atomic>
#include <cstdio>

using namespace cuda;

__global__ void consumer(atomic<int>* flag, int* data, int* result0, int*result1) {
    // while (flag->load(memory_order_acquire) == 0) {}
    *result1 = flag->load(memory_order_relaxed);

        // Get the start time
    clock_t start = clock();

    // Busy-wait loop until the specified time has elapsed
    while ((clock() - start) * 1000 / CLOCKS_PER_SEC < 1000) {
        // Do nothing, just wait
    }

    *result0 = *data;
}

#define SAFE(x) if (0 != x) { abort(); }

void run(){

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

    // Launch the consumer asynchronously
    // consumer<<<1,1>>>(flag, data, result0, result1);
    
    // gpU
    consumer<<<5,1024>>>(flag, data, result0, result1); 

    
    
  //delay
    int i = 1000;
    while(i ==0){
        sleep(1);
    }

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
    printf("result0=%d result1=%d \n", *result0, *result1);

       // Free the allocated memory at the end
    SAFE(cudaFree(flag));
    if (data_in_unified_memory) {
        SAFE(cudaFree(data));
    } else {
        SAFE(cudaFree(data));
    }
    SAFE(cudaFreeHost(result0));
    SAFE(cudaFreeHost(result1));


}
int main(int argc, char* argv[]) {
    for (int i=0; i< 1000; i++)
        run();
    return 0;
}