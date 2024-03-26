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

#define SAFE(x) if (0 != x) { abort(); }

int main(int argc, char* argv[]) {
    atomic<int>* flag;
    int N = 10;
    int* data;
    int* result0, *result1;

    int data_in_unified_memory = 1;

    ////////////////////////////////////////////////////////////////////////////

    // Flag in unified memory
    SAFE(cudaMallocManaged(&flag, sizeof(atomic<int>)));

    // Data placed as specified
    if (data_in_unified_memory) {
        SAFE(cudaMallocManaged(&data, N*sizeof(int)));
    } else {
        SAFE(cudaMalloc(&data, N*sizeof(int)));
    }

    // Result array pinned in CPU memory
    SAFE(cudaMallocHost(&result0, N*sizeof(int)));
    SAFE(cudaMallocHost(&result1, N*sizeof(int)));

    // Initial values: data = <unknown>, flag = 0
    flag->store(0, memory_order_relaxed);
    data[0] = 0;
    data[1] = 0;
    data[2] = 0;
    data[3] = 0;
    data[4] = 0;

    ////////////////////////////////////////////////////////////////////////////

    // Launch the consumer asynchronously
    consumer<<<1,1>>>(flag, data, result0, result1);
    
    // Producer sequence
    if (data_in_unified_memory) {
        data[0] = 42;
        data[1] = 43;
        data[2] = 44;
        data[3] = 45;
        data[4] = 46;

    } else {
        int h_data[N];
        int k = 42;
        for (int i=0; i< N; i++){
            h_data[i] = k;  // 42, 43, 44, 45, 46
            k++;
        }

        SAFE(cudaMemcpy(data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

        // SAFE(cudaMemcpy(data[0], &h_data, sizeof(int), cudaMemcpyHostToDevice));
        // SAFE(cudaMemcpy(data[1], &h_data1, sizeof(int), cudaMemcpyHostToDevice));
        // SAFE(cudaMemcpy(data[2], &h_data2, sizeof(int), cudaMemcpyHostToDevice));
        // SAFE(cudaMemcpy(data[3], &h_data3, sizeof(int), cudaMemcpyHostToDevice));
        // SAFE(cudaMemcpy(data[4], &h_data4, sizeof(int), cudaMemcpyHostToDevice));

    }
    flag->store(1, memory_order_relaxed);


    // Wait for consumer to finish
    SAFE(cudaDeviceSynchronize());

    // Print the result
    // printf("data = %d (expected 42) flag = %d \n", *data, flag->load(memory_order_acquire));
    for (int i=0; i< N; i++)
        printf("I = %d: data = %d flag = %d \n", i, result0[i], result1[i]);

    // missing to free the device data
    
    return 0;
}