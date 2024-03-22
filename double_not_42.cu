// To compile and run:
// nvcc test.cu -arch=sm_80 -o test
// ./test

#include <cuda/atomic>
#include <cstdio>

using namespace cuda;

__global__ void consumer(atomic<double>* flag, double* data, double* result) {
    while (flag->load(memory_order_acquire) == 0.0) {}
    *result = *data;
}

#define SAFE(x) if (0 != x) { abort(); }

int main(int argc, char* argv[]) {
    atomic<double>* flag;
    double* data;
    double* result;

    int data_in_unified_memory = 1;

    ////////////////////////////////////////////////////////////////////////////

    // Flag in unified memory
    SAFE(cudaMallocManaged(&flag, sizeof(atomic<double>)));

    // Data placed as specified
    if (data_in_unified_memory) {
        SAFE(cudaMallocManaged(&data, sizeof(double)));
    } else {
        SAFE(cudaMalloc(&data, sizeof(double)));
    }

    // Result array pinned in CPU memory
    SAFE(cudaMallocHost(&result, sizeof(double)));

    // Initial values: data = <unknown>, flag = 0
    flag->store(0.0, memory_order_relaxed);

    ////////////////////////////////////////////////////////////////////////////

    // Launch the consumer asynchronously
    consumer<<<1,1>>>(flag, data, result);

    // Producer sequence
    if (data_in_unified_memory) {
        *data = 42.99;
    } else {
        double h_data = 42.99;
        SAFE(cudaMemcpy(data, &h_data, sizeof(double), cudaMemcpyHostToDevice));
    }
    flag->store(1.99, memory_order_release);

    // Wait for consumer to finish
    SAFE(cudaDeviceSynchronize());

    // Print the result
    double epsilon = 1e-5;
    double delta = fabs(42.99 - (*result));

    // printf("\n What's the delta = %.6lf and epsilon = %.6lf", delta, epsilon);

    if (fabs(delta) < epsilon){
        printf("As expected, delta = %lf\n", delta);
    } else {
        printf("Not matching, delta = %lf", delta);
    }

    return 0;
}
