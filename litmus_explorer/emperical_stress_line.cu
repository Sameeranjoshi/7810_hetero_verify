// Stress Cache line - 
// 1 thread writes others just copy data into scratchpad(shared) memory
// Ideas from https://www.cis.upenn.edu/~devietti/classes/cis601-spring2017/slides/gpu-concurrency.pdf

#include <cuda/atomic>
#include <cstdio>
#include <stdlib.h>
#include <time.h> // for seeding the random number generator

using namespace cuda;
#define THREADS 1024

__global__ void init_stress_region(int *stress_data){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    stress_data[tid] = rand();
}

    int stress_mem_region = 1024;
    int stress_line_size = 64;
    int word_size = 1;

    int default_stress_line_offset= 0;  // index to access

__global__ void consumer(atomic<int>* flag, int* data, int* result0/*flag*/, int*result1/*data*/, int *stress_data, int *stress_thread_numbers, int count_of_stress_threads) {
    __shared__ int buffer1[THREADS];
    __shared__ int buffer2[THREADS];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {   // some random thread
        *result0 = flag->load(memory_order_relaxed);
        *result1 = *data;    
    }
    for (int i=0; i< count_of_stress_threads; i++){
        if (tid == stress_thread_numbers[i] && tid!=0){
            int count_of_lines=stress_mem_region/stress_line_size;
            // Generate a random number between 0 and count_of_lines
            int random_line_offset = rand() % (count_of_lines + 1);
            int random_target_offset = rand() % (stress_line_size + 1);
            int random_target = stress_data + stress_line_size*random_line_offset + random_target_offset;
            buffer1[tid] = stress_data[random_target];  // access this data somewhere.

        }//else skip
    }
     else{ // stressing threads
        

        int counter=0;
        while(counter!=0){
            stress_data[];
            stress_data[] ;
        counter--;
        }
    }
}

int all_are_same(const int *result) {
    int sum = 0;
    for (int i = 0; i < THREADS; ++i) {
        sum += result[i];
    }
    float avg = (float)sum / THREADS; // Convert sum to float before division
    if (avg != result[0]) {
        // for (int i = 0; i < THREADS; ++i) {
        //     printf("%d - ", result[i]);
        // }
        // printf("\n Avg = %f", avg);
        // printf("\n Result[0] = %d", result[0]);
        return 0;
    } else {
        return 1;
    }
}
    
#define SAFE(x) if (0 != x) { abort(); }

struct Result{
int seq1, seq2, interleave, weak;
};


void run(Result *count_local){
    // int THREADS = 1024;

    atomic<int>* flag;
    int* data;
    int *stress_data;
    atomic<int>* stress_flag;
    int* result0;
    int* result1; // r0= flag, r1=data, GPUHarbor way

    int data_in_unified_memory = 1;

    int stress_mem_region = 1024;
    int stress_line_size = 64;
    int word_size = 1;

    int default_stress_line_offset= 0;  // index to access

    ////////////////////////////////////////////////////////////////////////////

    // Flag in unified memory
    SAFE(cudaMallocManaged(&flag, sizeof(atomic<int>)));
    SAFE(cudaMallocManaged(&stress_flag, sizeof(atomic<int>)));

    // Data placed as specified
    if (data_in_unified_memory) {
        SAFE(cudaMallocManaged(&data, sizeof(int)));
        SAFE(cudaMallocManaged(&stress_data, stress_mem_region * sizeof(int)));
    }

    // Result array pinned in CPU memory
    SAFE(cudaMallocHost(&result0, sizeof(int)));
    SAFE(cudaMallocHost(&result1, sizeof(int)));

    // Initial values: data = <unknown>, flag = 0
    flag->store(0, memory_order_relaxed);
    *data = 0;
    // init stress region
    srand(time(NULL));
    
    // Initialize each element of the array with a random number
    // Cache on CPU/ partially
    for (int i = 0; i < stress_mem_region/2; i++) {
        stress_data[i] = rand();
    }    
    // Cache on GPU.
    init_stress_region<<<1, 512>>>(stress_data);    // 512 = half of before
    stress_flag->store(0, memory_order_relaxed);

    ////////////////////////////////////////////////////////////////////////////

    // Launch the consumer asynchronously
    // consumer<<<1,1>>>(flag, data, result0, result1);
    
    // gpU
    // max go till 2 blocks and n threads.
    consumer<<<1,1>>>(flag, data, result0, result1); 
    // cudaError_t error = cudaGetLastError();
    // if (error != cudaSuccess) {
    //     printf("CUDA kernel launch error: %s\n", cudaGetErrorString(error));
    //     exit(1);
    // }

    
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

     //r0=flag, r1=data
    if (*result0 == 0 && *result1 == 0){
        count_local->seq1 += 1 ;  //# t1->t2
    }
    else if(*result0 == 1 && *result1 == 42){
        count_local->seq2 += 1 ;  //# t2-t1
    }
    else if(*result0 == 0 && *result1 == 42){
        count_local->interleave += 1;
    }
    else if(*result0 == 1 && *result1 == 0){
        count_local->weak += 1;
    }
        

        

       // Free the allocated memory at the end
    SAFE(cudaFree(flag));
    SAFE(cudaFree(stress_flag));
    if (data_in_unified_memory) {
        SAFE(cudaFree(data));
        SAFE(cudaFree(stress_data));
    } else {
        SAFE(cudaFree(data));
    }
    SAFE(cudaFreeHost(result0));
    SAFE(cudaFreeHost(result1));


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

