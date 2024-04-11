// Stress Cache line - 
// 1 thread writes others just copy data into scratchpad(shared) memory
// Ideas from https://www.cis.upenn.edu/~devietti/classes/cis601-spring2017/slides/gpu-concurrency.pdf

#include <cuda/atomic>
#include <cstdio>
#include <iostream>
#include <cstdlib> // for rand() and srand()
#include <ctime>   // for time()

using namespace cuda;
#define THREADS 1024


__global__ void consumer(atomic<int>* flag, int* data, atomic<int>* umflag_near_flag, int *um_buffer1, int* result0/*flag*/, int*result1/*data*/) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {   // some random thread
        *result0 = flag->load(memory_order_relaxed);
        *result1 = *data;    
    }// else{
    //         // for (int i=0; i< 4096; i++)
    //             um_buffer1[tid] = umflag_near_flag->load(memory_order_relaxed);
    // }
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

void caching(atomic<int> *x, int *y, atomic<int> *flag, int *data ){

    for (int i=0; i<100; i++){
        *x = *data;
        *y = flag->load(memory_order_relaxed);
    }
}
struct Result{
int seq1, seq2, interleave, weak;
};


void run(Result *count_local){
    // int THREADS = 1024;

    atomic<int>* flag;
    atomic<int>* umflag_near_flag;
    int* data;
    int* um_buffer1;  // page size bytes
    int* result0;
    int* result1; // r0= flag, r1=data, GPUHarbor way

    int data_in_unified_memory = 1;

    ////////////////////////////////////////////////////////////////////////////

    // Flag in unified memory
    SAFE(cudaMallocManaged(&flag, sizeof(atomic<int>)));
    SAFE(cudaMallocManaged(&umflag_near_flag, sizeof(atomic<int>)));

    // Data placed as specified
    if (data_in_unified_memory) {
        SAFE(cudaMallocManaged(&data, sizeof(int)));
        SAFE(cudaMallocManaged(&um_buffer1, 4096 * sizeof(int)));
    } else {
        SAFE(cudaMalloc(&data, sizeof(int)));
        SAFE(cudaMalloc(&um_buffer1, 4096 * sizeof(int))); 
    }

    // Result array pinned in CPU memory
    SAFE(cudaMallocHost(&result0, sizeof(int)));
    SAFE(cudaMallocHost(&result1, sizeof(int)));

    // Initial values: data = <unknown>, flag = 0
    flag->store(0, memory_order_relaxed);   // real one
    // umflag_near_flag->store(100, memory_order_relaxed); // fake 
    *data = 0;  // real one
    // random values
    // srand(static_cast<unsigned int>(time(nullptr)));
    // for (int i=0; i< 4096; i++)
    //     um_buffer1[i] = rand() %100 + 1;  // fake
    

    ////////////////////////////////////////////////////////////////////////////

    // Launch the consumer asynchronously
    // consumer<<<1,1>>>(flag, data, result0, result1);
    
    // gpU
    // max go till 2 blocks and n threads.
    consumer<<<1,1024>>>(flag, data, umflag_near_flag, um_buffer1, result0, result1); 

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
    
    // // THIS IS THE PART WHICH TRIGGERS LOTS OF BEHAVIORS
    // // Read data and cache it in CPU.
    // for (int i=0; i< 1024; i++){
    //     um_buffer1[i] = *data;    // reads data and cache it in CPU side.
    //     umflag_near_flag->store(flag->load(memory_order_relaxed), memory_order_relaxed);    // cross read to make more confusion.
    // }
    // cpu allocation data doesn't help
    // 


    atomic<int> *x;//= (int*)malloc(sizeof(int));
    int *y;// = (int*)malloc(sizeof(int));
    SAFE(cudaMallocManaged(&x, sizeof(atomic<int>)));   // x == flag
    SAFE(cudaMallocManaged(&y, sizeof(int)));   // y==data
    caching(x,y,flag, data);
    // Producer sequence
    if (data_in_unified_memory) {
        
        *data = 42;
        // // for no reason write in this buffer
        // for (int i=0; i< 4096; i++)
        //     um_buffer1[i] = rand() %100 + 1;  // fake        
    } else {
        int h_data = 42;
        SAFE(cudaMemcpy(data, &h_data, sizeof(int), cudaMemcpyHostToDevice));
    }
    // for (int i=0; i<100; i++){
    //     *x = *data;
    //     *y = flag->load(memory_order_relaxed);
    // }    
    flag->store(1, memory_order_relaxed);


  // Wait for consumer to finish
    SAFE(cudaDeviceSynchronize());

    // // perform some checks to make sure all are same
    // if ((all_are_same(result0) == 0) || (all_are_same(result1) == 0)){  // someone is not same
    //     printf("\n Isssue with implementation please fix!");
    // }


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
    if (data_in_unified_memory) {
        SAFE(cudaFree(data));
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

