// WRITE INTO SCRATCHPAD
// LOOP FOR EACH THREAD
// <<<1,1024>>>
// STRESSING VS TESTING THREADS
// PREFETCH ON GPU AND CPU
// PRODUCER=T0, CONSUMER=1023, MIDDLE = STRESSING THREADS
// SPINLOOP
// BANK CONFLICT IMPLEMENTED
// T0 AND T1 ARE NOT IN SAME SCOPE-TREE(BLOCK IN CUDA)
// CTA = BLOCK IN CUDA = IMPLEMENTED GLOBAL INDEXING OF THREADID.
// RANDOM THREAD ID
// THREAD SYNC 

#include <cuda/atomic>
#include <cstdio>
#include <cuda_runtime.h>
using namespace cuda;
#include <stdio.h>
#include <time.h>

// Function to introduce random delay
void randomDelay() {
    srand(time(NULL));
    int delay_ms = rand() % 1000;
    sleep(delay_ms / 1000); // divide by 1000 to convert milliseconds to seconds
}
__global__ void delayKernel() {
    // Get the global thread ID
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Introduce delay using a loop
    for (int i = 0; i < 1000000; ++i) {
        // Do some dummy computation
        float dummy = sqrtf(threadId * i);
    }
}


__device__ void spinLoop(int duration) {
    clock_t start = clock();
    while ((clock() - start) < duration) {}
}

#define spin_in_kernel(duration) \
    do { \
        for (int i = 0; i < duration; ++i) { \
            /* Dummy loop to introduce delay */ \
        } \
    } while (0)

__global__ void t0_producer(atomic<int>* d_flag, int *d_data, int *d_result, int *d_buffer, int tid0, int tid1) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;   // testing threads
        // spinLoop(100000);
        // for (int i=0; i< 1000; i++){
            *d_data = 42;
            spin_in_kernel(10000000);
            spin_in_kernel(10000000);
            spin_in_kernel(10000000);
            spin_in_kernel(10000000);
            spin_in_kernel(10000000);
            spin_in_kernel(10000000);
            spin_in_kernel(10000000);
            spin_in_kernel(10000000);
            spin_in_kernel(10000000);
            d_flag->store(1, memory_order_relaxed);
            spin_in_kernel(10000000);spin_in_kernel(10000000);spin_in_kernel(10000000);spin_in_kernel(10000000);spin_in_kernel(10000000);spin_in_kernel(10000000);
        // }
        // spinLoop(100000);
}
__global__ void t1_consumer(atomic<int>* d_flag, int *d_data, int *d_result, int *d_buffer, int tid0, int tid1) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;   // testing threads
        spinLoop(100000);
        for (int i =1000; i!=0; i--){
            d_result[0] = d_flag->load(memory_order_relaxed);
            d_result[1] = *d_data;
        }
        spinLoop(100000);
}
__global__ void tn_stress(atomic<int>* d_flag, int *d_data, int *d_result, int *d_buffer, int tid0, int tid1) {
    __shared__ char s_buffer1[1024];
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;   // testing threads
            for(int i=0; i< 1000; i++){
                __shared__ int temp;
                temp = *d_data;
                d_buffer[threadId] = temp;
                
            }
            s_buffer1[threadId] = d_buffer[threadId % 32]; // Bank conflicts may occur here
            // __threadfence();
}

// Kernel function to access data on GPU by two threads
__global__ void accessData(atomic<int>* d_flag, int *d_data, int *d_result, int *d_buffer, int tid0, int tid1) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;   // testing threads
    // int maxThreadsPossible = blockDim.x;   // max value of threads use this for random number generator.

    int testing_t0_id = tid0; //(threadId * 1103 + 12345) % (maxThreadsPossible); // can't use rand() inside kernel use custom one.
    int testing_t1_id = tid1; //((testing_t0_id) * 1103 + 12345) % (maxThreadsPossible);   // scope trees different(1023 before rand())

        // printf("\n tid0 = %d", testing_t0_id);
        // printf("\n tid1 = %d", testing_t1_id);
        // printf("\n maxThreadsPossible = %d", maxThreadsPossible);

    // Generate random numbers within the range [0, maxThreadsPossible]

    int testing_warp0_id = testing_t0_id/32;
    int testing_warp1_id = testing_t1_id/32;

    __shared__ char s_buffer1[1024];
    __shared__ char s_buffer2[1024];

    // all threads have to be here at this point before we start real work.
    // __syncthreads();       // possibly threadsync incantation.
    if (threadId == testing_t0_id) {    // t0 = writer
        // printf("\n t0 = %d", threadId);
        spinLoop(100000);
        for (int i=0; i< 1000; i++){
            *d_data = 42;
            d_flag->store(1, memory_order_relaxed);
        }
        spinLoop(100000);
    }
    else if (threadId == testing_t1_id) { // t1 = reader
        // printf("\n t1 =%d", threadId);
        // printf("\n maxThreadsPossible = %d", maxThreadsPossible);
        spinLoop(100000);
        for (int i =1000; i!=0; i--){
            d_result[0] = d_flag->load(memory_order_relaxed);
            d_result[1] = *d_data;
        }
        spinLoop(100000);
    }   
    else{   // stressing threads
        // threads in same warp - create bank conflict.
        // statically I know 1 warp = 32 threads 
        // find warp id of stress thread and if equal to testing thread warp id - this thred has potential to create bank conflict.
        int stress_warp_id = threadId/32;

        if(stress_warp_id == testing_warp0_id || stress_warp_id == testing_warp1_id){
            // my stress thread is in same warp as testing thread, this will create bank conflict.
            // s_buffer1[2*threadId] = d_data[2*threadId];  // another way of bank conflict
            // Each thread accesses shared memory with bank conflicts
            s_buffer1[threadId] = d_buffer[threadId % 32]; // Bank conflicts may occur here
            __syncthreads();

        } else{ // warps are not same of testing and stress threads.
            // perform basic incantation of stressing.
            for(int i=0; i< 1000; i++){
                __shared__ int temp;
                temp = *d_data;
                d_buffer[threadId] = temp;
                __syncthreads();
            }
        }
    }
}

struct Result{
int seq1, seq2, interleave, weak;
};



void run(Result *count_local){
    atomic<int> h_flag=0;
    int h_data=0;
    atomic<int>* d_flag;
    int *d_data;
    int *d_buffer;
    int h_result[2] = {99,99};
    int* d_result;


    // Allocate memory on GPU
    cudaMalloc(&d_flag,  sizeof(atomic<int>));
    cudaMalloc(&d_data,  sizeof(int));
    cudaMalloc(&d_buffer,  1024*sizeof(int));
    cudaMalloc(&d_result, 2*sizeof(int));
    
    // Copy data from host to device
    cudaMemcpy(d_flag, &h_flag, sizeof(atomic<int>), cudaMemcpyHostToDevice);   // init
    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);   // init
    cudaMemcpy(d_result, &h_result, 2*sizeof(int), cudaMemcpyHostToDevice);   // init

   
        // cudaMemPrefetchAsync(d_flag, sizeof(atomic<int>), 0, NULL);         
        // cudaMemPrefetchAsync(d_data, sizeof(int), 0, NULL);
    int BLOCKS = 1;
    int THREADS= 1024; 

    int maxthreadspossible = 0; // couldn't get min () to work, reverting to manual way
    int div = THREADS/BLOCKS;
    if (THREADS <= div){
        maxthreadspossible = THREADS;
    } else{
        maxthreadspossible = div;
    }
    
    

   // Generate t0
    
    int t0 = rand();
    t0 = rand() % (maxthreadspossible);
    int t1 = 1;
    do {
        t1 = rand() % (maxthreadspossible);
    } while (t1 == t0);
    

    //sanity check
    if ((t0 < 0 || t0 >= maxthreadspossible) || (t1 < 0 || t1 >= maxthreadspossible)){
        printf("\n Bug in CUDA implementation(tid<0 || tid>maxthreads)! exiting");
        printf("\n tid0 = %d", t0);
        printf("\n tid1 = %d", t1);
        printf("\n maxThreadsPossible = %d", maxthreadspossible);
        return;
    }

    // printf("\n Before running CUDA kernel");
    // printf("\n Testing thread IDs: t0 = %d, t1 = %d", t0, t1);
    // printf("\n BLOCKS= %d, THREADS=%d",BLOCKS, THREADS );
    // both t0 and t1 and different
    cudaDeviceSynchronize();
    
    t1_consumer<<<BLOCKS, 1>>>(d_flag, d_data, d_result, d_buffer, t0, t1);
    randomDelay();
    randomDelay();
    randomDelay();
    randomDelay();
    randomDelay();
    t0_producer<<<BLOCKS, 1>>>(d_flag, d_data, d_result, d_buffer, t0, t1);
    randomDelay();
    tn_stress<<<BLOCKS, THREADS>>>(d_flag, d_data, d_result, d_buffer, t0, t1);

randomDelay();
randomDelay();
randomDelay();
    // tested data(blocks)
    // accessData<<<555, 10>>>(d_flag, d_data, d_result, d_buffer);
    // accessData<<<1024, 1024>>>(d_flag, d_data, d_result, d_buffer);
    // accessData<<<199, 1024>>>(d_flag, d_data, d_result, d_buffer);
    // tested these(threads)
    // accessData<<<1, 1>>>(d_flag, d_data, d_result, d_buffer);
    // accessData<<<1, 999>>>(d_flag, d_data, d_result, d_buffer);
    // accessData<<<1, 50>>>(d_flag, d_data, d_result, d_buffer);
    // accessData<<<1, 868>>>(d_flag, d_data, d_result, d_buffer);
                  


    // Synchronize to ensure kernel finishes before accessing data
    cudaDeviceSynchronize();

        // cudaMemPrefetchAsync(d_flag, sizeof(atomic<int>), cudaCpuDeviceId, NULL); 
        // cudaMemPrefetchAsync(d_data, sizeof(int), cudaCpuDeviceId, NULL); 
    
    // Copy data back from device to host
    cudaMemcpy(h_result, d_result, 2*sizeof(int), cudaMemcpyDeviceToHost);
    
    // // Print modified data
    // printf("flag: %d, data %d\n", h_result[0], h_result[1]);


    if (h_result[0]== 99 && h_result[1] == 99){
        printf("\n Bug in CUDA implementation! exiting");
        exit(1);
    }
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }            
     //r0=flag, r1=data
    if (h_result[0] == 0 && h_result[1] == 0){
        count_local->seq1 += 1 ;  //# t1->t2
    }
    else if(h_result[0] == 1 && h_result[1] == 42){
        count_local->seq2 += 1 ;  //# t2-t1
    }
    else if(h_result[0] == 0 && h_result[1] == 42){
        count_local->interleave += 1;
    }
    else if(h_result[0] == 1 && h_result[1] == 0){
        count_local->weak += 1;
    }

    // Free device memory
    cudaFree(d_flag);
    cudaFree(d_data);
    cudaFree(d_buffer);

}


int main(int argc, char* argv[]) {
        if (argc !=2 ){
            printf("\n ./a.out <number of tests>");
            exit(1);
        }
     int loop_size = atoi(argv[1]);
     Result count_local{0};
    srand(time(0));
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