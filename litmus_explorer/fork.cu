// To compile and run:
// nvcc test.cu -arch=sm_80 -o test
// ./test

#include <cuda/atomic>
#include <cstdio>

#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

using namespace cuda;

__global__ void consumer(atomic<int>* flag, int* data, int* result0/*flag*/, int*result1/*data*/) {

    // // Get the start time
    // clock_t start = clock();    

    // while (flag->load(memory_order_acquire) == 0) {}
    *result0 = flag->load(memory_order_relaxed);

        // // Busy-wait loop until the specified time has elapsed
        // while ((clock() - start) * 1000 / CLOCKS_PER_SEC < 1000) {
        //     // Do nothing, just wait
        // }

    *result1 = *data;
    
}

#define SAFE(x) if (0 != x) { abort(); }

struct Result{
int seq1, seq2, interleave, weak;
};

void run(Result *count_local){

    atomic<int>* flag;
    int* data;
    int* result0, *result1; // r0= flag, r1=data, GPUHarbor way

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

    int pid1, pid2;
    int status;

    pid1 = fork();
    if (pid1 == 0){ // child
            consumer<<<1,1>>>(flag, data, result0, result1);
            if (data_in_unified_memory) {
                *data = 42;
            } else {
                int h_data = 42;
                SAFE(cudaMemcpy(data, &h_data, sizeof(int), cudaMemcpyHostToDevice));
            }
            flag->store(1, memory_order_relaxed);   

    } else if (pid1 >0 ) {
                    consumer<<<1,1>>>(flag, data, result0, result1);
            if (data_in_unified_memory) {
                *data = 42;
            } else {
                int h_data = 42;
                SAFE(cudaMemcpy(data, &h_data, sizeof(int), cudaMemcpyHostToDevice));
            }
            flag->store(1, memory_order_relaxed);   
    }
    // This is the parent process and child to finish
    waitpid(pid1, &status, 0);


    
    
//   //delay
//     int i = 1000;
//     while(i ==0){
//         sleep(1);
//     }

  // Wait for consumer to finish
    SAFE(cudaDeviceSynchronize());

    // Print the result
    // printf("data = %d (expected 42) flag = %d \n", *data, flag->load(memory_order_acquire));
    // printf("result0=%d result1=%d \n", *result0, *result1);

    
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
        run(&count_local);            
        // printf("i=%d\n", i);
        
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