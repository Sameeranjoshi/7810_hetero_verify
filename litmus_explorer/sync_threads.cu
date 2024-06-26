// try sync as mentioned in GPU COncurrency paper.

#include <cuda/atomic>
#include <cstdio>

using namespace cuda;

__global__ void consumer(atomic<int>* flag, int* data, int* result0/*flag*/, int*result1/*data*/) {
    
    *data = 42;
    __syncthreads();
    __syncthreads();
    __syncthreads();    
    __syncthreads();
    __syncthreads();
    __syncthreads();
    __syncthreads();    
    __syncthreads();
    __syncthreads();
    __syncthreads();
    __syncthreads();    
    __syncthreads();
    __syncthreads();
    __syncthreads();
    __syncthreads();    
    __syncthreads();
    __syncthreads();
    __syncthreads();
    __syncthreads();    
    __syncthreads();
    __syncthreads();
    __syncthreads();
    __syncthreads();    
    __syncthreads();
    __syncthreads();
    __syncthreads();
    __syncthreads();    
    __syncthreads();
    __syncthreads();
    __syncthreads();
    __syncthreads();    
    __syncthreads();            
    
    flag->store(1, memory_order_relaxed);    
    
    
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
    SAFE(cudaMallocManaged(&data, sizeof(int)));

    // Result array pinned in CPU memory
    SAFE(cudaMallocHost(&result0, sizeof(int)));
    SAFE(cudaMallocHost(&result1, sizeof(int)));
    
    // Initial values: data = <unknown>, flag = 0
    flag->store(0, memory_order_relaxed);
    *data = 0;
    
    ////////////////////////////////////////////////////////////////////////////
    // Launch the consumer asynchronously
    consumer<<<1,1>>>(flag, data, result0, result1);
    *result1 = *data;
    *result0 = flag->load(memory_order_relaxed);


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