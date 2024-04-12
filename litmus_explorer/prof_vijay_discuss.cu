//Discussion done in the TA room
// flip producer/ consumer.
// cache on CPU
// cudaMallocHost.

#include <cuda/atomic>
#include <cstdio>

using namespace cuda;
// GPU - writer
__global__ void producer(atomic<int>* flag, int* data, int* result0/*flag*/, int*result1/*data*/) {
    // // Get the start time
    // clock_t start = clock();    
        // // Busy-wait loop until the specified time has elapsed
        // while ((clock() - start) * 1000 / CLOCKS_PER_SEC < 1000) {
        //     // Do nothing, just wait
        // }    

        // for (int i=0; i< 10000; i++){
            *data = 42;      
            flag->store(1, memory_order_relaxed);
        // }

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
    atomic<int>* flag;
    int* data;
    int* result0, *result1; // r0= flag, r1=data, GPUHarbor way
    int data_in_unified_memory = 1;



    // ALLOCATION
    // ---------------------------------------
    // Flag in unified memory
    SAFE(cudaMallocHost(&flag, sizeof(atomic<int>)));
    SAFE(cudaMallocHost(&data, sizeof(int)));
    // Result array pinned in CPU memory
    SAFE(cudaMallocHost(&result0, sizeof(int)));
    SAFE(cudaMallocHost(&result1, sizeof(int)));
    
    // atomic<int> *x;//= (int*)malloc(sizeof(int));
    // int *y;// = (int*)malloc(sizeof(int));
    // SAFE(cudaMallocHost(&x, sizeof(atomic<int>)));   // x == flag
    // SAFE(cudaMallocHost(&y, sizeof(int)));   // y==data 
    // caching(x,y,flag, data);   
    // INIT
    
    int x = flag->load(memory_order_relaxed);// simple caching
    int y = *data;
    // ---------------------------------------
    // Initial values: data = <unknown>, flag = 0
    flag->store(0, memory_order_relaxed);
    *data = 0;

    // Launch the PRODUCER asynchronously
    producer<<<1,1>>>(flag, data, result0, result1);
    
    // for(int j=0; j<1000; j++){
    // CPU consumer - reader
    *result0 = flag->load(memory_order_relaxed);
    *result1 = *data;
    // }
  


  
  // Wait for consumer to finish
    SAFE(cudaDeviceSynchronize());
    
    //r0=flag, r1=data
    if (*result0 == 0 && *result1 == 0){    // cpu-gpu
        count_local->seq1 += 1 ;  //# t1->t2
    }
    else if(*result0 == 1 && *result1 == 42){   // gpu-cpu
        count_local->seq2 += 1 ;  //# t2-t1
    }
    else if(*result0 == 0 && *result1 == 42){   //intrvl
        count_local->interleave += 1;
    }
    else if(*result0 == 1 && *result1 == 0){    // weak
        count_local->weak += 1;
    }
        

       // Free the allocated memory at the end
    SAFE(cudaFreeHost(flag));
    SAFE(cudaFreeHost(data));
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