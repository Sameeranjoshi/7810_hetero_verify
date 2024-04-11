// To compile and run:
// nvcc test.cu -arch=sm_80 -o test
// ./test

#include <cuda/atomic>
#include <cstdio>

using namespace cuda;

struct managed_data{
    atomic<int> flag;
    int data;
};
__global__ void consumer(managed_data *um_data, int* result0/*flag*/, int*result1/*data*/) {
    
    *result0 = um_data->flag.load(memory_order_relaxed);
    *result1 = um_data->data;
}

#define SAFE(x) if (0 != x) { abort(); }

struct Result{
int seq1, seq2, interleave, weak;
};


void run(Result *count_local){

    managed_data *um_data;
    int* result0, *result1; // r0= flag, r1=data, GPUHarbor way

    int data_in_unified_memory = 1;

    SAFE(cudaMallocManaged(&um_data, sizeof(managed_data)));

    // // Flag in unified memory
    // SAFE(cudaMallocManaged(&um_data.flag, sizeof(atomic<int>)));

    // // Data placed as specified
    // if (data_in_unified_memory) {
    //     SAFE(cudaMallocManaged(&um_data.data, sizeof(int)));
    // } else {
    //     SAFE(cudaMalloc(&um_data.data, sizeof(int)));
    

    // Result array pinned in CPU memory
    SAFE(cudaMallocHost(&result0, sizeof(int)));
    SAFE(cudaMallocHost(&result1, sizeof(int)));

    // Initial values: data = <unknown>, flag = 0
    um_data->flag.store(0, memory_order_relaxed);
    (um_data->data) = 0;

    // // Launch the consumer asynchronously
    consumer<<<1,1>>>(um_data, result0, result1);    
    
    // Producer sequence
    if (data_in_unified_memory) {
        (um_data->data) = 42;
    } else {
        int h_data = 42;
        SAFE(cudaMemcpy(&um_data->data, &h_data, sizeof(int), cudaMemcpyHostToDevice));
    }
    um_data->flag.store(1, memory_order_relaxed);


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
    SAFE(cudaFree(um_data));
    // if (data_in_unified_memory) {
    //     SAFE(cudaFree(um_data.data));
    // } else {
    //     SAFE(cudaFree(um_data.data));
    // }
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