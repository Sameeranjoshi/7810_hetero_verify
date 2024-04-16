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

struct Result{
int seq1, seq2, interleave, weak;
};
struct stress {
    int testIterations;
    int testingWorkgroups;
    int maxWorkgroups;
    int workgroupSize;
    int shufflePct;
    int barrierPct;
    int stressLineSize;
    int stressTargetLines;
    int scratchMemorySize;
    int memStride;
    int memStressPct;
    int memStressIterations;
    int memStressPattern;
    int preStressPct;
    int preStressIterations;
    int preStressPattern;
    int stressAssignmentStrategy;
    int permuteThread;
};
struct test {
    int numOutputs;
    int numMemLocations;
    int numResults;
    int permuteLocation;
    int aliasedMemory;
    int workgroupMemory;
    int checkMemory;
};

__device__ void spinLoop(int duration) {
    clock_t start = clock();
    while ((clock() - start) < duration) {}
}

__global__ void print_verify_init_ok_gpu(int *barrier){
    for (int i=0; i< 12; i++){
        printf("\n %d = %d", i, barrier[i]);
    }
}
int setBetween(int min, int max) {
  if (min == max) {
    return min;
  } else {
    int size = rand() % (max - min);
    return min + size;
  }
}

bool percentageCheck(int percentage) {
  bool x = rand() % 100 < percentage;
  return x;
}

void setShuffledWorkgroups(int* h_shuffledWorkgroups, int numWorkgroups, int shufflePct) {
  for (int i = 0; i < numWorkgroups; i++) {
    h_shuffledWorkgroups[i] = i;
  }
  if (percentageCheck(shufflePct)) {
    for (int i = numWorkgroups - 1; i > 0; i--) {
      int swap = rand() % (i + 1);
      int temp = h_shuffledWorkgroups[i];
      h_shuffledWorkgroups[i] = h_shuffledWorkgroups[swap];
      h_shuffledWorkgroups[swap] = temp;
    }
  }
}

void setScratchLocations(int *h_locations, int numWorkgroups, struct stress* params) {
//   set <int> usedRegions;
  int numRegions = params->scratchMemorySize / params->stressLineSize;
  for (int i = 0; i < params->stressTargetLines; i++) {
    int region = rand() % numRegions;
    int counter=100;
    while(counter!=0){
      region = rand() % numRegions;
      counter--;
    }
    int locInRegion = rand() % (params->stressLineSize);
    switch (params->stressAssignmentStrategy) {
      case 0:
        for (int j = i; j < numWorkgroups; j += params->stressTargetLines) {
          h_locations[j] = ((region * params->stressLineSize) + locInRegion);
        }
        break;
      case 1:
        int workgroupsPerLocation = numWorkgroups/params->stressTargetLines;
        for (int j = 0; j < workgroupsPerLocation; j++) {
          h_locations[i*workgroupsPerLocation + j] =  ((region * params->stressLineSize) + locInRegion);
        }
        if (i == params->stressTargetLines - 1 && numWorkgroups % params->stressTargetLines != 0) {
          for (int j = 0; j < numWorkgroups % params->stressTargetLines; j++) {
            h_locations[numWorkgroups - j - 1] = ((region * params->stressLineSize) + locInRegion);
          }
        }
        break;
    }
  }
}

// Kernel function to access data on GPU by two threads
__global__ void accessData(atomic<int> *test_locations, int *shuffled_workgroups, int* barrier, int* scratchpad, int* scratch_locations, int* stress_params) {
    // printf("\n In kernel");
    int get_group_id = blockIdx.x;
    int get_local_size = blockDim.x;
    int get_local_id = threadIdx.x;

    int shuffled_workgroup = shuffled_workgroups[get_group_id]; // blockIdx.x
  if(shuffled_workgroup < stress_params[9]) {
    int total_ids = get_local_size * stress_params[9];  // blockDim.x
    int id_0 = shuffled_workgroup * get_local_size + get_local_id; // get_local_id() = threadIdx.x
    // int new_workgroup = stripe_workgroup(shuffled_workgroup, get_local_id, stress_params[9]);
    int new_workgroup = (shuffled_workgroup + 1 + get_local_id %(stress_params[9] - 1)) % stress_params[9];
    int p1 = (get_local_id*stress_params[7]) % get_local_size;
    int id_1 = new_workgroup * get_local_size + p1; 
    int p2 = (id_0 * stress_params[8]) % total_ids;
    int p3 = (id_1 * stress_params[8]) % total_ids;
    int x_0 = (id_0) * stress_params[10] * 2;
    int y_0 = p2 * stress_params[10] * 2 + stress_params[11];
    int x_1 = (id_1) * stress_params[10] * 2;
    int y_1 = p3 * stress_params[10] * 2 + stress_params[11];

//             d_result[0] = d_flag->load(memory_order_relaxed);
//             d_result[1] = *d_data;
//             *d_data = 42;
//             d_flag->store(1, memory_order_relaxed);
    test_locations[x_0].store(1, memory_order_relaxed);
    test_locations[y_0].store(1, memory_order_relaxed);
    int r0 = test_locations[y_1].load(memory_order_relaxed);
    int r1 = test_locations[x_1].load(memory_order_relaxed);
    if (r0 == 1 && r1 == 0){
        printf("\n Weak found");
    }
    // atomic_store(&read_results[id_1 * 2 + 1], r1);
    // atomic_store(&read_results[id_1 * 2], r0);
  }
}
// // Kernel function to access data on GPU by two threads
// __global__ void accessData(atomic<int>* d_flag, int *d_data, int *d_result, int *d_buffer, int tid0, int tid1) {
//     int threadId = threadIdx.x + blockIdx.x * blockDim.x;   // testing threads
//     // int maxThreadsPossible = blockDim.x;   // max value of threads use this for random number generator.

//     int testing_t0_id = tid0; //(threadId * 1103 + 12345) % (maxThreadsPossible); // can't use rand() inside kernel use custom one.
//     int testing_t1_id = tid1; //((testing_t0_id) * 1103 + 12345) % (maxThreadsPossible);   // scope trees different(1023 before rand())

//         // printf("\n tid0 = %d", testing_t0_id);
//         // printf("\n tid1 = %d", testing_t1_id);
//         // printf("\n maxThreadsPossible = %d", maxThreadsPossible);

//     // Generate random numbers within the range [0, maxThreadsPossible]

//     int testing_warp0_id = testing_t0_id/32;
//     int testing_warp1_id = testing_t1_id/32;

//     __shared__ char s_buffer1[1024];
//     __shared__ char s_buffer2[1024];

//     // all threads have to be here at this point before we start real work.
//     // __syncthreads();       // possibly threadsync incantation.
//     if (threadId == testing_t0_id) {    // t0 = writer
//         // printf("\n t0 = %d", threadId);
//         spinLoop(100000);
//         for (int i=0; i< 1000; i++){
//             *d_data = 42;
//             d_flag->store(1, memory_order_relaxed);
//         }
//         spinLoop(100000);
//     }
//     else if (threadId == testing_t1_id) { // t1 = reader
//         // printf("\n t1 =%d", threadId);
//         // printf("\n maxThreadsPossible = %d", maxThreadsPossible);
//         spinLoop(100000);
//         for (int i =1000; i!=0; i--){
//             d_result[0] = d_flag->load(memory_order_relaxed);
//             d_result[1] = *d_data;
//         }
//         spinLoop(100000);
//     }   
//     else{   // stressing threads
//         // threads in same warp - create bank conflict.
//         // statically I know 1 warp = 32 threads 
//         // find warp id of stress thread and if equal to testing thread warp id - this thred has potential to create bank conflict.
//         int stress_warp_id = threadId/32;

//         if(stress_warp_id == testing_warp0_id || stress_warp_id == testing_warp1_id){
//             // my stress thread is in same warp as testing thread, this will create bank conflict.
//             // s_buffer1[2*threadId] = d_data[2*threadId];  // another way of bank conflict
//             // Each thread accesses shared memory with bank conflicts
//             s_buffer1[threadId] = d_buffer[threadId % 32]; // Bank conflicts may occur here
//             __syncthreads();

//         } else{ // warps are not same of testing and stress threads.
//             // perform basic incantation of stressing.
//             for(int i=0; i< 1000; i++){
//                 __shared__ int temp;
//                 temp = *d_data;
//                 d_buffer[threadId] = temp;
//                 __syncthreads();
//             }
//         }
//     }
// }

void run(Result *count_local){
    // init struct
    struct stress stress_params = {
        .testIterations = 1,
        .testingWorkgroups = 1024,
        .maxWorkgroups = 1024,
        .workgroupSize = 1,
        .shufflePct = 0,
        .barrierPct = 0,
        .stressLineSize = 1,
        .stressTargetLines = 0,
        .scratchMemorySize = 1,
        .memStride = 1,
        .memStressPct = 0,
        .memStressIterations = 0,
        .memStressPattern = 0,
        .preStressPct = 0,
        .preStressIterations = 0,
        .preStressPattern = 0,
        .stressAssignmentStrategy = 0,
        .permuteThread = 0
    };
    struct test test_params = {
        .numOutputs=2,
        .numMemLocations=1,
        .numResults=4,
        .permuteLocation=1031,
        .aliasedMemory=0,
        .workgroupMemory=0,
        .checkMemory=0
    };
  
    // 0, 1, 4 are dynamic values, keeping 0 for now.
    int h_stressParams[12] = {  0,  /*0*/
                                0,  /*1*/
                                stress_params.memStressIterations, 
                                stress_params.memStressPattern,
                                0,  /*4 - dynamic value*/
                                stress_params.preStressIterations,
                                stress_params.preStressPattern, 
                                stress_params.permuteThread, 
                                test_params.permuteLocation,
                                stress_params.testingWorkgroups,
                                stress_params.memStride,
                                (test_params.aliasedMemory == 1) ? 0 : stress_params.memStride
                                };

    int testingThreads = stress_params.workgroupSize * stress_params.testingWorkgroups;
    int testLocSize = testingThreads * test_params.numMemLocations * stress_params.memStride;

    // init other variables
    atomic<int> h_flag=0;
    int h_data=0;
    atomic<int>* d_flag;
    int *d_data;
    int *d_buffer;
    int h_result[2] = {99,99};
    int* d_result;

    // pointers
    atomic<int>* testLocations;
    int* shuffledWorkgroups;
    int* barrier;
    int* scratchpad;
    int* scratchLocations;
    int* stressParams;
    


    // Allocate memory on GPU
    cudaMalloc(&d_flag,  sizeof(atomic<int>));
    cudaMalloc(&d_data,  sizeof(int));
    cudaMalloc(&d_buffer,  1024*sizeof(int));
    cudaMalloc(&d_result, 2*sizeof(int));
    // allocations
    cudaMalloc(&testLocations, testLocSize*sizeof(atomic<int>));
    cudaMalloc(&shuffledWorkgroups, stress_params.maxWorkgroups * sizeof(int));
    cudaMalloc(&barrier, 1 * sizeof(int));
    cudaMalloc(&scratchpad, stress_params.scratchMemorySize * sizeof(int));
    cudaMalloc(&scratchLocations, stress_params.maxWorkgroups * sizeof(int));
    cudaMalloc(&stressParams, 12 * sizeof(int));
    
    
    // Initialize
    cudaMemset(testLocations, 0, testLocSize * sizeof(atomic<int>));
    cudaMemcpy(stressParams, &h_stressParams, 12 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(barrier, 0, 1 * sizeof(int));
    cudaMemset(scratchpad, 0, stress_params.scratchMemorySize * sizeof(int));
    // Copy data from host to device
    cudaMemcpy(d_flag, &h_flag, sizeof(atomic<int>), cudaMemcpyHostToDevice);   // init
    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);   // init
    cudaMemcpy(d_result, &h_result, 2*sizeof(int), cudaMemcpyHostToDevice);   // init
//    print_verify_init_ok_gpu<<<1,1>>>(stressParams);
   
// ---------------------------------------------------------------
    //for (int i = 0; i < stress_params.testIterations; i++) {

    int numWorkgroups = setBetween(stress_params.testingWorkgroups, stress_params.maxWorkgroups);   // basically blocks
    int workGroupSize = stress_params.workgroupSize;    //1
    int BLOCKS = numWorkgroups; // 1024
    int THREADS= workGroupSize; // 1 

    int* h_shuffledWorkgroups = (int *)malloc(numWorkgroups*sizeof(int));   // on cpu used for copying.
    setShuffledWorkgroups(h_shuffledWorkgroups, numWorkgroups, stress_params.shufflePct);   // random indexes
    cudaMemcpy(shuffledWorkgroups, &h_shuffledWorkgroups, numWorkgroups * sizeof(int), cudaMemcpyHostToDevice);

    int* h_scratchLocations = (int *)malloc(numWorkgroups*sizeof(int));   // on cpu used for copying.
    setScratchLocations(h_scratchLocations, numWorkgroups, &stress_params);   // random indexes
    cudaMemcpy(scratchLocations, &h_scratchLocations, numWorkgroups * sizeof(int), cudaMemcpyHostToDevice);

    accessData<<<BLOCKS, THREADS>>>(testLocations, shuffledWorkgroups, barrier, scratchpad, scratchLocations, stressParams);
    // accessData<<<BLOCKS, THREADS>>>(d_flag, d_data, d_result, d_buffer, t0, t1); 
    // Synchronize to ensure kernel finishes before accessing data
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
    // Copy data back from device to host
    cudaMemcpy(h_result, d_result, 2*sizeof(int), cudaMemcpyDeviceToHost);
    
    // // Print modified data
    // printf("flag: %d, data %d\n", h_result[0], h_result[1]);


    if (h_result[0]== 99 && h_result[1] == 99){
        printf("\n Bug in CUDA implementation! exiting");
        exit(1);
    }
    error = cudaGetLastError();
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
    free(h_shuffledWorkgroups);

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
    // srand(time(0));
    srand(time(NULL));
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
