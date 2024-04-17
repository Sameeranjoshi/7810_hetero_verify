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
long int seq1, seq2, interleave, weak;
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

void useless(){
    // printf("\n x_0 = %d", x_0);
    // printf("\n stress_params[10] = %d", stress_params[10]);
    // printf("\n id_0 = %d", id_0);
    // printf("\n get_local_size = %d", get_local_size);
    // printf("\n get_local_id = %d", get_local_id);
    // printf("\n id1 = %d", id_1);
    // printf("\n X[0] = %d, Y[0] = %d", x_0, y_0);
    // printf("\n X[1] = %d, Y[1] = %d", x_1, y_1);
    // printf("\n shuffled_workgroup = %d", shuffled_workgroup);
    // printf("\n get_group_id = %d", get_group_id);
    // printf("\n stress_params[9] = %d", stress_params[9]);    
 }

// Kernel function to access data on GPU by two threads
__global__ void accessData(atomic<int> *test_locations, atomic<int> *read_results, int *shuffled_workgroups, int* barrier, int* scratchpad, int* scratch_locations, int* stress_params, int numWorkgroups, int*globalVar_weak, int *globalVar_seq1, int *globalVar_seq2, int* globalVar_interleave, int *else__) {
    atomicAdd(else__, 1);
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
 
    test_locations[x_0].store(1, memory_order_relaxed); // data
    test_locations[y_0].store(1, memory_order_relaxed);  // flag
    
    int r0 = test_locations[y_1].load(memory_order_relaxed);  // flag
    int r1 = test_locations[x_1].load(memory_order_relaxed);  // data
    read_results[id_1 * 2 + 1].store(r1, memory_order_relaxed); // r1 = data
    read_results[id_1 * 2].store(r0, memory_order_relaxed); // r0 = flag
    // id_capture[id_1*2] = r0;
    if (r0 == 1 && r1 == 0){
      atomicAdd(globalVar_weak, 1);
    } else if (r0==1 && r1 == 1){
      atomicAdd(globalVar_seq2, 1);
    } else if (r0==0 && r1==0){
      atomicAdd(globalVar_seq1, 1);
    } else if (r0==0 && r1 == 1){
      atomicAdd(globalVar_interleave, 1);
    }else{
      printf("\n r0=%d, r1=%d", r0, r1);
    }
  }
}

void run(Result *count_local){
    // init struct
    struct stress stress_params = {
        .testIterations = 10,
        .testingWorkgroups = 2,
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

    // pointers
    atomic<int>* testLocations;
    atomic<int>*readResults;
    atomic<int>*h_readResults = (atomic<int>*)malloc(test_params.numOutputs * testingThreads * sizeof(atomic<int>));
    int* shuffledWorkgroups;
    int* barrier;
    int* scratchpad;
    int* scratchLocations;
    int* stressParams;
    

    // allocations
    cudaMallocManaged(&testLocations, testLocSize*sizeof(atomic<int>));
    cudaMallocManaged(&readResults, test_params.numOutputs * testingThreads * sizeof(atomic<int>));
    cudaMallocManaged(&shuffledWorkgroups, stress_params.maxWorkgroups * sizeof(int));
    cudaMallocManaged(&barrier, 1 * sizeof(int));
    cudaMallocManaged(&scratchpad, stress_params.scratchMemorySize * sizeof(int));
    cudaMallocManaged(&scratchLocations, stress_params.maxWorkgroups * sizeof(int));
    cudaMallocManaged(&stressParams, 12 * sizeof(int));
    
    // Initialize
    for (int i=0; i<testLocSize; i++){
      testLocations[i].store(0);
    }
    for (int i=0; i<12; i++){
      stressParams[i] = h_stressParams[i];
    }
    barrier = 0;
    for (int i=0; i< stress_params.scratchMemorySize; i++){
      scratchpad[i] = 0;
    }
    // Copy data from host to device
   
// ---------------------------------------------------------------
  for (int i = 0; i < stress_params.testIterations; i++) {
    //device params    
    int* d_globalVar_weak;
    int *d_else__;
    int* d_globalVar_seq1;
    int* d_globalVar_seq2;
    int* d_globalVar_interleave;


    cudaMallocManaged(&d_else__, sizeof(int));
    *d_else__ = 0;
    cudaMallocManaged(&d_globalVar_weak, sizeof(int));
    *d_globalVar_weak = 0;
    cudaMallocManaged(&d_globalVar_seq1, sizeof(int));
    *d_globalVar_seq1 = 0;
    cudaMallocManaged(&d_globalVar_seq2, sizeof(int));
    *d_globalVar_seq2 = 0;
    cudaMallocManaged(&d_globalVar_interleave, sizeof(int));
    *d_globalVar_interleave = 0;
    
    // init sizes
    int numWorkgroups = setBetween(stress_params.testingWorkgroups, stress_params.maxWorkgroups);   // basically blocks
    int workGroupSize = stress_params.workgroupSize;    //1
    int* h_shuffledWorkgroups = (int *)malloc(numWorkgroups*sizeof(int));   // on cpu used for copying.
    int* h_scratchLocations = (int *)malloc(numWorkgroups*sizeof(int));   // on cpu used for copying.
    int BLOCKS = numWorkgroups; // 1024
    int THREADS= workGroupSize; // 1 

    // Real shuffling algorithm

    setShuffledWorkgroups(h_shuffledWorkgroups, numWorkgroups, stress_params.shufflePct);   // random indexes
    for(int i=0; i< numWorkgroups; i++){
      shuffledWorkgroups[i] = h_shuffledWorkgroups[i];
    }
    setScratchLocations(h_scratchLocations, numWorkgroups, &stress_params);   // random indexes
    for(int i=0; i< numWorkgroups; i++){
      scratchLocations[i] = h_scratchLocations[i];
    }

    ////////////////////////////////////////////
    accessData<<<BLOCKS, THREADS>>>(testLocations, readResults, shuffledWorkgroups, barrier, scratchpad, scratchLocations, stressParams, numWorkgroups, d_globalVar_weak, d_globalVar_seq1, d_globalVar_seq2,d_globalVar_interleave, d_else__);
    ////////////////////////////////////////////
    // Synchronize to ensure kernel finishes before accessing data
    cudaDeviceSynchronize();

    // check if any errors from kernel side
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(1);
    }

    cudaDeviceSynchronize();
    // global result capture
    count_local->seq1 += (*d_globalVar_seq1);
    count_local->seq2 += (*d_globalVar_seq2);
    count_local->interleave += (*d_globalVar_interleave);
    count_local->weak += (*d_globalVar_weak);
    
    // Free device memory and host mem.

    free(h_shuffledWorkgroups);
    cudaFree(d_globalVar_weak);
    cudaFree(d_else__);
    cudaFree(d_globalVar_seq1);
    cudaFree(d_globalVar_seq2);
    cudaFree(d_globalVar_interleave);
  }
    
}


int main(int argc, char* argv[]) {
  Result count_local{0};
  srand(time(NULL));
  run(&count_local);

  int total = count_local.seq1 + count_local.seq2 + count_local.interleave + count_local.weak;
  printf("\n Histogram after %d runs\n", total);
  printf("seq1 (flag)=0; (data)=0;  = %d\n", count_local.seq1);
  printf("seq2 (flag)=1; (data)=42; = %d\n", count_local.seq2);
  printf("intlv(flag)=0; (data)=42; = %d\n", count_local.interleave);
  printf("weak (flag)=1; (data)=0;  = %d\n", count_local.weak);
  return 0;
}
