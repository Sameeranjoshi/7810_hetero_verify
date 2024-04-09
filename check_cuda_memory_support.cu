#include <stdio.h>

int main() {
  int d;
  cudaGetDevice(&d);

  int pma = 0;
  cudaDeviceGetAttribute(&pma, cudaDevAttrPageableMemoryAccess, d);
  printf("cudaDevAttrPageableMemoryAccess: %s\n", pma == 1? "YES" : "NO");
  
  int cma = 0;
  cudaDeviceGetAttribute(&cma, cudaDevAttrConcurrentManagedAccess, d);
  printf("cudaDevAttrConcurrentManagedAccess: %s\n", cma == 1? "YES" : "NO");


  int mm = 0;
  cudaDeviceGetAttribute(&mm, cudaDevAttrManagedMemory, d);
  printf("managedMemory: %s\n", mm == 1? "YES" : "NO");


  int dma = 0;
  cudaDeviceGetAttribute(&dma, cudaDevAttrDirectManagedMemAccessFromHost, d);
  printf("directManagedMemAccessFromHost: %s\n", dma == 1? "YES" : "NO");


  int x = 0;
  cudaDeviceGetAttribute(&x, cudaDevAttrPageableMemoryAccessUsesHostPageTables, d);
  printf("pageableMemoryAccessUsesHostPageTables: %s\n", x == 1? "YES" : "NO");


  return 0;
}
