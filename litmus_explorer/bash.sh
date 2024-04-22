ml gcc/11.2.0 openmpi cuda python/3.11.7
nvcc -Xcompiler -fopenmp -arch=sm_60 gpuharbor_acq_rel.cu -o gpuharbor_acq_rel

#for i in {1..100} ; do
#	./gpuharbor_acq_rel
#done 
