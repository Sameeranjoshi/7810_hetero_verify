#!/usr/bin/env python3
import os

# List of filenames
filenames = [
    "base_test_daniel.cu",
    "I_cpu_P_cpu_C_cpu.cu",
    "I_cpu_P_cpu_C_gpu.cu",
    "I_cpu_P_gpu_C_cpu.cu",
    "I_cpu_P_gpu_C_gpu.cu",
    "I_gpu_P_cpu_C_cpu.cu",
#    "I_gpu_P_cpu_C_gpu.cu",    // This goes infinite ? Why?
    "I_gpu_P_gpu_C_cpu.cu",
    "I_gpu_P_gpu_C_gpu.cu"
]

# Iterate through files
for filename in filenames:
    # Check if the file exists
    if os.path.isfile(filename):
        print(f"Running {filename}")
        # Compile CUDA file
        os.system(f"nvcc {filename} -arch=sm_70 -o exe/{filename}")
        # Run the executable
        os.system(f"./exe/{filename}")
    else:
        print(f"File {filename} not found")
