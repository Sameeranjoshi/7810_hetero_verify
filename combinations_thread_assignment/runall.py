#!/usr/bin/env python3
import os
import subprocess

# List of filenames
filenames = [
    "base_test_daniel.cu",
    "I_cpu_P_cpu_C_cpu.cu",
    "I_cpu_P_cpu_C_gpu.cu",
    "I_cpu_P_gpu_C_cpu.cu",
    "I_cpu_P_gpu_C_gpu.cu",
    "I_gpu_P_cpu_C_cpu.cu",
    "I_gpu_P_cpu_C_gpu.cu",    # This goes infinite ? Why?
    "I_gpu_P_gpu_C_cpu.cu",
    "I_gpu_P_gpu_C_gpu.cu"
]

# Iterate through files
for filename in filenames:
    # Check if the file exists
    if os.path.isfile(filename):
        print(f"Running {filename}")
        # get smi
        command = "nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -n 1"
        output = subprocess.check_output(command, shell=True)
        smi_value = output.decode().strip()       
        version_number = smi_value.replace('.', '')
        result = f"sm_{version_number}"         
        # print(result)
        # Compile CUDA file
        compilecmd = f"nvcc {filename} -arch={result} -o exe/{filename}"
        os.system(compilecmd)
        # Run the executable
        runcmd = f"./exe/{filename}"
        os.system(runcmd)
    else:
        print(f"File {filename} not found")
