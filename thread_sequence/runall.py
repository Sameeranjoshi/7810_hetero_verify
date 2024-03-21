#!/usr/bin/env python3
import os
import subprocess

# List of filenames
filenames = [
"cpu_C_I_P.cu",
"cpu_I_C_P.cu",
"cpu_P_C_I.cu",
"gpu_C_I_P.cu",
"gpu_I_C_P.cu",
"gpu_P_C_I.cu",
"cpu_C_P_I.cu",
"cpu_I_P_C.cu",
"cpu_P_I_C.cu",
"gpu_C_P_I.cu",
"gpu_I_P_C.cu",
"gpu_P_I_C.cu"
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
