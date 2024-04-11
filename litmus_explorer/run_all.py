import os
import subprocess
import random

# List of file names
file_names = [
    "false_sharing_base.cu",
    "loop_in_each_thread.cu",
    "weak_prefetcher_full_array.cu",
    "weak_base_modify2.cu",
    "weak_base_modify4.cu",
    "false_sharing_struct.cu",
    "weak_base_modify3_1.cu",
    #"cache_line_stress.cu",
    #"fork.cu",
    #"weak_base.cu",
    #"weak_base_modify1.cu",
    #"weak_base_modify3.cu",
    #"weak_prefetcher_no_array.cu"
]

# Loop through each file name
for fname in file_names:
    seq1 = 0
    weak = 0
    seq2 = 0
    interleave = 0

    num_iterations = 2
    count_of_tests = 2
    filename=fname
    want_flags=random.randint(0, 1)
    nvcc_flags = f"--threads=1024 --extra-device-vectorization --use_fast_math -O0"

    # get smi
    command = "nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -n 1"
    output = subprocess.check_output(command, shell=True)
    smi_value = output.decode().strip()       
    version_number = smi_value.replace('.', '')
    result = f"sm_{version_number}"         
    
    os.system("mkdir -p exe/")
    if want_flags == 1:
        # Compile CUDA file
        compilecmd = f"nvcc {filename} -arch={result} {nvcc_flags} -o exe/{filename}"
    else:
        compilecmd = f"nvcc {filename} -arch={result} -o exe/{filename}"

    # print(compilecmd)
    os.system(compilecmd)

    # Define output file path
    output_file = f"exe/a_100{filename}.txt"

    with open(output_file, "w") as f:
        for _ in range(num_iterations):
            # Check if the file exists
            if os.path.isfile(f"exe/{filename}"):

                # Run the executable
                runcmd = f"./exe/{filename} {count_of_tests}"
                output = subprocess.check_output(runcmd, shell=True)
                output_lines = output.decode().split('\n')
                seq1_local = 0
                seq2_local = 0
                weak_local = 0
                interleave_local = 0
                for line in output_lines:
                    if "seq1" in line:
                        seq1_local = int(line.split('=')[-1].strip())  
                    elif "seq2" in line:
                        seq2_local = int(line.split('=')[-1].strip())  
                    elif "intlv" in line:
                        interleave_local = int(line.split('=')[-1].strip())  
                    elif "weak" in line:
                        weak_local = int(line.split('=')[-1].strip())  
                    #f.write(line + "\n")

                seq1 += seq1_local
                seq2 += seq2_local
                interleave += interleave_local
                weak += weak_local

            else:
                f.write(f"File {filename} not found\n")

    # Append the summary to the output file
    with open(output_file, "a") as f:
        f.write("\n ----------------------------------------------\n")
        f.write(f"Iterations: {num_iterations}\n")
        f.write(f"Test count: {count_of_tests}\n")
        f.write(f"filename: {filename}\n")
        f.write(f"want_flags: {want_flags}\n")
        f.write(f"flags: {nvcc_flags}\n")
        f.write(f"\n Results after {num_iterations} many iterations\n")
        f.write(f"seq1 (flag)=0; (data)=0;  = {seq1}\n")
        f.write(f"seq2 (flag)=1; (data)=42; = {seq2}\n")
        f.write(f"intlv(flag)=0; (data)=42; = {interleave}\n")
        f.write(f"weak (flag)=1; (data)=0;  = {weak}\n")

    print(f"Results for {filename} saved to {output_file}")