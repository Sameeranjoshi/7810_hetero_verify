import os
import subprocess

# List of filenames
filenames = [
    "weak.cu",
]

seq1 = 0
weak = 0
seq2 = 0
interleave = 0
filename = "weak.cu"

# Take the input parameter from the user for the number of loop iterations
num_iterations = int(input("Enter the number of loop iterations: "))
count_of_tests = int(input("Enter the number of tests to run: "))

for _ in range(num_iterations):
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
        os.system("mkdir -p exe/")
        compilecmd = f"nvcc {filename} -arch={result} -o exe/{filename}"
        # print(compilecmd)
        os.system(compilecmd)
        # Run the executable
        runcmd = f"./exe/{filename} {count_of_tests}"
        output = subprocess.check_output(runcmd, shell=True)
        output_str = output.decode()  # Print the captured output
        print (output_str)
        # # Parse the output and update counters
        # flag = int(output_str.split("result0=")[1].split()[0])   # result0 = data
        # data = int(output_str.split("result1=")[1].split()[0])   # result1 = flag
        
        # #histogram logic for top level loop
        # # r0=flag, r1=data
        # if flag == 0 and data == 0:
        #     seq1 += 1   # t1->t2
        # elif flag == 1 and data == 42:
        #     seq2 += 1   # t2-t1
        # elif flag == 0 and data == 42:
        #     interleave += 1
        # elif flag == 1 and data == 0:
        #     weak += 1
        
    else:
        print(f"File {filename} not found")

print(f"\n Results after {num_iterations} many iterations")
print(f"seq1 (flag)=0; (data)=0;  = {seq1}")
print(f"seq2 (flag)=1; (data)=42; = {seq2}")
print(f"intlv(flag)=0; (data)=42; = {interleave}")
print(f"weak (flag)=1; (data)=0;  = {weak}")
