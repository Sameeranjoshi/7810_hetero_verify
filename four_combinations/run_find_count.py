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
        os.system(compilecmd)
        # Run the executable
        runcmd = f"./exe/{filename}"
        output = subprocess.check_output(runcmd, shell=True)
        output_str = output.decode()  # Print the captured output
        # print (output_str)
        # Parse the output and update counters
        result0 = int(output_str.split("result0=")[1].split()[0])
        result1 = int(output_str.split("result1=")[1].split()[0])

        if result0 == 0 and result1 == 0:
            seq1 += 1
        elif result0 == 0 and result1 == 1:
            weak += 1
        elif result0 == 42 and result1 == 1:
            seq2 += 1
        elif result0 == 42 and result1 == 0:
            interleave += 1
    else:
        print(f"File {filename} not found")


print(f"seq1 = {seq1}")
print(f"seq2 = {seq2}")
print(f"interleave = {interleave}")
print(f"weak = {weak}")
