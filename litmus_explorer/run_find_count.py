import os
import subprocess


seq1 = 0
weak = 0
seq2 = 0
interleave = 0

# Take the input parameter from the user for the number of loop iterations
num_iterations = int(input("Enter the number of loop iterations: "))
count_of_tests = int(input("Enter the number of tests to run: "))
filename = str(input("Enter the filename to test: "))

# get smi
command = "nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -n 1"
output = subprocess.check_output(command, shell=True)
smi_value = output.decode().strip()       
version_number = smi_value.replace('.', '')
result = f"sm_{version_number}"         
# print(result)
os.system("mkdir -p exe/")
# Compile CUDA file
compilecmd = f"nvcc {filename} -arch={result} -o exe/{filename}"
# print(compilecmd)
os.system(compilecmd)

for _ in range(num_iterations):
    # Check if the file exists
    if os.path.isfile(filename):
        print(f"Running {filename}")

        # Run the executable
        runcmd = f"./exe/{filename} {count_of_tests}"
        output = subprocess.check_output(runcmd, shell=True)
        output_lines = output.decode().split('\n')
        seq1_local=0
        seq2_local=0
        weak_local=0
        interleave_local=0
        for line in output_lines:
            if "seq1" in line:
                seq1_local = int(line.split('=')[-1].strip())  
            elif "seq2" in line:
                seq2_local = int(line.split('=')[-1].strip())  
            elif "intlv" in line:
                interleave_local = int(line.split('=')[-1].strip())  
            elif "weak" in line:
                weak_local = int(line.split('=')[-1].strip())  
            print(line)

        # print(seq1_local)
        # print(seq2_local)
        # print(interleave_local)
        # print(weak_local)
        seq1 = seq1 + seq1_local
        seq2 = seq2 + seq2_local
        interleave = interleave + interleave_local
        weak = weak + weak_local
        
    else:
        print(f"File {filename} not found")

print(f"\n ----------------------------------------------\n")
print(f"\n Results after {num_iterations} many iterations")
print(f"seq1 (flag)=0; (data)=0;  = {seq1}")
print(f"seq2 (flag)=1; (data)=42; = {seq2}")
print(f"intlv(flag)=0; (data)=42; = {interleave}")
print(f"weak (flag)=1; (data)=0;  = {weak}")
