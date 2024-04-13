import pandas as pd
import os
import re

# Function to extract data from a text file
def extract_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = {}
        for i, line in enumerate(lines):
            if "Results after" in line:
                iterations = int(re.findall(r'\d+', line)[0])
                data['Iterations'] = iterations
                for j in range(i+1, len(lines)):
                    if '=' in lines[j]:
                        parts = lines[j].split(';')
                        print
                        for part in parts:
                            key, value = part.split('=')
                            data[key.strip()] = int(value.strip())
                    else:
                        break
                break
    return data

# Function to process all files in a directory
def process_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            data.append(extract_data(file_path))
    return data

# Main function
def main():
    directory = '/uufs/chpc.utah.edu/common/home/u1418973/other/7810_project/7810_hetero_verify/results/a100'  # Change this to the directory where your text files are stored
    data = process_files(directory)
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    # Write DataFrame to Excel file
    excel_file = 'output.xlsx'
    df.to_excel(excel_file, index=False)
    print("Excel file '{}' has been created.".format(excel_file))

if __name__ == "__main__":
    main()
