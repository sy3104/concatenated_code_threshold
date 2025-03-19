import sys
import os
import time
import concurrent.futures
import subprocess

def run_script(arg):
    try:
        start_time = time.time()
        print(f'case{arg[3]} starts: {arg}')
        result = subprocess.run(['python3'] + arg, capture_output=True, text=True, check=True)
        end_time = time.time()
        print(f'case{arg[3]} ends: elapsed time = {end_time-start_time} sec.')
    except subprocess.CalledProcessError as e:
        print(f'Error occurred while running {script_name}: {e}')
        print(e.stdout)
        print(e.stderr)

if __name__ == '__main__':
    if len(sys.argv)<2 or (sys.argv[1]=='hamming_code.py' and len(sys.argv) != 8) or (sys.argv[1]!='hamming_code.py' and len(sys.argv) != 7):
        print('parallel_execution.py [script_name] [num_shots] [starting_p] [num_p] [error_model] [parameter1] [parameter2 (if script_name==\'hamming_code.py\')]')
        exit(0)
    script_name = str(sys.argv[1])
    num_shots = int(sys.argv[2])
    starting_p = float(sys.argv[3])
    num_p = int(sys.argv[4])
    error_model = str(sys.argv[5])
    if script_name == 'hamming_code.py':
        parameter = [int(sys.argv[6]), int(sys.argv[7])]
    else:
        parameter = [int(sys.argv[6])]
    error_list = [10**(x/12)*starting_p for x in range(num_p)]

    if script_name=='hamming_code.py':
        output_file_name = f'../data/hamming_code/output_{os.path.splitext(os.path.basename(script_name))[0]}_{str(2**parameter[0]-1)}qubit_below_{str(2**parameter[1]-1)}qubit_error_model_{str(error_model)}_{num_shots}runs.json'
        arg_list = [[script_name, output_file_name, num_shots, i, error_list[i], error_model, parameter[0], parameter[1]] for i in range(num_p)]
    else:
        output_file_name = f'../data/underlying_code/output_{os.path.splitext(os.path.basename(script_name))[0]}_level{str(parameter[0])}_error_model_{str(error_model)}_{num_shots}runs.json'
        arg_list = [[script_name, output_file_name, num_shots, i, error_list[i], error_model, parameter[0]] for i in range(num_p)]

    for arg in arg_list:
        for i in range(len(arg)):
            arg[i] = str(arg[i])
    with open(output_file_name, 'w') as file:
        file.write('{\n')

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(run_script, arg_list)
    
    with open(output_file_name, 'r') as file:
        content = file.read()
    new_content = content[:-2]
    with open(output_file_name, 'w') as file:
        file.write(new_content)
        file.write('\n}')