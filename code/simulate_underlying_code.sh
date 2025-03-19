error_model_list=(a b c)
for error_model in ${error_model_list[@]}
do
    python3 parallel_execution.py c4c6.py 1000000 1e-5 25 ${error_model} 1
    python3 parallel_execution.py c4c6.py 1000000 1e-5 25 ${error_model} 2
    python3 parallel_execution.py concatenated_steane.py 10000000 1e-5 25 ${error_model} 1
    python3 parallel_execution.py concatenated_steane.py 10000000 1e-4 13 ${error_model} 2
    python3 parallel_execution.py c4steane.py 10000000 1e-4 13 ${error_model} 2
    python3 parallel_execution.py steane_conventional.py 1000000 1e-5 25 ${error_model} 1
done