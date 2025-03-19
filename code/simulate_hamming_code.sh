error_model_list=(a b c)
for error_model in ${error_model_list[@]}
do
    r_list=(3 4 5 6 7)
    for r in ${r_list[@]}
    do
        python3 parallel_execution.py hamming_code.py 1000000 1e-5 37 ${error_model} 3 ${r}
    done
    r_list=(4 5 6 7)
    for r in ${r_list[@]}
    do
        python3 parallel_execution.py hamming_code.py 1000000 5e-7 37 ${error_model} 4 ${r}
    done
    r_list=(5 6 7)
    for r in ${r_list[@]}
    do
        python3 parallel_execution.py hamming_code.py 1000000 1e-7 37 ${error_model} 5 ${r}
    done
    r_list=(6 7)
    for r in ${r_list[@]}
    do
        python3 parallel_execution.py hamming_code.py 1000000 5e-9 49 ${error_model} 6 ${r}
    done
    r_list=(7 8)
    for r in ${r_list[@]}
    do
        python3 parallel_execution.py hamming_code.py 1000000 5e-10 49 ${error_model} 7 ${r}
    done
done