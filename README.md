# concatenated_code_threshold

This code is accompanied to the paper Satoshi Yoshida, Shiro Tamiya, Hayata Yamasaki, "Concatenated codes, save qubits", [npj Quantum Information 11, 88 (2025)](https://doi.org/10.1038/s41534-025-01035-8), [arXiv:2402.09606](https://arxiv.org/abs/2402.09606).

## Contents
- `code`: Source code to evaluate the logical error rates of quantum Hamming codes, C4/C6 code, C4/Steane code, and concatenated Steane codes.
- `data`: Logical error rates obtained from the simulation (including those for the surface code).
- `plot`: Source for the threshold plot and space overhead estimation.
- `fig`: Graphs for the threshold plot and space overhead estimation.

## Usage
The required Python packages are described in `requirements.txt`.

The script `estimate_logical_error_rate.sh` reproduces the logical error rates of quantum Hamming codes, C4/C6 code, C4/Steane code, and concatenated Steane codes stored in `data`.

The script `estimate_threshold_and_plot.sh` reproduces the figures stored in `fig`.
