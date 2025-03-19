import stim
import random
import numpy as np
import json
import sys
import math

def append_noisy_wait(circuit,list_loc,p,steps=1):
    N=7
    ew =  3/4*(1-(1-4/3*gamma)**steps)
    for i in list_loc:
        for j in range(N):
            circuit.append("DEPOLARIZE1", i+j, ew)

def reset(circuit,loc):
    N=7
    for i in range(N):
        circuit.append("M", loc+i)
    for i in range(N):
        circuit.append("CNOT", [stim.target_rec(i-N), loc+i])
        
def noisy_reset(circuit,loc,p):
    N=7
    for i in range(N):
        circuit.append("M", loc+i)
    for i in range(N):
        circuit.append("CNOT", [stim.target_rec(i-N), loc+i])
    for i in range(N):
        circuit.append("X_ERROR", loc+i, p)
        
def append_0prep(circuit,loc1):
    r=3
    T=3
    N=7

    reset(circuit,loc1)
        
    for i in range(r):
        circuit.append("H", loc1+(2**i-1))
    
    #Encode
    for t in range(T):
        for i in range(r):
            for n in range(N):
                if latin_rectangle[n][i]==t+1:
                    circuit.append("CNOT",[loc1+(2**i-1), loc1+n])

def append_noisy_0prep(circuit, loc1, loc2, p):
    global detector_now
    r=3
    T=3
    N=7

    for i in range(r):
        circuit.append("H", loc1+(2**i-1))
        circuit.append("H", loc2+(2**i-1))

    #Encode
    for t in range(T):
        wait_qubits = set(range(N))
        for i in range(r):
            for n in range(N):
                if latin_rectangle[n][i]==t+1:
                    circuit.append("CNOT",[loc1+(2**i-1), loc1+n])
                    circuit.append("DEPOLARIZE2", [loc1+(2**i-1), loc1+n], p)
                    circuit.append("CNOT",[loc2+(2**i-1), loc2+n])
                    circuit.append("DEPOLARIZE2", [loc2+(2**i-1), loc2+n], p)
                    wait_qubits.remove(2**i-1)
                    wait_qubits.remove(n)
        for n in wait_qubits:
            circuit.append("DEPOLARIZE1", loc1+n, p)
            circuit.append("DEPOLARIZE1", loc2+n, p)

    #Verification
    for i in range(N):
        circuit.append("CNOT",[loc1+i, loc2+i])
        circuit.append("DEPOLARIZE2", [loc1+i, loc2+i], p)
        circuit.append("X_ERROR", loc2+i, p)
        circuit.append("DEPOLARIZE1", loc1+i, p)
        circuit.append("M", loc2+i)
        circuit.append("DETECTOR", [stim.target_rec(-1)])

    detector_0prep_start=detector_now
    detector_now+=N
    detector_0prep_end=detector_now
    return detector_0prep_start, detector_0prep_end

def append_noisy_0prep_no_verification(circuit, loc, p):
    global detector_now
    r=3
    T=3
    N=7
        
    for i in range(r):
        circuit.append("H", loc+(2**i-1))

    #Encode
    for t in range(T):
        wait_qubits = set(range(N))
        for i in range(r):
            for n in range(N):
                if latin_rectangle[n][i]==t+1:
                    circuit.append("CNOT",[loc+(2**i-1), loc+n])
                    circuit.append("DEPOLARIZE2", [loc+(2**i-1), loc+n], p)
                    wait_qubits.remove(2**i-1)
                    wait_qubits.remove(n)
        for n in wait_qubits:
            circuit.append("DEPOLARIZE1", loc+n, p)

def post_selection(m):
    r=3
    N=7
    flag = True
    #Stabilizer check
    for i in range(r):
        check = np.sum(check_matrix[i,:]*m[:])%2
        if check != 0:
            flag = False
            break
    #Z check
    for k in range(N-2*r):
        check_Z = np.sum(logical_op[k][:]*m[:])%2
        if check_Z !=0:
            flag = False
            break
    return flag

def decode_measurement(m,correction=True):
    r=3
    N=7

    outcome=np.zeros(N-2*r)
    for a in range(N-2*r):
        outcome[a] = np.sum(m[:]*logical_op[a][:])%2
    
    if correction == True:
        syndrome = 0
        for i in range(r):
            e = np.sum(m[:]*check_matrix[i,:])%2
            syndrome += e*(2**i)
        for a in range(N-2*r):
            if syndrome > 0:
                outcome[a] = (outcome[a]+logical_op[a][int(syndrome)-1])%2
    
    return outcome
    
def append_h(circuit, loc):
    N=7
    for i in range(N):
        circuit.append("H", loc+i)
        
def append_noisy_h(circuit, loc, p):
    N=7
    for i in range(N):
        circuit.append("H", loc+i)
    for i in range(N):
        circuit.append("DEPOLARIZE1", loc+i, p)
    
def append_cnot(circuit, loc1, loc2):
    N=7
    for i in range(N):
        circuit.append("CNOT", [loc1+i, loc2+i])
        
def append_noisy_cnot(circuit, loc1, loc2, p):
    N=7
    for i in range(N):
        circuit.append("CNOT", [loc1+i, loc2+i])
    for i in range(N):
        circuit.append("DEPOLARIZE2", [loc1+i,loc2+i], p)

def append_m(circuit, loc):
    global detector_now
    N=7
    detector_m_start = detector_now
    for i in range(N):
        circuit.append("M", loc+i)
    for i in range(N):
        circuit.append("DETECTOR",stim.target_rec(i-N))
    detector_now += N
    detector_m_end = detector_now
    
    return detector_m_start, detector_m_end

def append_noisy_m(circuit, loc, p):
    global detector_now
    N=7
    detector_m_start = detector_now
    for i in range(N):
        circuit.append("X_ERROR", loc+i, p)
    for i in range(N):
        circuit.append("M", loc+i)
    for i in range(N):
        circuit.append("DETECTOR",stim.target_rec(i-N))
    detector_now += N
    detector_m_end = detector_now
    
    return detector_m_start, detector_m_end
                       
def append_noisy_ec(circuit, loc1, loc2, loc3, p, no_verification=-1):
    #no_verification: -1:all verificarion, 0: first no verification, 1: second no verification, 2: no verification at all
    global detector_now
    N=7
    detector_0prep = []
    
    noisy_reset(circuit,loc2,p)
    noisy_reset(circuit,loc3,p)
    
    if no_verification == -1 or no_verification == 1:
        list_detector_0prep = append_noisy_0prep(circuit,loc2,loc3, p)
        detector_0prep.append(list_detector_0prep)
    else:
        append_noisy_wait(circuit,range(loc1,loc1+N), p,T+2)
        append_noisy_0prep_no_verification(circuit,loc2, p)
    
    append_noisy_cnot(circuit,loc1,loc2, p)
    
    # X-measurement is written as ideal H+Z-measurement
    append_h(circuit,loc1)
    detector_Z = append_noisy_m(circuit,loc1, p)
    noisy_reset(circuit, loc1, p)
    noisy_reset(circuit, loc3, p)
    
    #|+>-prep is written as |0>-prep + ideal H
    if no_verification == -1 or no_verification == 0:
        list_detector_0prep = append_noisy_0prep(circuit,loc1,loc3, p)
        detector_0prep.append(list_detector_0prep)
    else:
        append_noisy_wait(circuit,range(loc3,loc3+N), p,T+2)
        append_noisy_wait(circuit,range(loc2,loc2+N), p,T+2)
        append_noisy_0prep_no_verification(circuit,loc1, p)
        
    append_h(circuit,loc1)
    
    append_noisy_wait(circuit,range(loc2,loc2+N), p,T+3)
    
    append_noisy_cnot(circuit,loc1,loc2, p)
    
    detector_X=append_noisy_m(circuit,loc2, p)
    append_noisy_wait(circuit,range(loc1,loc1+N), p)
    
    return detector_0prep, detector_Z, detector_X

def estimate_prob_0prep(p,num_shots):
    global detector_now
    N=7

    detector_now=0

    circuit = stim.Circuit()

    #noisy 0 state preparation
    noisy_reset(circuit,0,p)
    noisy_reset(circuit,N,p)
    detector_0prep = append_noisy_0prep(circuit, 0, N, p)

    sample = circuit.compile_detector_sampler().sample(shots=num_shots)
    
    #estimate prob_0prep
    sample=[x for x in sample if post_selection(x)]
    
    prob_0prep = len(sample)/num_shots
    prob_0prep_variance = (num_shots-len(sample))/num_shots**2

    return prob_0prep, prob_0prep_variance

def accept(x,list_detector_m,list_detector_X,list_detector_Z,Q=1):
    r=3
    N=7
    num_correction=2*Q
    X_propagate=[[1],[3],[1,3],[3]]
    Z_propagate=[[0],[2],[0],[0,2]]
    outcome = np.zeros([4,N-2*r])
    correction_x = np.zeros([num_correction,N-2*r])
    correction_z = np.zeros([num_correction,N-2*r])
    
    for i in range(4):
        outcome[i,:] = decode_measurement(x[list_detector_m[i][0]:list_detector_m[i][1]],True)
    for i in range(num_correction):
        correction_x[i,:] = decode_measurement(x[list_detector_X[i][0]:list_detector_X[i][1]],True)
        correction_z[i,:] = decode_measurement(x[list_detector_Z[i][0]:list_detector_Z[i][1]],True)
    
    for a in range(N-2*r):
        for i in range(num_correction):
            pos=(i+2)%4
            if correction_x[i,a]==1:
                outcome[X_propagate[pos],a] = (outcome[X_propagate[pos],a]+1)%2
            if correction_z[i,a]==1:
                outcome[Z_propagate[pos],a] = (outcome[Z_propagate[pos],a]+1)%2
                
    num_errors=0

    for a in range(N-2*r):
        flag=False
        for i in range(4):
            if outcome[i,a]==1:
                flag=True
        if flag:
            num_errors+=1
    return num_errors

def count_errors(x,list_detector_m,list_detector_X,list_detector_Z,Q=1):
    r=3
    N=7
    num_correction=2*Q
    X_propagate=[[1],[3],[1,3],[3]]
    Z_propagate=[[0],[2],[0],[0,2]]
    outcome = np.zeros([4,N-2*r])
    correction_x = np.zeros([num_correction,N-2*r])
    correction_z = np.zeros([num_correction,N-2*r])
    
    for i in range(4):
        outcome[i,:] = decode_measurement(x[list_detector_m[i][0]:list_detector_m[i][1]],True)
    for i in range(num_correction):
        correction_x[i,:] = decode_measurement(x[list_detector_X[i][0]:list_detector_X[i][1]],True)
        correction_z[i,:] = decode_measurement(x[list_detector_Z[i][0]:list_detector_Z[i][1]],True)
    
    for a in range(N-2*r):
        for i in range(num_correction):
            pos=(i+2)%4
            if correction_x[i,a]==1:
                outcome[X_propagate[pos],a] = (outcome[X_propagate[pos],a]+1)%2
            if correction_z[i,a]==1:
                outcome[Z_propagate[pos],a] = (outcome[Z_propagate[pos],a]+1)%2
                
    num_errors=[]

    for a in range(N-2*r):
        flag=False
        for i in range(4):
            if outcome[i,a]==1:
                flag=True
        if flag:
            num_errors.append(1)
        else:
            num_errors.append(0)
    return num_errors

def estimate_logical_cnot_error(p,num_shots):
    global detector_now
    r=3
    N=7
    
    Q=10

    list_detector_0prep=[]
    list_detector_X=[]
    list_detector_Z=[]
    list_detector_m=[]

    detector_now=0

    circuit = stim.Circuit()

    #Prepare ideal Bell pairs
    append_0prep(circuit,0)
    append_0prep(circuit,N)
    append_0prep(circuit,2*N)
    append_0prep(circuit,3*N)

    append_h(circuit,0)
    append_h(circuit,2*N)

    append_cnot(circuit,0,N)
    append_cnot(circuit,2*N,3*N)

    #noisy CNOT (or CZ)+EC gadget
    
    for q in range(Q):
        append_noisy_cnot(circuit, 0, 2*N, p)

        [detector_0prep, detector_Z, detector_X] = append_noisy_ec(circuit,0,4*N,5*N, p, -1)
        list_detector_0prep.extend(detector_0prep)
        list_detector_X.append(detector_X)
        list_detector_Z.append(detector_Z)

        [detector_0prep, detector_Z, detector_X] = append_noisy_ec(circuit,2*N,6*N,7*N, p, -1)
        list_detector_0prep.extend(detector_0prep)
        list_detector_X.append(detector_X)
        list_detector_Z.append(detector_Z)

    #ideal CNOT+ideal projective Bell measurement
    if Q%2==1:
        append_cnot(circuit, 0, 2*N)

    append_cnot(circuit,0,N)
    append_cnot(circuit,2*N,3*N)

    append_h(circuit,0)
    append_h(circuit,2*N)

    list_detector_m.append(append_m(circuit,0))
    list_detector_m.append(append_m(circuit,N))
    list_detector_m.append(append_m(circuit,2*N))
    list_detector_m.append(append_m(circuit,3*N))
    
    sample = circuit.compile_detector_sampler().sample(shots=num_shots)

    for a in list_detector_0prep:
        sample=[x for x in sample if post_selection(x[a[0]:a[1]])]

    num = len(sample)

    err = sum([sum(count_errors(x,list_detector_m,list_detector_X,list_detector_Z,Q)) for x in sample])
    logical_error = err/(num*Q*(N-2*r))
    variance = err/(num*Q*(N-2*r))**2

    return logical_error, variance

if __name__ == '__main__':
    output_file_name = str(sys.argv[1])
    num_shots = int(sys.argv[2])
    case_number = int(sys.argv[3])
    p = float(sys.argv[4])
    error_model = str(sys.argv[5])
    l = int(sys.argv[6])

    if error_model == 'a':
        gamma = p/10
    elif error_model == 'b':
        gamma = p/2
    elif error_model == 'c':
        gamma = p

    r=3
    N = int(2**r-1)
    r_prev = 3
    n_prev = 2**r_prev-1-2*r_prev
    Q = 10

    #----------Generate check matrix and logical operators----------#
    check_matrix = np.zeros([r, N])

    for i in range(r):
        for n in range(N):
            check_matrix[i,n] = ((n+1)//(2**i))%2

    H = check_matrix
    ordering = []
    for i in range(r):
        ordering.append(2**i-1)
    for n in range(N):
        if math.floor(math.log2(n+1))!=math.log2(n+1):
            ordering.append(n)
    H=H[:,ordering]
    H_prime=H[:,r:N]
    G=np.block([H_prime.transpose(),np.identity(N-r)])

    processed=0

    while processed<N-2*r:
        for i in range(processed,N-r):
            b = np.copy(G[i,:])
            if np.mod(np.floor(np.sum(b*b)),2)==1:
                G[i,:] = np.copy(G[processed,:])
                G[processed,:] = np.copy(b)
                processed = processed+1
                for j in range(processed,N-r):
                    G[j,:] = G[j,:]+np.sum(b[:]*G[j,:])*b[:]
                    G[j,:] = np.mod(G[j,:], 2)
    logical_op=np.zeros([N-2*r,N])

    for i in range(N-2*r):
        for j in range(N):
            logical_op[i,ordering[j]]=G[i,j]
    #----------Generate Latin rectangle----------#
    latin_rectangle = np.array([[0,0,0],[0,0,0],[3,1,0],[0,0,0],[2,0,3],[0,2,1],[1,3,2]])
    T = 3

    if l!=1:
        print('Warning: the simulation result is for level-1')

    logical_error, variance = estimate_logical_cnot_error(p,num_shots)
    with open(output_file_name, 'a') as file:
        output_data = {
            'physical_error': p,
            'logical_error': logical_error,
            'variance': variance
        }
        file.write('\"case' + str(case_number) + '\":' + json.dumps(output_data) +',\n')