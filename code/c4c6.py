import stim
import random
import numpy as np
import json
import sys

def append_noisy_wait(circuit,list_loc,N,p,steps=1):
    ew =  3/4*(1-(1-4/3*gamma)**steps)
    for i in list_loc:
        for j in range(N):
            circuit.append("DEPOLARIZE1", i+j, ew)

def reset(circuit,loc,N):
    for i in range(N):
        circuit.append("M", loc+i)
    for i in range(N):
        circuit.append("CNOT", [stim.target_rec(i-N), loc+i])
        
def noisy_reset(circuit,loc,N,p):
    for i in range(N):
        circuit.append("M", loc+i)
    for i in range(N):
        circuit.append("CNOT", [stim.target_rec(i-N), loc+i])
    for i in range(N):
        circuit.append("X_ERROR", loc+i, p)

def append_h(circuit, loc, N_prev, N_now):
    if N_prev==1:
        for i in range(N_now):
            circuit.append("H", loc+i)
        if N_now==4:
            circuit.append("SWAP", [loc+1, loc+2])
    else:
        for i in range(N_now):
            append_h(circuit, (loc+i)*N_prev, 1, N_prev)
        if N_now==3:
            for i in range(3):
                circuit.append("SWAP", [(loc+i)*N_prev+1, (loc+i)*N_prev+3])
                circuit.append("SWAP", [(loc+i)*N_prev+1, (loc+i)*N_prev+2])
def append_transversal_h(circuit, loc, N_prev, N_now):
    if N_prev==1:
        for i in range(N_now):
            circuit.append("H", loc+i)
    else:
        for i in range(N_now):
            append_h(circuit, (loc+i)*N_prev, 1, N_prev)
            
def append_swap(circuit, loc1, loc2, N_prev, N_now):
    for i in range(N_prev*N_now):
        circuit.append("SWAP", [N_prev*loc1+i, N_prev*loc2+i])

    
def append_cnot(circuit, loc1, loc2, N_prev, N_now):
    N=N_prev*N_now
    for i in range(N):
        circuit.append("CNOT", [loc1*N_prev+i, loc2*N_prev+i])
        
def append_noisy_cnot(circuit, loc1, loc2, N_prev, N_now, p):
    N=N_prev*N_now
    for i in range(N):
        circuit.append("CNOT", [loc1*N_prev+i, loc2*N_prev+i])
    for i in range(N):
        circuit.append("DEPOLARIZE2", [loc1*N_prev+i,loc2*N_prev+i], p)

def append_m(circuit, loc, N_prev, N_now):
    global detector_now
    
    if N_prev==1:
        for i in range(N_now):
            circuit.append("M", loc+i)
        for i in range(N_now):
            circuit.append("DETECTOR",stim.target_rec(i-N_now))
        detector_m=[detector_now, detector_now+N_now]
        detector_now+=N_now
    else:
        detector_m=[append_m(circuit, (loc+i)*N_prev, 1, N_prev) for i in range(N_now)]
    
    return detector_m

def append_noisy_m(circuit, loc,N_prev,N_now, p):
    global detector_now
    
    if N_prev==1:
        for i in range(N_now):
            circuit.append("X_ERROR", loc+i, p)
        for i in range(N_now):
            circuit.append("M", loc+i)
        for i in range(N_now):
            circuit.append("DETECTOR",stim.target_rec(i-N_now))
        detector_m=[detector_now, detector_now+N_now]
        detector_now+=N_now
    else:
        detector_m=[append_noisy_m(circuit, (loc+i)*N_prev, 1, N_prev, p) for i in range(N_now)]
    
    return detector_m
        
def append_0prep(circuit, loc1, N_prev, N_now):
    if N_prev==1:
        reset(circuit,loc1,N_now)
    else:
        for i in range(N_now):
            append_0prep(circuit, (loc1+i)*N_prev, 1, N_prev)

    if N_now==N_c4:
        append_h(circuit,(loc1+0)*N_prev,1,N_prev)
        append_cnot(circuit, (loc1+0)*N_prev, (loc1+1)*N_prev,1,N_prev)
        append_cnot(circuit, (loc1+0)*N_prev, (loc1+2)*N_prev,1,N_prev)
        append_cnot(circuit, (loc1+0)*N_prev, (loc1+3)*N_prev,1,N_prev)

def append_noisy_0prep(circuit, loc1, loc2, N_prev, N_now, p):
    global detector_now
    
    if N_now==N_c6:
        n_now=3
    else:
        n_now=N_now
    
    if N_prev==1:
        noisy_reset(circuit,loc1,N_now,p)
        noisy_reset(circuit,loc2,N_now,p)
        detector_0prep = []
    else:
        detector_0prep = []
        for i in range(n_now):
            detector_0prep.append(append_noisy_0prep(circuit, (loc1+i)*N_prev, (loc1+n_now+i)*N_prev, 1, N_prev, p))
        for i in range(n_now):
            detector_0prep.append(append_noisy_0prep(circuit, (loc2+i)*N_prev, (loc2+n_now+i)*N_prev, 1, N_prev, p))
    if N_now==N_c4:
        for i in range(4):
            circuit.append("H", loc2+i)
        
        for i in range(N_now):
            append_noisy_cnot(circuit, (loc2+i)*N_prev, (loc1+i)*N_prev, 1, N_prev, p)
        
        for i in range(N_now):
            append_noisy_cnot(circuit, (loc1+i)*N_prev, (loc2+(i+1)%N_now)*N_prev, 1, N_prev, p)

        detector_0prep.append(append_noisy_m(circuit,loc2,N_prev,N_now,p))
        for i in range(N_now-1):
            for j in range(N_now-1):
                if j>=i:
                    append_cnot(circuit, (loc2+i)*N_prev, (loc1+j)*N_prev,1,N_prev)

        return detector_0prep
    elif N_now==N_c6:
        for i in range(3):
            append_h(circuit,(loc2+i)*N_prev,1,N_c4)
            append_noisy_cnot(circuit,(loc2+i)*N_prev,(loc1+i)*N_prev,1,N_c4,p)
        for i in range(3):
            append_noisy_cnot(circuit,(loc1+i)*N_prev,(loc2+((i+1)%3))*N_prev,1,N_c4,p)
        detector_0prep.append(append_noisy_m(circuit,loc2,N_prev,3,p))
        
        append_cnot(circuit,(loc2+0)*N_prev, (loc1+0)*N_prev,1,N_prev)
        append_cnot(circuit,(loc2+0)*N_prev, (loc1+1)*N_prev,1,N_prev)
        append_cnot(circuit,(loc2+1)*N_prev, (loc1+1)*N_prev,1,N_prev)
        
        circuit.append("SWAP", [(loc1+1)*N_c4+1, (loc1+1)*N_c4+2])
        circuit.append("SWAP", [(loc1+1)*N_c4+1, (loc1+1)*N_c4+3])
        
        circuit.append("SWAP", [(loc1+2)*N_c4+1, (loc1+2)*N_c4+3])
        circuit.append("SWAP", [(loc1+2)*N_c4+1, (loc1+2)*N_c4+2])
        
        return detector_0prep

def decode_measurement_c4(m, m_type='x'):
    #Hard-decision decoder for c4
    if (int(m[0])+int(m[1])+int(m[2])+int(m[3]))%2==1:
        outcome=[-1,-1]
    else:
        outcome=[(int(m[0])+int(m[2]))%2,(int(m[2])+int(m[3]))%2]
    
    return outcome

def decode_measurement_c6(m, m_type='x'):
    if len(m)==3:
        M=[m[0][0],m[0][1],m[1][0],m[1][1],m[2][0],m[2][1]]
    else:
        M=m
    if [M[0],M[2],M[4]].count(-1)>1:
        outcome=[-1,-1]
    elif M[0]==-1:
        outcome=[(M[2]+M[3]+M[5])%2, (M[3]+M[4])%2]
    elif M[2]==-1:
        outcome=[(M[1]+M[4]+M[5])%2, (M[0]+M[5])%2]
    elif M[4]==-1:
        outcome=[(M[0]+M[1]+M[3])%2, (M[1]+M[2])%2]
    else:
        if (M[0]+M[1]+M[2]+M[5])%2==1 or (M[0]+M[3]+M[4]+M[5])%2==1:
            outcome=[-1,-1]
        else:
            outcome=[(M[2]+M[3]+M[5])%2, (M[3]+M[4])%2]
    return outcome

def append_noisy_ec(circuit, loc1, loc2, loc3, loc4, N_prev, N_now, p, no_verification=-1):
    global detector_now, no_ec
    detector_0prep = []
    detector_Z=[]
    detector_X=[]
    
    if N_now==1:
        return None
    
    if N_now==N_c6:
        n_now=3
    else:
        n_now=N_now

    detector_0prep.extend(append_noisy_0prep(circuit, loc2, loc4, N_prev, N_now, p))
    detector_0prep.extend(append_noisy_0prep(circuit, loc3, loc4, N_prev, N_now, p))
    append_h(circuit,loc2,N_prev,n_now)
    append_noisy_cnot(circuit,loc2,loc3,N_prev,n_now, p)
    
    if N_now==N_c6 and no_ec==False:#Error detecting teleportation
        for i in range(3):
            detector_0prep_c4, detector_Z_c4, detector_X_c4 = append_noisy_ec(circuit,(loc2+i)*N_prev,(loc4+0)*N_prev,(loc4+1)*N_prev,(loc4+2)*N_prev,1,N_prev,p)
            append_h(circuit,(loc2+i)*N_prev,1,N_prev)
            append_cnot(circuit,(loc4+1)*N_prev,(loc2+i)*N_prev,1,N_prev)
            append_h(circuit,(loc2+i)*N_prev,1,N_prev)
            append_cnot(circuit,(loc4+0)*N_prev,(loc2+i)*N_prev,1,N_prev)
            detector_0prep.extend(detector_0prep_c4)
            detector_Z.append(detector_Z_c4)
            detector_X.append(detector_X_c4)
        for i in range(3):
            detector_0prep_c4, detector_Z_c4, detector_X_c4 = append_noisy_ec(circuit,(loc3+i)*N_prev,(loc4+0)*N_prev,(loc4+1)*N_prev,(loc4+2)*N_prev,1,N_prev,p)
            append_h(circuit,(loc3+i)*N_prev,1,N_prev)
            append_cnot(circuit,(loc4+1)*N_prev,(loc3+i)*N_prev,1,N_prev)
            append_h(circuit,(loc3+i)*N_prev,1,N_prev)
            append_cnot(circuit,(loc4+0)*N_prev,(loc3+i)*N_prev,1,N_prev)
            detector_0prep.extend(detector_0prep_c4)
            detector_Z.append(detector_Z_c4)
            detector_X.append(detector_X_c4)
    
    append_noisy_cnot(circuit,loc1,loc2,N_prev,n_now, p)
    
    append_h(circuit,loc1,N_prev,n_now)
    
    detector_Z.append(append_noisy_m(circuit,loc1,N_prev,n_now, p))
    detector_X.append(append_noisy_m(circuit,loc2,N_prev,n_now, p))
    
    append_swap(circuit,loc1,loc3,N_prev,n_now)
    
    return detector_0prep, detector_Z, detector_X

def decode_ec_hd(x, detector_X, detector_Z):
    mx=[[]]*3
    mz=[[]]*3
    
    for i in range(3):
        detx=detector_X[i]
        detz=detector_Z[i]
        mx[i]=decode_measurement_c4(x[detx[0]:detx[1]], 'x')
        mz[i]=decode_measurement_c4(x[detz[0]:detz[1]], 'z')
    
    correction_x = decode_measurement_c6(mx, 'x')
    correction_z = decode_measurement_c6(mz, 'z')
    
    return correction_x, correction_z

def decode_m_hd(x, detector_m):
    m=[[]]*3
    for i in range(3):
        det=detector_m[i]
        m[i]=decode_measurement_c4(x[det[0]:det[1]])
    
    outcome = decode_measurement_c6(m, 'x')
    
    return outcome

def post_selection_c4(x,detector_0prep):
    if sum(x[detector_0prep[0]:detector_0prep[1]])%2==0:
        return True
    else:
        return False

def post_selection_c6(x,detector_0prep):
    outcome=[[]]*3
    for i in range(3):
        outcome[i]=decode_measurement_c4(x[detector_0prep[i][0]:detector_0prep[i][1]])
        if outcome[i][0]==-1:
            return False
    if (outcome[0][0]+outcome[1][0]+outcome[2][0])%2==0 and (outcome[0][1]+outcome[1][1]+outcome[2][1])%2==0:
        return True
    else:
        return False

def accept(x,list_detector_m,list_detector_X,list_detector_Z,Q=1):
    global no_ec
    num_correction=2*Q

    X_propagate=[[1],[3]]
    Z_propagate=[[0],[2]]
    outcome = np.zeros([4,2])
    correction_x = np.zeros([num_correction,2])
    correction_z = np.zeros([num_correction,2])
    
    # Post selection at error detecting telportation
    if no_ec==False:
        for i in range(num_correction):
            for a in range(6):
                if decode_measurement_c4(x[list_detector_X[i][a][0][0]:list_detector_X[i][a][0][1]])[0]==-1:
                    return -1
                if decode_measurement_c4(x[list_detector_Z[i][a][0][0]:list_detector_Z[i][a][0][1]])[0]==-1:
                    return -1
    
    for i in range(num_correction):
        if no_ec==False:
            correction_x[i,:], correction_z[i,:] = decode_ec_hd(x, list_detector_X[i][6], list_detector_Z[i][6])
        else:
            correction_x[i,:], correction_z[i,:] = decode_ec_hd(x, list_detector_X[i][0], list_detector_Z[i][0])
    
    for i in range(4):
        outcome[i,:] = decode_m_hd(x, list_detector_m[i])

    for a in range(2):
        for i in range(num_correction):
            pos = i%2
            for x_prop in X_propagate[pos]:
                if outcome[x_prop,a]==-1:
                    continue
                if correction_x[i,a]==1:
                    outcome[x_prop,a] = (outcome[x_prop,a]+1)%2
                if correction_x[i,a]==-1:
                    outcome[x_prop,a] = -1
            for z_prop in Z_propagate[pos]:
                if outcome[z_prop,a]==-1:
                    continue
                if correction_z[i,a]==1:
                    outcome[z_prop,a] = (outcome[z_prop,a]+1)%2
                if correction_z[i,a]==-1:
                    outcome[z_prop,a] = -1
    
    num_errors = 0
    for a in range(2):
        flag=1
        for i in range(4):
            if outcome[i,0]==1:
                flag=0
            if outcome[i,0]==-1:
                flag*=0.5
        num_errors += (1-flag)
    return num_errors

def accept_c4(x,list_detector_m,list_detector_X,list_detector_Z,Q=1):
    num_correction=2*Q

    X_propagate=[[1],[3]]
    Z_propagate=[[0],[2]]
    outcome = np.zeros([4,2])
    correction_x = np.zeros([num_correction,2])
    correction_z = np.zeros([num_correction,2])
    
    for i in range(4):
        outcome[i,:] = decode_measurement_c4(x[list_detector_m[i][0]:list_detector_m[i][1]], 'x')

    for i in range(num_correction):
        correction_x[i,:] = decode_measurement_c4(x[list_detector_X[i][0][0]:list_detector_X[i][0][1]], 'x')
        correction_z[i,:] = decode_measurement_c4(x[list_detector_Z[i][0][0]:list_detector_Z[i][0][1]], 'z')


    for a in range(2):
        for i in range(num_correction):
            pos = i%2
            for x_prop in X_propagate[pos]:
                if outcome[x_prop,a]==-1:
                    continue
                if correction_x[i,a]==1:
                    outcome[x_prop,a] = (outcome[x_prop,a]+1)%2
                if correction_x[i,a]==-1:
                    outcome[x_prop,a] = -1
            for z_prop in Z_propagate[pos]:
                if outcome[z_prop,a]==-1:
                    continue
                if correction_z[i,a]==1:
                    outcome[z_prop,a] = (outcome[z_prop,a]+1)%2
                if correction_z[i,a]==-1:
                    outcome[z_prop,a] = -1

    num_errors=0
    for a in range(2):
        flag=1
        for i in range(4):
            if outcome[i,0]==1:
                flag=0
            if outcome[i,0]==-1:
                flag*=0.5
        num_errors+= (1-flag)
    return num_errors

def estimate_prob_0prep_l1(p,num_shots):
    global detector_now, no_ec

    no_ec=False

    N_prev = 1
    N_now = N_c4

    list_detector_0prep=[]
    list_detector_X=[]
    list_detector_Z=[]
    list_detector_m=[]

    detector_now=0

    circuit = stim.Circuit()

    #Prepare ideal Bell pairs
    detector_0prep = append_noisy_0prep(circuit,0,N_now,N_prev,N_now,p)
    
    sample = circuit.compile_detector_sampler().sample(shots=num_shots)
    
    sample = [x for x in sample if post_selection_c4(x,detector_0prep[0])]
    
    num=len(sample)
    
    post_selection_prob_l1 = 1-num/num_shots
    variance_post_selection_prob_l1 = (num_shots-num)/(num_shots**2)

    return post_selection_prob_l1, variance_post_selection_prob_l1

def estimate_prob_0prep_l2(p,num_shots):
    global detector_now, no_ec

    no_ec=False

    N_prev=N_c4
    N_now=N_c6
    n_now=3

    list_detector_0prep=[]
    list_detector_X=[]
    list_detector_Z=[]
    list_detector_m=[]

    detector_now=0

    circuit = stim.Circuit()

    #Prepare ideal Bell pairs
    detector_0prep = append_noisy_0prep(circuit,0,N_now,N_prev,N_now,p)
    
    sample = circuit.compile_detector_sampler().sample(shots=num_shots)
    
    for a in range(6):
        sample = [x for x in sample if post_selection_c4(x,detector_0prep[a][0])]
    
    num1=len(sample)
    
    sample = [x for x in sample if post_selection_c6(x,detector_0prep[6])]
    
    num2=len(sample)
    
    post_selection_prob_l2 = 1-num2/num1
    variance_post_selection_prob_l2 = (num1-num2)/(num1**2)

    return post_selection_prob_l2, variance_post_selection_prob_l2

def estimate_prob_edt_l2(p,num_shots):
    global detector_now, no_ec

    no_ec=False

    N_prev=N_c4
    N_now=N_c6
    n_now=3

    list_detector_0prep=[]
    list_detector_X=[]
    list_detector_Z=[]
    list_detector_m=[]

    detector_now=0

    circuit = stim.Circuit()

    #Prepare ideal Bell pairs
    detector_0prep1 = append_noisy_0prep(circuit,0,2*N_now,N_prev,N_now,p)
    detector_0prep2 = append_noisy_0prep(circuit,1,2*N_now,N_prev,N_now,p)
    
    detector_0prep_c4=[[]]*3
    detector_Z_c4=[[]]*3
    detector_X_c4= [[]]*3
    
    for a in range(3):
        detector_0prep_c4[a], detector_Z_c4[a], detector_X_c4[a] = append_noisy_ec(circuit,(0)*N_prev,(2*N_now)*N_prev,(2*N_now+1)*N_prev,(2*N_now+2)*N_prev,1,N_prev,p)
    
    sample = circuit.compile_detector_sampler().sample(shots=num_shots)
    
    for a in range(6):
        sample = [x for x in sample if post_selection_c4(x,detector_0prep1[a][0])]
        sample = [x for x in sample if post_selection_c4(x,detector_0prep2[a][0])]
    
    for a in range(3):
        sample = [x for x in sample if post_selection_c4(x,detector_0prep_c4[a][0])]
    
    sample = [x for x in sample if post_selection_c6(x,detector_0prep1[6])]
    sample = [x for x in sample if post_selection_c6(x,detector_0prep2[6])]
    
    num1=len(sample)
    
    for a in range(3):
        sample = [x for x in sample if post_selection_c4(x,detector_X_c4[a][0])]
        sample = [x for x in sample if post_selection_c4(x,detector_Z_c4[a][0])]
    
    num2=len(sample)
    
    edt_prob_l2 = 1-num2/num1
    variance_edt_prob_l2 = (num1-num2)/(num1**2)

    return edt_prob_l2, variance_edt_prob_l2

# Estimate the logical error rate of level-1
def estimate_logical_cnot_error_l1(p,num_shots):
    global detector_now, no_ec

    no_ec=False

    N_prev=1
    N_now=N_c4
    
    Q=10

    list_detector_0prep=[]
    list_detector_X=[]
    list_detector_Z=[]
    list_detector_m=[]

    detector_now=0

    circuit = stim.Circuit()

    #Prepare ideal Bell pairs
    append_0prep(circuit,0,N_prev,N_now)
    append_0prep(circuit,8,N_prev,N_now)
    append_0prep(circuit,16,N_prev,N_now)
    append_0prep(circuit,24,N_prev,N_now)

    append_h(circuit,0,N_prev,N_now)
    append_h(circuit,16,N_prev,N_now)

    append_cnot(circuit,0,8,N_prev,N_now)
    append_cnot(circuit,16,24,N_prev,N_now)

    #error-free CNOT + noisy CNOT (or CZ)+EC gadget

    for q in range(Q):
        append_cnot(circuit, 0, 16,N_prev,N_now)
        
        append_noisy_cnot(circuit, 0, 16,N_prev,N_now, p)

        [detector_0prep, detector_Z, detector_X] = append_noisy_ec(circuit,0,32,40,48,N_prev,N_now, p, -1)
        list_detector_0prep.extend(detector_0prep)
        list_detector_X.append(detector_X)
        list_detector_Z.append(detector_Z)

        [detector_0prep, detector_Z, detector_X] = append_noisy_ec(circuit,16,32,40,48,N_prev,N_now, p, -1)
        list_detector_0prep.extend(detector_0prep)
        list_detector_X.append(detector_X)
        list_detector_Z.append(detector_Z)

    append_cnot(circuit,0,8,N_prev,N_now)
    append_cnot(circuit,16,24,N_prev,N_now)

    append_h(circuit,0,N_prev,N_now)
    append_h(circuit,16,N_prev,N_now)

    list_detector_m.append(append_m(circuit,0,N_prev,N_now))
    list_detector_m.append(append_m(circuit,8,N_prev,N_now))
    list_detector_m.append(append_m(circuit,16,N_prev,N_now))
    list_detector_m.append(append_m(circuit,24,N_prev,N_now))
    
    sample = circuit.compile_detector_sampler().sample(shots=num_shots)

    for a in list_detector_0prep:
        sample=[x for x in sample if post_selection_c4(x,a)]
    
    num=len(sample)

    err = sum([accept_c4(x,list_detector_m,list_detector_X,list_detector_Z,Q) for x in sample])
    
    logical_error_l1=err/(num*Q*2)
    variance_l1=err/(num*Q*2)**2

    return logical_error_l1, variance_l1

# Estimate the logical error rate of level-2 with post-selection
def estimate_logical_cnot_error_l2(p,num_shots):
    global detector_now, no_ec

    no_ec=False

    N_prev=N_c4
    N_now=N_c6
    n_now=3
    
    Q=10

    list_detector_0prep=[]
    list_detector_X=[]
    list_detector_Z=[]
    list_detector_m=[]

    detector_now=0

    circuit = stim.Circuit()

    #Prepare ideal Bell pairs
    append_noisy_0prep(circuit,0,24,N_prev,N_now,0)
    append_noisy_0prep(circuit,6,24,N_prev,N_now,0)
    append_noisy_0prep(circuit,12,24,N_prev,N_now,0)
    append_noisy_0prep(circuit,18,24,N_prev,N_now,0)

    append_h(circuit,0,N_prev,n_now)
    append_h(circuit,12,N_prev,n_now)

    append_cnot(circuit,0,6,N_prev,n_now)
    append_cnot(circuit,12,18,N_prev,n_now)

    #error-free CNOT + noisy CNOT (or CZ)+EC gadget

    for q in range(Q):
        append_cnot(circuit, 0, 12, N_prev,n_now)
        
        append_noisy_cnot(circuit, 0, 12, N_prev,n_now, p)

        [detector_0prep, detector_Z, detector_X] = append_noisy_ec(circuit,0,24,30,36,N_prev,N_now, p, -1)
        list_detector_0prep.extend(detector_0prep)
        list_detector_X.append(detector_X)
        list_detector_Z.append(detector_Z)

        [detector_0prep, detector_Z, detector_X] = append_noisy_ec(circuit,12,24,30,36,N_prev,N_now, p, -1)
        list_detector_0prep.extend(detector_0prep)
        list_detector_X.append(detector_X)
        list_detector_Z.append(detector_Z)

    append_cnot(circuit,0,6,N_prev,n_now)
    append_cnot(circuit,12,18,N_prev,n_now)

    append_h(circuit,0,N_prev,n_now)
    append_h(circuit,12,N_prev,n_now)

    list_detector_m.append(append_m(circuit,0,N_prev,n_now))
    list_detector_m.append(append_m(circuit,6,N_prev,n_now))
    list_detector_m.append(append_m(circuit,12,N_prev,n_now))
    list_detector_m.append(append_m(circuit,18,N_prev,n_now))
    
    num=0
    err=0
    
    sample = circuit.compile_detector_sampler().sample(shots=num_shots)

    for a in list_detector_0prep:
        if len(a)==1:
            sample=[x for x in sample if post_selection_c4(x,a[0])]
        elif len(a)==2:
            sample=[x for x in sample if post_selection_c4(x,a)]
        elif len(a)==3:
            sample=[x for x in sample if post_selection_c6(x,a)]

    sample=[x for x in sample if accept(x,list_detector_m,list_detector_X,list_detector_Z,Q)!=-1]

    num+=len(sample)

    err+= sum([accept(x,list_detector_m,list_detector_X,list_detector_Z,Q) for x in sample])
    
    logical_error_l2=err/(num*Q*2)
    variance_l2=err/(num*Q*2)**2

    return logical_error_l2, variance_l2

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

    N_c4=4
    N_c6=6

    check_matrix = [[1, 1, 1, 1]]
    if l!=1 and l!=2:
        print('l should be 1 or 2!')
    else:
        if l==1:
            logical_error, variance = estimate_logical_cnot_error_l1(p,num_shots)
        elif l==2:
            logical_error, variance = estimate_logical_cnot_error_l2(p,num_shots)
        with open(output_file_name, 'a') as file:
            output_data = {
                'physical_error': p,
                'logical_error': logical_error,
                'variance': variance
            }
            file.write('\"case' + str(case_number) + '\":' + json.dumps(output_data) +',\n')
        