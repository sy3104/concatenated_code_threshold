import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ptick
import json

color1=[1,75/255,0]#red
color2=[0,90/255,1]#blue
color3=[3/255,175/255,122/255]#green
color4=[77/255,196/255,1]#cyan
color5=[246/255,170/255,0]#orange
color6=[221/255,204/255,119/255]#sand
color7=[0,0,0]#black
color8=[153/255,0,153/255]#purple
color9=[132/255, 145/255, 158/255]#gray
color10=[68/255, 170/255, 153/255]#teal
color11=[153/255, 153/255, 51/255]#olive
color12=[136/255, 34/255, 85/255]#wine
color13=[204/255, 102/255, 119/255]#rose

def fibonacci(a):
    if a==0:
        return 1
    elif a==1:
        return 1
    else:
        return fibonacci(a-2)+fibonacci(a-1)

def calculate_ancilla_overhead(a):
    return 2**(a+1)-1

def c4c6(x,a):
    global A, B
    if a==0:
        return x
    else:
        return A*(B*x)**fibonacci(a)

def surface(x,a):
    global A_surface, B_surface
    if a==0:
        return x
    else:
        return A_surface*(B_surface*x)**math.floor((a+1)/2)

def surface_overhead(a):
    if a==0:
        return 1
    else:
        return a**2

def c4c6_overhead(a):
    if a==0:
        return 1
    else:
        return (4/2)*(6/2)**(a-1)
    
def steane(x,i):
    global A_steane
    if i==1:
        return A_steane[0]*x**2
    if i==2:
        return A_steane[1]*x**4
    if i==3:
        return A_steane[0]*(A_steane[1]*x**4)**2
    if i==4:
        return A_steane[1]*(A_steane[1]*x**4)**4
    if i==5:
        return A_steane[0]*(A_steane[1]*(A_steane[1]*x**4)**4)**2
    if i==6:
        return A_steane[1]*(A_steane[1]*(A_steane[1]*x**4)**4)**4
    if i==7:
        return A_steane[0]*(A_steane[1]*(A_steane[1]*(A_steane[1]*x**4)**4)**4)**2
    if i==8:
        return A_steane[1]*(A_steane[1]*(A_steane[1]*(A_steane[1]*x**4)**4)**4)**4

def steane_overhead(i):
    return 7**i

def c4steane(x,i):
    global A, B, A_c4steane
    if i==1:
        return A*B*x
    if i==2:
        return A_c4steane*x**3
    if i>2:
        return steane(c4steane(x,2),i-2)
def c4steane_overhead(i):
    return 2*7**(i-1)

def next(array):
    if sum(array)==0:
        return 1
    else:
        min_index = min([i for i in range(len(array)) if array[i]>0])
        return min_index+1

def hamming(x,a,b,k):
    if k==0:
        return x
    if x>a**(1/(2**k)-1):
        return 1
    else:
        return b*((a*x)**(2**(k-1))/a)**2

def concatenated_hamming(x,ham,b,c,d,e,f):
    l1 = hamming(x,ham[0,0],ham[0,next([c,d,e,f])],b)
    l2 = hamming(l1,ham[1,0],ham[1,next([d,e,f])],c)
    l3 = hamming(l2,ham[2,0],ham[2,next([e,f])],d)
    l4 = hamming(l3,ham[3,0],ham[3,next([f])],e)
    l5 = hamming(l4,ham[4,0],ham[4,1],f)

    return l5

def hamming_overhead(b,c,d,e,f):
    if b==0 and c==0 and d==0 and e==0 and f==0:
        return 1
    else:
        bottom_code=7
        if b==0:
            bottom_code=15
            if c==0:
                bottom_code=31
                if d==0:
                    bottom_code=63
                    if e==0:
                        bottom_code=127
        return 7**b*(15/7)**c*(31/21)**d*(63/51)**e*(127/113)**f
    
def load_parameters(error_model):
    global A, B, ham, A_surface, B_surface, A_steane, A_c4steane

    parameters = json.load(open('estimated_parameters.json', 'r'))
    ham = np.zeros([5,5])
    A = parameters[f'A_c4c6_{error_model}']['val']
    B = parameters[f'B_c4c6_{error_model}']['val']
    for r in range(5):
        for r2 in range(max(5-r,2)):
            ham[r,r2] = parameters[f'a_{2**(r+3)-1}_{2**(r+3+r2)-1}_{error_model}']['val']
    A_surface = parameters[f'A_surface_{error_model}']['val']
    B_surface = parameters[f'B_surface_{error_model}']['val']
    A_steane=[parameters[f'a_steane_l1_{error_model}']['val'],parameters[f'a_steane_l2_{error_model}']['val']]
    A_c4steane=parameters[f'a_c4steane_{error_model}']['val']

    
def estimate_overhead():
    global A, B, ham, A_surface, B_surface, A_steane, A_c4steane

    fig2, ax2 = plt.subplots(figsize=(6,4))

    for j in [0,1,2]:
        print()
        error_model = ['a','b','c'][j]
        print('error model:',error_model)

        load_parameters(error_model)

        threshold_c4c6 = 1/B
        print('C4/C6 threshold:', threshold_c4c6*1e2, '%')
        threshold_surface = 1/B_surface
        print('surface threshold:',threshold_surface*1e2, '%')
        threshold_steane = 1/ham[0,0]
        print('steane threshold:',threshold_steane*1e2, '%')

        overhead_c4c6hamming=[]
        error_c4c6hamming=[]
        overhead_surface=[]
        error_surface=[]
        p=1e-3

        i_first=0
        i_second=0

        for i in range(1,6,1):
            error=c4c6(p,i)
            overhead=c4c6_overhead(i)
            error_c4c6hamming.append(error)
            overhead_c4c6hamming.append(overhead)
            if i_first==0 and error<1e-10:
                i_first=i-1
        bbb = [0,0,0,0,0]
        list_index=[2,3,4,4]
        for i in range(4):
            bbb[list_index[i]]+=1
            error=concatenated_hamming(c4c6(p,5),ham,bbb[0],bbb[1],bbb[2],bbb[3],bbb[4])
            overhead=c4c6_overhead(5)*hamming_overhead(bbb[0],bbb[1],bbb[2],bbb[3],bbb[4])
            error_c4c6hamming.append(error)
            overhead_c4c6hamming.append(overhead)
            if i_first==0 and error<1e-10:
                i_first=i+5
            if i_second==0 and error<1e-24:
                i_second=i+5
                break
        
        g_first=0
        g_second=0

        for g in list(range(3,201,2)):
            error=surface(p,g)
            overhead=surface_overhead(g)
            overhead_surface.append(overhead)
            error_surface.append(error)
            if g_first==0 and error<1e-10:
                g_first=(g-3)//2
            if g_second==0 and error<1e-24:
                g_second=(g-3)//2
                break
        overhead_c4c6hamming=np.array(overhead_c4c6hamming)
        error_c4c6hamming=np.array(error_c4c6hamming)
        overhead_surface=np.array(overhead_surface)
        error_surface=np.array(error_surface)
        fig, ax = plt.subplots(figsize=(6,4))
        x = np.linspace(0, 10, 100)
        y1 = 4 + 2 * np.sin(2 * x)

        xmax=1e26

        ax.axhline(y=0, color='black', linestyle=':', alpha=0.5)
        ax.axvline(x=1/1e-24, color='black', linestyle='--', alpha=0.5)
        ax.axvline(x=1/1e-10, color='black', linestyle='-.', alpha=0.5)
        ax.plot(1/error_surface,overhead_surface, '-o',label='Surface code', color=color10)
        ax.plot(1/error_c4c6hamming,overhead_c4c6hamming, '-^',label='Proposed protocol', color=color1)
        ax.axhline(y=overhead_surface[g_second], xmin=0, xmax=np.log(1/error_surface[g_second])/np.log(xmax), color=color10, linestyle='--')
        ax.axhline(y=overhead_surface[g_first], xmin=0, xmax=np.log(1/error_surface[g_first])/np.log(xmax), color=color10, linestyle='-.')
        ax.axhline(y=overhead_c4c6hamming[i_second], xmin=0, xmax=np.log(1/error_c4c6hamming[i_second])/np.log(xmax), color=color1, linestyle='--')
        ax.axhline(y=overhead_c4c6hamming[i_first], xmin=0, xmax=np.log(1/error_c4c6hamming[i_first])/np.log(xmax), color=color1, linestyle='-.')
        print('Surface:',overhead_surface[g_second], overhead_surface[g_first])
        print('C4C6Hamming',overhead_c4c6hamming[i_second],overhead_c4c6hamming[i_first])
        ax.set_xlabel('1/(Logical CNOT error rate)')
        ax.set_ylabel('Space overhead')
        ax.set_xlim([1,xmax])
        ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(3,3))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[::-1],labels=labels[::-1], loc='upper left', bbox_to_anchor=(1, 1))
        ax.loglog()

        xmax=1e37

        if j==0:
            surface_label = r'Surface code ($\gamma = p/10$)'
            proposed_label = r'Proposed protocol ($\gamma=p/10$)'
            surface_ls = ':d'
            proposed_ls = ':<'
        if j==1:
            surface_label = r'Surface code ($\gamma = p/2$)'
            proposed_label = r'Proposed protocol ($\gamma=p/2$)'
            surface_ls = '--s'
            proposed_ls = '--v'
        if j==2:
            surface_label = r'Surface code ($\gamma = p$)'
            proposed_label = r'Proposed protocol ($\gamma=p$)'
            surface_ls = '-o'
            proposed_ls = '-^'

        if j==0:
            ax2.axhline(y=0, color='black', linestyle=':', alpha=0.5)
            ax2.axvline(x=1/1e-24, color='black', linestyle='--', alpha=0.5)
            ax2.axvline(x=1/1e-10, color='black', linestyle='-.', alpha=0.5)
        ax2.plot(1/error_surface,overhead_surface, surface_ls,label=surface_label, color=color10, markersize=4)
        ax2.plot(1/error_c4c6hamming,overhead_c4c6hamming, proposed_ls,label=proposed_label, color=color1, markersize=4)

        if j==2:
            ax2.set_xlabel('1/(Logical CNOT error rate)')
            ax2.set_ylabel('Space overhead')
            ax2.set_xlim([1,xmax])
            ax2.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
            ax2.ticklabel_format(style="sci", axis="y", scilimits=(3,3))
            handles, labels = ax2.get_legend_handles_labels()
            ax2.legend(handles=handles[::-1],labels=labels[::-1], loc='upper left', bbox_to_anchor=(1, 1))
            ax2.loglog()

        plt.show()
        fig.savefig(f'../fig/overhead_total_{error_model}.pdf', bbox_inches='tight')
    fig2.savefig(f'../fig/overhead_total_all_error_models.pdf', bbox_inches='tight')

def optimize_overhead():
    global A, B, ham, A_surface, B_surface, A_steane, A_c4steane

    for j in [0,1,2]:
        print()
        error_model = ['a','b','c'][j]
        print('error model:',error_model)

        load_parameters(error_model)

        threshold_c4c6 = 1/B
        threshold_surface = 1/B_surface
        threshold_steane = 1/ham[0,0]

        list_c4c6hamming_overhead=[]
        list_surfacehamming_overhead=[]
        list_c4steanehamming_overhead=[]
        list_steanehamming_overhead=[]

        list_p=list(np.exp(np.linspace(np.log(1e-4), np.log(1e-2),401)))
        for p in list_p:
            # C4C6+Hamming
            overheads = []
            variables = []
            if p<threshold_c4c6:
                for a in range(10):
                    l1 = c4c6(p,a)
                    for b in range(2):
                        for c in range(5):
                            for d in range(5):
                                for e in range(5):
                                    for f in range(5):
                                        logical_error = concatenated_hamming(l1,ham,b,c,d,e,f)
                                        overhead = c4c6_overhead(a)*hamming_overhead(b,c,d,e,f)
                                        if logical_error<1e-24:
                                            overheads.append(overhead)
                                            variables.append([a,b,c,d,e,f])
                                            break
            if len(overheads)==0:
                list_c4c6hamming_overhead.append(1e30)
            else:
                list_c4c6hamming_overhead.append(min(overheads))
            if p in [list_p[0],list_p[200],list_p[400]]:
                min_variable = variables[np.argmin(overheads)]
                print('At p=', p, ', the best variable for C4C6/Hamming code is', min_variable, 'with the overhead', min(overheads))
            # surface+Hamming
            overheads = []
            variables = []
            if p<threshold_surface:
                if p>threshold_steane:
                    g_start=math.ceil(np.log(threshold_steane/A_surface)/np.log(B_surface*p))*2-1
                else:
                    g_start=1
                for g in list(range(g_start,g_start+201,2))+[0]:
                    l1=surface(p,g)
                    for b in range(2):
                        for c in range(5):
                            for d in range(5):
                                for e in range(5):
                                    for f in range(5):
                                        logical_error = concatenated_hamming(l1,ham,b,c,d,e,f)
                                        overhead = surface_overhead(g)*hamming_overhead(b,c,d,e,f)
                                        if logical_error<1e-24:
                                            overheads.append(overhead)
                                            variables.append([g,b,c,d,e,f])
                                            break
            if len(overheads)==0:
                list_surfacehamming_overhead.append(1e30)
            else:
                list_surfacehamming_overhead.append(min(overheads))
            if p in [list_p[0],list_p[200]]:
                min_variable = variables[np.argmin(overheads)]
                print('At p=', p, ', the best variable for surface/Hamming code is', min_variable, 'with the overhead', min(overheads))
            # steane+Hamming
            overheads = []
            variables = []
            if p<A_steane[1]**(1/3):
                for a in range(1,9,1):
                    l1 = steane(p,a)
                    for b in range(2):
                        for c in range(5):
                            for d in range(5):
                                for e in range(5):
                                    for f in range(5):
                                        logical_error = concatenated_hamming(l1,ham,b,c,d,e,f)
                                        overhead = steane_overhead(a)*hamming_overhead(b,c,d,e,f)
                                        if logical_error<1e-24:
                                            overheads.append(overhead)
                                            variables.append([a,b,c,d,e,f])
                                            break
            if len(overheads)==0:
                list_steanehamming_overhead.append(1e30)
            else:
                list_steanehamming_overhead.append(min(overheads))
            if p in [list_p[0]]:
                min_variable = variables[np.argmin(overheads)]
                print('At p=', p, ', the best variable for Steane/Hamming code is', min_variable, 'with the overhead', min(overheads))
            # c4/steane+Hamming
            overheads = []
            variables = []
            if p<(A_steane[1]**(1/3)/A_c4steane)**(1/3):
                for a in range(1,10,1):
                    l1 = c4steane(p,a)
                    for b in range(2):
                        for c in range(5):
                            for d in range(5):
                                for e in range(5):
                                    for f in range(5):
                                        logical_error = concatenated_hamming(l1,ham,b,c,d,e,f)
                                        overhead = c4steane_overhead(a)*hamming_overhead(b,c,d,e,f)
                                        if logical_error<1e-24:
                                            overheads.append(overhead)
                                            variables.append([a,b,c,d,e,f])
                                            break
            if len(overheads)==0:
                list_c4steanehamming_overhead.append(1e30)
            else:
                list_c4steanehamming_overhead.append(min(overheads))
            if p in [list_p[0],list_p[200]]:
                min_variable = variables[np.argmin(overheads)]
                print('At p=', p, ', the best variable for C4Steane/Hamming code is', min_variable, 'with the overhead', min(overheads))
        plt.plot(list_p, list_c4c6hamming_overhead, '-', label=r'$C_4/C_6$+Hamming', color=color1)
        plt.plot(list_p, list_surfacehamming_overhead, '--', label='Surface+Hamming', color=color10)
        plt.plot(list_p, list_steanehamming_overhead, ':', label='Steane+Hamming',color=color2)
        plt.plot(list_p, list_c4steanehamming_overhead, linestyle='-.', label=r'$C_4$/Steane+Hamming',color=color8)

        plt.loglog()
        plt.xlim([1e-4,1e-2])
        plt.ylim([10, 1e6])
        plt.legend()
        plt.xlabel('Physical error rate')
        plt.ylabel('Space overhead')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
        plt.savefig('../fig/space_overhead_underlying'+str(j)+'.pdf', bbox_inches='tight')

if __name__ == '__main__':
    estimate_overhead()
    # optimize_overhead()