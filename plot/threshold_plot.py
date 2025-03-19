import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.ticker as ptick
import matplotlib.gridspec as gridspec
import json
import matplotlib.pyplot as plt
import numpy as np
import sinter
from typing import TYPE_CHECKING, Any, cast

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

blue1=np.array([8,81,156])/255
blue2=np.array([49,130,189])/255
blue3=np.array([107,174,214])/255
blue4=np.array([158,202,225])/255
blue5=np.array([198,219,239])/255

green1=np.array([0,109,44])/255
green2=np.array([49,163,84])/255
green3=np.array([116,196,118])/255
green4=np.array([186,228,179])/255

yellow1=np.array([102,102,0])/255
yellow2=np.array([204,204,0])/255
yellow3=np.array([240,230,140])/255

red1=np.array([222,45,38])/255
red2=np.array([252,146,114])/255

purple1=np.array([117,107,177])/255
purple2=np.array([188,189,220])/255

def yerr_log(y,var):
    yerr_log_list=np.zeros([2,len(y)])
    for i in range(len(y)):
        yerr_log_list[1,i] = (np.exp(np.sqrt(var[i])/y[i])-1)*y[i]
        yerr_log_list[0,i] = (1-np.exp(-1.0*np.sqrt(var[i])/y[i]))*y[i]
    return yerr_log_list

def load(filename):
    json_open=open(filename, 'r')
    json_load = json.load(json_open)

    p = np.array([v["physical_error"] for v in json_load.values()])
    err = np.array([v["logical_error"] for v in json_load.values()])
    var = np.array([v["variance"] for v in json_load.values()])

    return p,err,var

def read_file_error_rate(distance, num_shots, error_model, min_error=0, max_error=1):
    f = open(f'../data/surface_code/output_surface_d{distance}_error_model_{error_model}_{num_shots}runs', 'r')

    data_dict = {}

    for data in f:
        parts = data.split() 
        if len(parts) == 3:  # Check whether input is in a correct data format
            erate, ave_infid, std = parts
            erate_value = float(erate)
            ave_infid_value = float(ave_infid)
            std_value = float(std)

            # Check for erate range and then for duplicates with a lower ave_infid
            if min_error <= erate_value <= max_error:
                if erate_value not in data_dict or data_dict[erate_value][0] < ave_infid_value:
                    data_dict[erate_value] = (ave_infid_value, std_value)

    # Sort the erate values to maintain order
    sorted_erate_list = sorted(data_dict.keys())

    # Extract sorted lists from the dictionary
    ave_infid_list = [data_dict[erate][0] for erate in sorted_erate_list]
    if error_model == 'c':
        std_infid_list = [data_dict[erate][1] for erate in sorted_erate_list]
    else:
        std_infid_list = [np.sqrt(data_dict[erate][0]*(1.0-data_dict[erate][0])/num_shots) for erate in sorted_erate_list]
    erate_list = sorted_erate_list

    return erate_list, ave_infid_list, std_infid_list

def sort_lists(a, b, c):
    zipped_lists = zip(a, b, c)
    sorted_lists = sorted(zipped_lists, key=lambda x: x[0])
    a, b, c = zip(*sorted_lists)
    
    return list(a), list(b), list(c)

def func_threshold(variables, b0, b1, b2, mu, pth):
    p, ell = variables
    expr = (p - pth) * (ell ** (1 / mu))
    return cast(float, b0 + b1 * expr + b2 * (expr**2))


def func_fitting(variables, A, B):
    x, d = variables
    return A * (B * x)**((d + 1)/2)

def connect_with_and(input):
    ans=''
    for a in input:
        ans=ans+'$'+a+'$ & '
    ans=ans[:-2]
    return ans

def parabola(x,a):
    return a*x**2

def estimate_underlying_code_threshold(output_file_name):
    p_list=np.array([1e-10, 1])

    fig = plt.figure(figsize=(18, 8))
    fig2 = plt.figure(figsize=(18, 4))
    ax1 = [fig.add_subplot(2, 3, 1), fig.add_subplot(2, 3, 2), fig.add_subplot(2, 3, 3)]
    ax2 = [fig.add_subplot(2, 3, 4), fig.add_subplot(2, 3, 5), fig.add_subplot(2, 3, 6)]
    ax3 = [fig2.add_subplot(1, 3, 1), fig2.add_subplot(1, 3, 2), fig2.add_subplot(1, 3, 3)]

    for j in range(3):
        error_model = ['a','b','c'][j]
        print('error_model=',error_model)
        p_steanel1, err_steanel1, var_steanel1 = load(f'../data/underlying_code/output_concatenated_steane_level1_error_model_{error_model}_10000000runs.json')
        p_steanel2, err_steanel2, var_steanel2 = load(f'../data/underlying_code/output_concatenated_steane_level2_error_model_{error_model}_10000000runs.json')
        p_c4steane, err_c4steane, var_c4steane = load(f'../data/underlying_code/output_c4steane_level2_error_model_{error_model}_1000000runs.json')
        p_c4, err_c4, var_c4 = load(f'../data/underlying_code/output_c4c6_level1_error_model_{error_model}_1000000runs.json')
        p_c4c6, err_c4c6, var_c4c6 = load(f'../data/underlying_code/output_c4c6_level2_error_model_{error_model}_1000000runs.json')

        i=0
        while 1:
            if i>=len(err_c4steane):
                break
            if err_c4steane[i]==0:
                p_c4steane=np.delete(p_c4steane,i)
                err_c4steane=np.delete(err_c4steane,i)
                var_c4steane=np.delete(var_c4steane,i)
            else:
                i=i+1

        def linear(x,a):
            return a*x
        def square(x,a):
            return a*x**2
        def tri(x,a):
            return a*x**3
        def quatro(x,a):
            return a*x**4

        def func_c4c6(X,A,B):
            x,y=X
            return A*(B*x)**y

        p_c4c6_all = np.concatenate([p_c4, p_c4c6])
        err_c4c6_all = np.concatenate([err_c4, err_c4c6])
        var_c4c6_all = np.concatenate([var_c4, var_c4c6])
        level_c4c6 = np.concatenate([np.full(len(p_c4),1), np.full(len(p_c4c6),2)])
        var_level_c4c6 = np.zeros(len(p_c4)+len(p_c4c6))

        # Exclude 0 points
        i=0
        while i<len(p_c4c6_all):
            if err_c4c6_all[i]==0:
                p_c4c6_all=np.delete(p_c4c6_all,i)
                err_c4c6_all=np.delete(err_c4c6_all,i)
                var_c4c6_all=np.delete(var_c4c6_all,i)
                level_c4c6=np.delete(level_c4c6,i)
                var_level_c4c6=np.delete(var_level_c4c6,i)
            else:
                i=i+1

        popt_c4c6, pcov_c4c6 = curve_fit(func_c4c6,(p_c4c6_all,level_c4c6),err_c4c6_all,sigma=np.sqrt(var_c4c6_all),absolute_sigma=True, p0=[1,40])
        popt_steanel1, pcov_steanel1 = curve_fit(square,p_steanel1,err_steanel1,sigma=np.sqrt(var_steanel1),absolute_sigma=True)
        popt_steanel2, pcov_steanel2 = curve_fit(quatro,p_steanel2,err_steanel2,sigma=np.sqrt(var_steanel2),absolute_sigma=True, p0=2e10)
        popt_c4steane, pcov_c4steane = curve_fit(tri,p_c4steane,err_c4steane,sigma=np.sqrt(var_c4steane),absolute_sigma=True)

        print('A_c4c6 = ', popt_c4c6[0], '\pm', np.sqrt(pcov_c4c6[0][0]))
        print('B_c4c6 = ', popt_c4c6[1], '\pm' , np.sqrt(pcov_c4c6[1][1]))

        print('a_steane_l1=', popt_steanel1[0], '\pm', np.sqrt(pcov_steanel1[0][0]))
        print('a_steane_l2=', popt_steanel2[0], '\pm', np.sqrt(pcov_steanel2[0][0]))

        print('a_c4steane=', popt_c4steane[0], '\pm', np.sqrt(pcov_c4steane[0][0]))

        with open(output_file_name, 'a') as file:
            file.write('\"A_c4c6_'+error_model+'\":'+json.dumps({'val':popt_c4c6[0], 'err':np.sqrt(pcov_c4c6[0][0])})+',\n')
            file.write('\"B_c4c6_'+error_model+'\":'+json.dumps({'val':popt_c4c6[1], 'err':np.sqrt(pcov_c4c6[1][1])})+',\n')
            file.write('\"a_steane_l1_'+error_model+'\":'+json.dumps({'val':popt_steanel1[0], 'err':np.sqrt(pcov_steanel1[0][0])})+',\n')
            file.write('\"a_steane_l2_'+error_model+'\":'+json.dumps({'val':popt_steanel2[0], 'err':np.sqrt(pcov_steanel2[0][0])})+',\n')
            file.write('\"a_c4steane_'+error_model+'\":'+json.dumps({'val':popt_c4steane[0], 'err':np.sqrt(pcov_c4steane[0][0])})+',\n')

        p0=np.array([1e-7,1])

        fitting_c4 = popt_c4c6[0]*popt_c4c6[1]*p0
        fitting_c4c6 = popt_c4c6[0]*(popt_c4c6[1]*p0)**2
        fitting_steanel1 = popt_steanel1[0]*p0**2
        fitting_steanel2 = popt_steanel2[0]*p0**4
        fitting_c4steane = popt_c4steane[0]*p0**3

        ax1[j].errorbar(p_c4steane, err_c4steane, yerr=yerr_log(err_c4steane,var_c4steane), linestyle="None", color=color8, fmt='v', capsize=4)
        ax1[j].plot(p0,fitting_c4steane, '-v',color=color8, label='level-2 $C_4$/Steane code')
        ax1[j].errorbar(p_steanel2, err_steanel2, yerr=yerr_log(err_steanel2,var_steanel2), linestyle="None", color=color2, fmt='D', capsize=4)
        ax1[j].plot(p0,fitting_steanel2, '-D',color=color2, label='level-2 Steane code')
        ax1[j].errorbar(p_steanel1, err_steanel1, yerr=yerr_log(err_steanel1,var_steanel1), linestyle="None", color=color4, fmt='^', capsize=4)
        ax1[j].plot(p0,fitting_steanel1, '-^',color=color4, label='level-1 Steane code')
        ax1[j].errorbar(p_c4c6, err_c4c6, yerr=yerr_log(err_c4c6,var_c4c6), linestyle="None", color=color1, fmt='s', capsize=4)
        ax1[j].plot(p0,fitting_c4c6, '-s',color=color1, label='level-2 $C_4/C_6$ code')
        ax1[j].errorbar(p_c4, err_c4, yerr=yerr_log(err_c4,var_c4), linestyle="None", color=color5, fmt='o', capsize=4)
        ax1[j].plot(p0,fitting_c4,'-o',color=color5, label='$C_4$ code')

        ax1[j].set_xscale('log')
        ax1[j].set_yscale('log')
        ax1[j].set_xlim([5e-6,5e-3])
        ax1[j].set_ylim([1e-8,1])
        ax1[j].set_xlabel('Physical error rate')
        ax1[j].set_ylabel('Logical CNOT error rate')

        if j==0:
            title=r"(a) $\gamma = p/10$"
        if j==1:
            title=r"(b) $\gamma = p/2$"
        if j==2:
            title=r"(c) $\gamma = p$"
        ax1[j].set_title(title)

        start = 0
        end = -1

        distance_fit = []
        error_prob_fit = []
        infid_fit = []
        std_fit = []

        distance_list = [5, 7, 9, 11]

        for distance in distance_list:
            if error_model == 'c':
                min_error=0.00002
                max_error=0.002
            else:
                min_error=0.0003
                max_error=0.002
            erate_list, ave_infid_list, std_infid_list = read_file_error_rate(distance, num_shots=1000000, error_model=error_model, min_error=min_error, max_error=max_error)
            erate_list = np.array(erate_list)
            ave_infid_list = np.array(ave_infid_list)
            std_infid_list = np.array(std_infid_list)
                    
            if distance==5:
                lab='d=5'
                mark='o'
                col=color9
            if distance==7:
                lab='d=7'
                mark='s'
                col=color6
            if distance==9:
                lab='d=9'
                mark='^'
                col=color11
            if distance==11:
                lab='d=11'
                mark='v'
                col=color10
            
            i=0
            while i<len(ave_infid_list):
                if ave_infid_list[i]==0:
                    erate_list=np.delete(erate_list,i)
                    ave_infid_list=np.delete(ave_infid_list,i)
                    std_infid_list=np.delete(std_infid_list,i)
                else:
                    i=i+1
            ax2[j].errorbar(erate_list, ave_infid_list, yerr_log(ave_infid_list,std_infid_list**2), marker = mark, label = 'Surface code ('+lab+')', linestyle='None', color=col, capsize=4)
            
            distance_fit.extend([int(distance) for _ in range(len(erate_list[start:end]))])
            infid_fit.extend(ave_infid_list[start:end])
            error_prob_fit.extend(erate_list[start:end])
            std_fit.extend(std_infid_list[start:end])


        # Exclude 0 points
        i=0
        while i<len(std_fit):
            if infid_fit[i]==0:
                distance_fit=np.delete(distance_fit,i)
                infid_fit=np.delete(infid_fit,i)
                error_prob_fit=np.delete(error_prob_fit,i)
                std_fit=np.delete(std_fit,i)
            else:
                i=i+1

        popt_surface, pcov_surface = curve_fit(func_fitting, (error_prob_fit, distance_fit), infid_fit, sigma=std_fit, maxfev=100000, absolute_sigma=True)
        print('A_surface =',popt_surface[0],'\pm',np.sqrt(pcov_surface[0][0]))
        print('B_surface =',popt_surface[1],'\pm',np.sqrt(pcov_surface[1][1]))

        for distance in distance_list:
            if distance==5:
                lab='d=5'
                mark='o'
                col=color9
            if distance==7:
                lab='d=7'
                mark='s'
                col=color6
            if distance==9:
                lab='d=9'
                mark='^'
                col=color11
            if distance==11:
                lab='d=11'
                mark='v'
                col=color10
            fitting_results = func_fitting((p0, distance), popt_surface[0], popt_surface[1])
            ax2[j].plot(p0, fitting_results, color=col)

        distance_fit = []
        error_prob_fit = []
        infid_fit = []
        std_fit = []

        for distance in distance_list:
            if error_model == 'c':
                min_error=0.002
                max_error=0.004
            else:
                min_error=0.0035
                max_error=0.006
            erate_list, ave_infid_list, std_infid_list = read_file_error_rate(distance, num_shots=1000000, error_model=error_model, min_error=min_error, max_error=max_error)
            ave_infid_list=np.array(ave_infid_list)
            std_infid_list=np.array(std_infid_list)

            i=0
            while i<len(ave_infid_list):
                if ave_infid_list[i]==0:
                    erate_list=np.delete(erate_list,i)
                    ave_infid_list=np.delete(ave_infid_list,i)
                    std_infid_list=np.delete(std_infid_list,i)
                else:
                    i=i+1

            distance_fit.extend([int(distance) for _ in range(len(erate_list[start:end]))])
            infid_fit.extend(ave_infid_list[start:end])
            error_prob_fit.extend(erate_list[start:end])
            std_fit.extend(std_infid_list[start:end])

            if distance==5:
                lab='d=5'
                mark='o'
                col=color9
            if distance==7:
                lab='d=7'
                mark='s'
                col=color6
            if distance==9:
                lab='d=9'
                mark='^'
                col=color11
            if distance==11:
                lab='d=11'
                mark='v'
                col=color10

            ax3[j].errorbar(erate_list, ave_infid_list, yerr_log(ave_infid_list,std_infid_list**2), marker = mark, label = 'Surface code ('+lab+')', linestyle='None', color=col, capsize=4)

        fitting_results = []

        popt,pcov = curve_fit(func_threshold, (error_prob_fit, distance_fit), infid_fit, sigma=std_fit, maxfev=100000, absolute_sigma=True)

        print('C_surface=',popt[0],'\pm',np.sqrt(pcov[0][0]))
        print('D_surface=',popt[1],'\pm',np.sqrt(pcov[1][1]))
        print('E_surface=',popt[2],'\pm',np.sqrt(pcov[2][2]))
        print('mu=',popt[3],'\pm',np.sqrt(pcov[3][3]))
        print('p_th_surface=',popt[4],'\pm',np.sqrt(pcov[4][4]))

        with open(output_file_name, 'a') as file:
            file.write('\"A_surface_'+error_model+'\":'+json.dumps({'val':popt_surface[0], 'err':np.sqrt(pcov_surface[0][0])})+',\n')
            file.write('\"B_surface_'+error_model+'\":'+json.dumps({'val':popt_surface[1], 'err':np.sqrt(pcov_surface[1][1])})+',\n')
            file.write('\"C_surface_'+error_model+'\":'+json.dumps({'val':popt[0], 'err':np.sqrt(pcov[0][0])})+',\n')
            file.write('\"D_surface_'+error_model+'\":'+json.dumps({'val':popt[1], 'err':np.sqrt(pcov[1][1])})+',\n')
            file.write('\"E_surface_'+error_model+'\":'+json.dumps({'val':popt[2], 'err':np.sqrt(pcov[2][2])})+',\n')
            file.write('\"mu_'+error_model+'\":'+json.dumps({'val':popt[3], 'err':np.sqrt(pcov[3][3])})+',\n')
            file.write('\"p_th_surface_'+error_model+'\":'+json.dumps({'val':popt[4], 'err':np.sqrt(pcov[4][4])})+',\n')

        ax3[j].axvline(x= popt[-1], color='black', linestyle='--', alpha=0.5)

        for distance in distance_list:
            fitting_results = []
            ppp = np.array(np.arange(0.001,0.006,0.00001))
            fitting_results = func_threshold((ppp, distance), popt[0], popt[1], popt[2], popt[3], popt[4])

            if distance==5:
                col=color9
            if distance==7:
                col=color6
            if distance==9:
                col=color11
            if distance==11:
                col=color10
            ax3[j].plot(ppp, fitting_results, color=col)

        pth_fitting = popt[-1]

        ax3[j].set_xlabel("Physical error rate")
        ax3[j].set_ylabel("Logical CNOT error rate")

        ax3[j].set_xlim([1.95e-3, 6.05e-3])
        ax3[j].set_xticks([0.002, 0.003,0.004,0.005,0.006])
        if error_model == "c":
            ax3[j].set_ylim([0, 1])
        else:
            ax3[j].set_ylim([0.07, 0.1])
            ax3[j].set_yticks([0.07, 0.08,0.09,0.1])

        ax3[j].set_title(title)

        ax2[j].set_xlabel("Physical error rate")
        ax2[j].set_ylabel("Logical CNOT error rate")
        ax2[j].set_yscale("log")
        ax2[j].set_xscale("log")
        ax2[j].set_xlim([5e-6,5e-3])
        ax2[j].set_ylim([1e-8, 1])

        handles, labels = ax1[2].get_legend_handles_labels()
        ax1[2].legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1, handles=handles[::-1],labels=labels[::-1])
        ax2[2].legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)
        ax3[2].legend(loc='upper left', bbox_to_anchor=(1.05, 1), ncol=1)

    plt.show()
    fig.savefig('../fig/underlying_threshold.pdf', bbox_inches='tight')
    fig2.savefig('../fig/surface_threshold.pdf', bbox_inches='tight')

def estimate_hamming_code_threshold(output_file_name):
    fig = plt.figure(figsize=(18, 4))
    ax=[fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)]
    p=np.array([1e-10, 1])

    for j in range(3):
        error_model = ['a','b','c'][j]
        p_7, err_7, var_7 = load(f'../data/hamming_code/output_hamming_code_7qubit_below_15qubit_error_model_{error_model}_1000000runs.json')
        p_15, err_15, var_15 = load(f'../data/hamming_code/output_hamming_code_15qubit_below_31qubit_error_model_{error_model}_1000000runs.json')
        p_31, err_31, var_31 = load(f'../data/hamming_code/output_hamming_code_31qubit_below_63qubit_error_model_{error_model}_1000000runs.json')
        p_63, err_63, var_63 = load(f'../data/hamming_code/output_hamming_code_63qubit_below_127qubit_error_model_{error_model}_1000000runs.json')
        p_127, err_127, var_127 = load(f'../data/hamming_code/output_hamming_code_127qubit_below_255qubit_error_model_{error_model}_1000000runs.json')

        i=0
        while 1:
            if i>=len(err_15):
                break
            if err_15[i]==0:
                p_15=np.delete(p_15,i)
                err_15=np.delete(err_15,i)
                var_15=np.delete(var_15,i)
            else:
                i=i+1
        i=0
        while 1:
            if i>=len(err_31):
                break
            if err_31[i]==0:
                p_31=np.delete(p_31,i)
                err_31=np.delete(err_31,i)
                var_31=np.delete(var_31,i)
            else:
                i=i+1
        i=0
        while 1:
            if i>=len(err_63):
                break
            if err_63[i]==0:
                p_63=np.delete(p_63,i)
                err_63=np.delete(err_63,i)
                var_63=np.delete(var_63,i)
            else:
                i=i+1

        i=0
        while 1:
            if i>=len(err_127):
                break
            if err_127[i]==0:
                p_127=np.delete(p_127,i)
                err_127=np.delete(err_127,i)
                var_127=np.delete(var_127,i)
            else:
                i=i+1

        cutoff=0.05

        cut_off_p_7 = np.array([p_7[i] for i in range(len(err_7)) if err_7[i]<cutoff])
        cut_off_err_7 = np.array([err_7[i] for i in range(len(err_7)) if err_7[i]<cutoff])
        cut_off_var_7 = np.array([var_7[i] for i in range(len(err_7)) if err_7[i]<cutoff])

        cut_off_p_15 = np.array([p_15[i] for i in range(len(err_15)) if err_15[i]<cutoff])
        cut_off_err_15 = np.array([err_15[i] for i in range(len(err_15)) if err_15[i]<cutoff])
        cut_off_var_15 = np.array([var_15[i] for i in range(len(err_15)) if err_15[i]<cutoff])

        cut_off_p_31 = np.array([p_31[i] for i in range(len(err_31)) if err_31[i]<cutoff])
        cut_off_err_31 = np.array([err_31[i] for i in range(len(err_31)) if err_31[i]<cutoff])
        cut_off_var_31 = np.array([var_31[i] for i in range(len(err_31)) if err_31[i]<cutoff])

        cut_off_p_63 = np.array([p_63[i] for i in range(len(err_63)) if err_63[i]<cutoff])
        cut_off_err_63 = np.array([err_63[i] for i in range(len(err_63)) if err_63[i]<cutoff])
        cut_off_var_63 = np.array([var_63[i] for i in range(len(err_63)) if err_63[i]<cutoff])

        cut_off_p_127 = np.array([p_127[i] for i in range(len(err_127)) if err_127[i]<cutoff])
        cut_off_err_127 = np.array([err_127[i] for i in range(len(err_127)) if err_127[i]<cutoff])
        cut_off_var_127 = np.array([var_127[i] for i in range(len(err_127)) if err_127[i]<cutoff])

        popt_7, pcov_7 = curve_fit(parabola,cut_off_p_7,cut_off_err_7,sigma=np.sqrt(cut_off_var_7),absolute_sigma=True)
        popt_15, pcov_15 = curve_fit(parabola,cut_off_p_15,cut_off_err_15,sigma=np.sqrt(cut_off_var_15),absolute_sigma=True,p0=1e5)
        popt_31, pcov_31 = curve_fit(parabola,cut_off_p_31,cut_off_err_31,sigma=np.sqrt(cut_off_var_31),absolute_sigma=True,p0=3e5)
        popt_63, pcov_63 = curve_fit(parabola,cut_off_p_63,cut_off_err_63,sigma=np.sqrt(cut_off_var_63),absolute_sigma=True,p0=3e7)
        popt_127, pcov_127 = curve_fit(parabola,cut_off_p_127,cut_off_err_127,sigma=np.sqrt(cut_off_var_127),absolute_sigma=True,p0=1e9)

        fitting_7=popt_7[0]*p**2
        fitting_15=popt_15[0]*p**2
        fitting_31=popt_31[0]*p**2
        fitting_63=popt_63[0]*p**2
        fitting_127=popt_127[0]*p**2
        print("error model=",error_model)
        print("a_7 = ", popt_7[0], "\pm", np.sqrt(pcov_7[0][0]))
        print("a_15 = ", popt_15[0], "\pm", np.sqrt(pcov_15[0][0]))
        print("a_31 = ", popt_31[0], "\pm", np.sqrt(pcov_31[0][0]))
        print("a_63 = ", popt_63[0], "\pm", np.sqrt(pcov_63[0][0]))
        print("a_127 = ", popt_127[0], "\pm", np.sqrt(pcov_127[0][0]))

        p0=q0=np.array([1e-10,1e-4])

        p1=popt_7[0]*q0**2
        p2=popt_15[0]*p1**2
        p3=popt_31[0]*p2**2
        p4=popt_63[0]*p3**2
        p5=popt_127[0]*p4**2

        q1=popt_15[0]*q0**2
        q2=popt_31[0]*q1**2
        q3=popt_63[0]*q2**2
        q4=popt_127[0]*q3**2

        ax[j].plot(p0,p1, ':', color=blue1, label='level-1')
        ax[j].plot(p0,p2, '--', color=green1, label='level-2')
        ax[j].plot(p0,p3, linestyle=(0, (5, 2, 1, 2, 1, 2)), color=yellow1, label='level-3')
        ax[j].plot(p0,p4, '-.', color=red1, label='level-4')
        ax[j].plot(p0,p5, color=purple1, label='level-5')
        ax[j].set_xlim([1e-9, 1e-5])
        ax[j].set_ylim([1e-24, 1e-2])
        ax[j].set_xscale('log')
        ax[j].set_yscale('log')
        ax[j].set_xlabel("Physical error rate")
        ax[j].set_ylabel("Logical CNOT error rate")
        if j==0:
            title=r"(a) $\gamma = p/10$"
        if j==1:
            title=r"(b) $\gamma = p/2$"
        if j==2:
            title=r"(c) $\gamma = p$"
        ax[j].set_title(title)
        if j==2:
            ax[j].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    fig.savefig("../fig/threshold_concatenated_hamming_code.pdf", bbox_inches='tight')

    fig = plt.figure(figsize=(18, 12))
    ax7 = [fig.add_subplot(3, 3, 1), fig.add_subplot(3, 3, 2), fig.add_subplot(3, 3, 3)]
    ax15 = [fig.add_subplot(3, 3, 4), fig.add_subplot(3, 3, 5), fig.add_subplot(3, 3, 6)]
    ax31 = [fig.add_subplot(3, 3, 7), fig.add_subplot(3, 3, 8), fig.add_subplot(3, 3, 9)]
    store_a7 = []
    store_a15 = []
    store_a31 = []
    store_a63 = []
    store_a127 = []
    for j in range(3):
        tmp=[]
        error_model=['a','b','c'][j]
        for i in range(5):
            above=['7','15','31','63','127'][i]
            color=[blue1,blue2,blue3,blue4,blue5][i]

            label=[r"$(r_l,r_{l+1})=(3,3)$",r"$(r_l,r_{l+1})=(3,4)$",r"$(r_l,r_{l+1})=(3,5)$",r"$(r_l,r_{l+1})=(3,6)$",r"$(r_l,r_{l+1})=(3,7)$"][i]

            p_7, err_7, var_7 = load(f'../data/hamming_code/output_hamming_code_7qubit_below_{above}qubit_error_model_{error_model}_1000000runs.json')

            cutoff=0.05

            cut_off_p_7 = np.array([p_7[i] for i in range(len(err_7)) if err_7[i]<cutoff])
            cut_off_err_7 = np.array([err_7[i] for i in range(len(err_7)) if err_7[i]<cutoff])
            cut_off_var_7 = np.array([var_7[i] for i in range(len(err_7)) if err_7[i]<cutoff])

            popt_7, pcov_7 = curve_fit(parabola,cut_off_p_7,cut_off_err_7,sigma=np.sqrt(cut_off_var_7),absolute_sigma=True)

            with open(output_file_name, 'a') as file:
                file.write('\"a_7_'+above+'_'+error_model+'\":'+json.dumps({'val':popt_7[0], 'err':np.sqrt(pcov_7[0][0])})+',\n')

            fitting_7=popt_7[0]*p**2
            tmp.append(str(popt_7[0])+' \pm '+str(np.sqrt(pcov_7[0][0])))

            ax7[j].errorbar(p_7, err_7, yerr=yerr_log(err_7,var_7), linestyle="None", fmt='os^Dv'[i], capsize=4, label=label, color=color)
            ax7[j].plot(p, fitting_7, color=color)
            ax7[j].set_xlim([1e-10, 2e-3])
            ax7[j].set_ylim([1e-8, 1])
            ax7[j].loglog()
            if j==2:
                ax7[j].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

            ax7[j].set_xlabel('Physical error rate')
            ax7[j].set_ylabel('Logical CNOT error rate')

            if j==0:
                title=r"(a) $\gamma = p/10$"
            if j==1:
                title=r"(b) $\gamma = p/2$"
            if j==2:
                title=r"(c) $\gamma = p$"

            ax7[j].set_title(title)
        store_a7.append(tmp)

        tmp=[]
        for i in range(4):
            above=['15','31','63','127'][i]
            color=[green1,green2,green3,green4][i]

            label=[r"$(r_l,r_{l+1})=(4,4)$",r"$(r_l,r_{l+1})=(4,5)$",r"$(r_l,r_{l+1})=(4,6)$",r"$(r_l,r_{l+1})=(4,7)$"][i]

            p_15, err_15, var_15 = load(f'../data/hamming_code/output_hamming_code_15qubit_below_{above}qubit_error_model_{error_model}_1000000runs.json')

            cutoff=0.05

            cut_off_p_15 = np.array([p_15[i] for i in range(len(err_15)) if err_15[i]<cutoff])
            cut_off_err_15 = np.array([err_15[i] for i in range(len(err_15)) if err_15[i]<cutoff])
            cut_off_var_15 = np.array([var_15[i] for i in range(len(err_15)) if err_15[i]<cutoff])


            popt_15, pcov_15 = curve_fit(parabola,cut_off_p_15,cut_off_err_15,sigma=np.sqrt(cut_off_var_15),absolute_sigma=True)

            with open(output_file_name, 'a') as file:
                file.write('\"a_15_'+above+'_'+error_model+'\":'+json.dumps({'val':popt_15[0], 'err':np.sqrt(pcov_15[0][0])})+',\n')

            fitting_15=popt_15[0]*p**2
            tmp.append(str(popt_15[0])+' \pm '+str(np.sqrt(pcov_15[0][0])))

            ax15[j].errorbar(p_15, err_15, yerr=yerr_log(err_15,var_15), linestyle="None", fmt='os^Dv'[i], capsize=4, label=label, color=color)
            ax15[j].plot(p, fitting_15, color=color)
            ax15[j].set_xlim([1e-10, 2e-3])
            ax15[j].set_ylim([1e-8, 1])
            ax15[j].loglog()
            if j==2:
                ax15[j].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

            ax15[j].set_xlabel('Physical error rate')
            ax15[j].set_ylabel('Logical CNOT error rate')
        store_a15.append(tmp)

        tmp=[]
        for i in range(3):
            above=['31','63','127'][i]
            color=[yellow1,yellow2,yellow3][i]

            label=[r"$(r_l,r_{l+1})=(5,5)$",r"$(r_l,r_{l+1})=(5,6)$",r"$(r_l,r_{l+1})=(5,7)$"][i]

            p_31, err_31, var_31 = load(f'../data/hamming_code/output_hamming_code_31qubit_below_{above}qubit_error_model_{error_model}_1000000runs.json')

            cutoff=0.05

            cut_off_p_31 = np.array([p_31[i] for i in range(len(err_31)) if err_31[i]<cutoff and err_31[i]>0])
            cut_off_err_31 = np.array([err_31[i] for i in range(len(err_31)) if err_31[i]<cutoff and err_31[i]>0])
            cut_off_var_31 = np.array([var_31[i] for i in range(len(err_31)) if err_31[i]<cutoff and err_31[i]>0])


            popt_31, pcov_31 = curve_fit(parabola,cut_off_p_31,cut_off_err_31,sigma=np.sqrt(cut_off_var_31),absolute_sigma=True,p0=1e6)

            with open(output_file_name, 'a') as file:
                file.write('\"a_31_'+above+'_'+error_model+'\":'+json.dumps({'val':popt_31[0], 'err':np.sqrt(pcov_31[0][0])})+',\n')

            fitting_31=popt_31[0]*p**2
            tmp.append(str(popt_31[0])+' \pm '+str(np.sqrt(pcov_31[0][0])))

            ax31[j].errorbar(p_31, err_31, yerr=yerr_log(err_31,var_31), linestyle="None", fmt='os^Dv'[i], capsize=4, label=label, color=color)
            ax31[j].plot(p, fitting_31, color=color)
            ax31[j].set_xlim([1e-10, 2e-3])
            ax31[j].set_ylim([1e-8, 1])
            ax31[j].loglog()
            if j==2:
                ax31[j].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

            ax31[j].set_xlabel('Physical error rate')
            ax31[j].set_ylabel('Logical CNOT error rate')
        store_a31.append(tmp)

        tmp=[]
        for i in range(2):
            above=['63','127'][i]
            color=[red1,red2][i]

            label=[r"$(r_l,r_{l+1})=(6,6)$",r"$(r_l,r_{l+1})=(6,7)$"][i]

            p_63, err_63, var_63 = load(f'../data/hamming_code/output_hamming_code_63qubit_below_{above}qubit_error_model_{error_model}_1000000runs.json')

            cutoff=0.05

            cut_off_p_63 = np.array([p_63[i] for i in range(len(err_63)) if err_63[i]<cutoff and err_63[i]>0])
            cut_off_err_63 = np.array([err_63[i] for i in range(len(err_63)) if err_63[i]<cutoff and err_63[i]>0])
            cut_off_var_63 = np.array([var_63[i] for i in range(len(err_63)) if err_63[i]<cutoff and err_63[i]>0])


            popt_63, pcov_63 = curve_fit(parabola,cut_off_p_63,cut_off_err_63,sigma=np.sqrt(cut_off_var_63),absolute_sigma=True,p0=1e8)

            with open(output_file_name, 'a') as file:
                file.write('\"a_63_'+above+'_'+error_model+'\":'+json.dumps({'val':popt_63[0], 'err':np.sqrt(pcov_63[0][0])})+',\n')

            fitting_63=popt_63[0]*p**2
            tmp.append(str(popt_63[0])+' \pm '+str(np.sqrt(pcov_63[0][0])))

            ax31[j].errorbar(p_63, err_63, yerr=yerr_log(err_63,var_63), linestyle="None", fmt='os^Dv'[i], capsize=4, label=label, color=color)
            ax31[j].plot(p, fitting_63, color=color)
            ax31[j].set_xlim([1e-10, 2e-3])
            ax31[j].set_ylim([1e-8, 1])
            ax31[j].loglog()
            if j==2:
                ax31[j].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

            ax31[j].set_xlabel('Physical error rate')
            ax31[j].set_ylabel('Logical CNOT error rate')
        store_a63.append(tmp)

        tmp=[]
        for i in range(2):
            above=['127','255'][i]
            color=[purple1,purple2][i]

            label=[r"$(r_l,r_{l+1})=(7,7)$",r"$(r_l,r_{l+1})=(7,8)$"][i]

            p_127, err_127, var_127 = load(f'../data/hamming_code/output_hamming_code_127qubit_below_{above}qubit_error_model_{error_model}_1000000runs.json')

            cutoff=0.05

            cut_off_p_127 = np.array([p_127[i] for i in range(len(err_127)) if err_127[i]<cutoff and err_127[i]>0])
            cut_off_err_127 = np.array([err_127[i] for i in range(len(err_127)) if err_127[i]<cutoff and err_127[i]>0])
            cut_off_var_127 = np.array([var_127[i] for i in range(len(err_127)) if err_127[i]<cutoff and err_127[i]>0])


            popt_127, pcov_127 = curve_fit(parabola,cut_off_p_127,cut_off_err_127,sigma=np.sqrt(cut_off_var_127),absolute_sigma=True,p0=1e9)

            with open(output_file_name, 'a') as file:
                file.write('\"a_127_'+above+'_'+error_model+'\":'+json.dumps({'val':popt_127[0], 'err':np.sqrt(pcov_127[0][0])})+',\n')

            fitting_127=popt_127[0]*p**2
            tmp.append(str(popt_127[0])+' \pm '+str(np.sqrt(pcov_127[0][0])))

            ax31[j].errorbar(p_127, err_127, yerr=yerr_log(err_127,var_127), linestyle="None", fmt='os^Dv'[i], capsize=4, label=label, color=color)
            ax31[j].plot(p, fitting_127, color=color)
            ax31[j].set_xlim([1e-10, 2e-3])
            ax31[j].set_ylim([1e-8, 1])
            ax31[j].loglog()
            if j==2:
                ax31[j].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

            ax31[j].set_xlabel('Physical error rate')
            ax31[j].set_ylabel('Logical CNOT error rate')
        store_a127.append(tmp)

    print(store_a7)

    plt.tight_layout()
    plt.savefig('../fig/hamming_logical_error_rate.pdf')
    plt.show()

def comparison_steane():
    error_model = 'c'
    p_steanel1, err_steanel1, var_steanel1 = load(f'../data/underlying_code/output_concatenated_steane_level1_error_model_{error_model}_10000000runs.json')
    p_steanel1_conventional, err_steanel1_conventional, var_steanel1_conventional = load(f'../data/underlying_code/output_steane_conventional_level1_error_model_{error_model}_1000000runs.json')

    popt, pcov = curve_fit(parabola,p_steanel1,err_steanel1,sigma=np.sqrt(var_steanel1),absolute_sigma=True)
    popt_conventional, pcov_conventional = curve_fit(parabola,p_steanel1_conventional,err_steanel1_conventional,sigma=np.sqrt(var_steanel1_conventional),absolute_sigma=True)

    p = np.array([1e-10,1])

    fitting=popt[0]*p**2
    fitting_conventional=popt_conventional[0]*p**2

    plt.errorbar(p_steanel1_conventional, err_steanel1_conventional, yerr=yerr_log(err_steanel1_conventional,var_steanel1_conventional), linestyle='None', fmt='o', color=color2, capsize=4, label='Conventional method')
    plt.plot(p,fitting_conventional, color=color2)
    plt.errorbar(p_steanel1, err_steanel1, yerr=yerr_log(err_steanel1,var_steanel1), linestyle='None', fmt='s', color=color4, capsize=4, label='[Goto (2016)]')
    plt.plot(p,fitting, color=color4)
    plt.loglog()
    plt.xlim([0.9e-5,1.1e-3])
    plt.ylim([2e-7,1.1e-2])
    plt.xlabel('Physical error rate')
    plt.ylabel('Logical CNOT error rate')
    plt.legend()

    plt.savefig('../fig/comparison_steane.pdf')

if __name__ == '__main__':
    output_file_name = 'estimated_parameters.json'
    with open(output_file_name, 'w') as file:
        file.write('{\n')

    estimate_underlying_code_threshold(output_file_name)
    estimate_hamming_code_threshold(output_file_name)
    comparison_steane()

    with open(output_file_name, 'r') as file:
        content = file.read()
    new_content = content[:-2]
    with open(output_file_name, 'w') as file:
        file.write(new_content)
        file.write('\n}')