import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
from selectinf.randomized.tests.test_multitask_lasso import test_coverage

k=5
#global_sparsity = [0.9167,0.9667,0.9833,0.9888]
global_sparsity = [0.9375,0.975,0.9875,0.99167]
task_sparsity = 0.2
#task_sparsity = 0.4

length_path = 8
lambdamin = 0.5
lambdamax = 4.0
feature_weight_list = np.arange(lambdamin, lambdamax,(lambdamax - lambdamin) / (length_path))
print(feature_weight_list)

df = pd.DataFrame(columns=['Task Sparsity', 'Method', 'Coverage', 'Length'])


p_list = [100,250,500,750]
n_list = [100,100,100,100]
coverage_by_p = {j: [[], [], [], [], [], [], []] for j in range(len(p_list))}
length_by_p = {j: [[], [], [], [], [], [], []] for j in range(len(p_list))}
f1_by_p = {j: [[], [], [], [], [], []] for j in range(len(p_list))}


for j in range(len(p_list)):
    positive = (1.-global_sparsity[j])*(1.-task_sparsity)*k*p_list[j]
    negative = k*p_list[j] - positive

    selective_error = []
    selective_error2 = []
    naive_error = []
    ds_error = []
    ds_error2 = []
    single_selective_error = []
    single_selective_error2 = []

    for i in range(len(feature_weight_list)):
        print((i,j),"(i,j)")
        weight = [feature_weight_list[i]]*7
        print(weight)
        sims = test_coverage(weight,[1.0,3.0],p_list[j],task_sparsity,global_sparsity[j],nsim=1)
        selective_error.append(sims[31])
        selective_error2.append(sims[32])
        naive_error.append(sims[33])
        ds_error.append(sims[34])
        ds_error2.append(sims[35])
        single_selective_error.append(sims[36])
        single_selective_error2.append(sims[37])

    idx_min_random_multitask = np.argmin(selective_error)
    idx_min_random_multitask2 = np.argmin(selective_error2)
    idx_min_naive_multitask = np.argmin(naive_error)
    idx_min_data_splitting = np.argmin(ds_error)
    idx_min_data_splitting2 = np.argmin(ds_error2)
    idx_min_k_random_lasso = np.argmin(single_selective_error)
    idx_min_k_random_lasso2 = np.argmin(single_selective_error2)

    feature_weight_list2 = [feature_weight_list[idx_min_random_multitask],feature_weight_list[idx_min_random_multitask2],
                           feature_weight_list[idx_min_naive_multitask], feature_weight_list[idx_min_data_splitting],
                           feature_weight_list[idx_min_data_splitting2],feature_weight_list[idx_min_k_random_lasso],
                           feature_weight_list[idx_min_k_random_lasso2]]


    sims = test_coverage(feature_weight_list2,[1.0,3.0],p_list[j],task_sparsity,global_sparsity[j],nsim=n_list[j])
    selective_coverage = sims[3]
    selective_coverage2 = sims[4]
    naive_coverage = sims[5]
    ds_coverage = sims[6]
    ds_coverage2 = sims[7]
    single_selective_coverage = sims[8]
    single_selective_coverage2 = sims[9]

    selective_lengths = sims[10]
    selective_lengths2 = sims[11]
    naive_lengths = sims[12]
    ds_lengths = sims[13]
    ds_lengths2 = sims[14]
    single_selective_lengths = sims[15]
    single_selective_lengths2 = sims[16]

    selective_sensitivity = sims[17]
    selective_sensitivity2 = sims[18]
    naive_sensitivity = sims[19]
    ds_sensitivity = sims[20]
    ds_sensitivity2 = sims[21]
    single_task_sensitivity = sims[22]
    single_task_sensitivity2 = sims[23]

    selective_specificity = sims[24]
    selective_specificity2 = sims[25]
    naive_specificity = sims[26]
    ds_specificity = sims[27]
    ds_specificity2 = sims[28]
    single_task_specificity = sims[29]
    single_task_specificity2 = sims[30]


    selective_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(selective_specificity)[n] for n in range(n_list[j])]), np.asarray(selective_sensitivity)]).T
    selective_f1 = np.asarray(
        [2.0 * selective_tp_fp_mat[n, 1] * positive / (2.0 * selective_tp_fp_mat[n, 1] * positive +
                                                       selective_tp_fp_mat[n, 0] * negative + (1.0 -
                                                       selective_tp_fp_mat[n, 1]) * positive) for n in range(n_list[j])])

    f1_by_p[j][0] = selective_f1

    selective2_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(selective_specificity2)[n] for n in range(n_list[j])]), np.asarray(selective_sensitivity2)]).T
    selective2_f1 = np.asarray(
        [2.0 * selective2_tp_fp_mat[n, 1] * positive / (2.0 * selective2_tp_fp_mat[n, 1] * positive +
                                                       selective2_tp_fp_mat[n, 0] * negative + (1.0 -
                                                       selective2_tp_fp_mat[n, 1]) * positive) for n in range(n_list[j])])

    f1_by_p[j][1] = selective2_f1

    ds_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(ds_specificity)[n]
                     for n in range(n_list[j])]), np.asarray(ds_sensitivity)]).T
    ds_f1 = np.asarray(
        [2.0 * ds_tp_fp_mat[n, 1] * positive / (2.0 * ds_tp_fp_mat[n, 1] * positive +
                                                       ds_tp_fp_mat[n, 0] * negative + (1.0 -
                                                        ds_tp_fp_mat[ n, 1]) * positive) for n in range(n_list[j])])

    f1_by_p[j][2] = ds_f1

    ds2_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(ds_specificity2)[n]
                     for n in range(n_list[j])]), np.asarray(ds_sensitivity2)]).T
    ds2_f1 = np.asarray(
        [2.0 * ds2_tp_fp_mat[n, 1] * positive / (2.0 * ds2_tp_fp_mat[n, 1] * positive +
                                                ds2_tp_fp_mat[n, 0] * negative + (1.0 -
                                                                                 ds2_tp_fp_mat[n, 1]) * positive) for n
         in range(n_list[j])])

    f1_by_p[j][3] = ds2_f1

    single_task_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(single_task_specificity)[n]
                     for n in range(n_list[j])]), np.asarray(single_task_sensitivity)]).T
    single_task_f1 = np.asarray(
        [2.0 * single_task_tp_fp_mat[n, 1] * positive / (2.0 * single_task_tp_fp_mat[n, 1] * positive +
                                                single_task_tp_fp_mat[n, 0] * negative + (1.0 -
                                                                                 single_task_tp_fp_mat[n, 1]) * positive) for n
         in range(n_list[j])])

    f1_by_p[j][4] = single_task_f1

    single_task2_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(single_task_specificity2)[n]
                     for n in range(n_list[j])]), np.asarray(single_task_sensitivity2)]).T

    single_task2_f1 = np.asarray(
        [2.0 * single_task2_tp_fp_mat[n, 1] * positive / (2.0 * single_task2_tp_fp_mat[n, 1] * positive +
                                                         single_task2_tp_fp_mat[n, 0] * negative + (1.0 -
                                                                                                   single_task2_tp_fp_mat[
                                                                                                       n, 1]) * positive)
         for n
         in range(n_list[j])])

    f1_by_p[j][5] = single_task2_f1

    print(np.mean(selective_f1),np.mean(selective2_f1),np.mean(ds_f1),np.mean(ds2_f1),np.mean(single_task_f1),np.mean(single_task2_f1),"F1 score means")


    coverage_by_p[j][0] = selective_coverage
    length_by_p[j][0] = selective_lengths

    coverage_by_p[j][1] = selective_coverage2
    length_by_p[j][1] = selective_lengths2

    coverage_by_p[j][2] = naive_coverage
    length_by_p[j][2] = naive_lengths

    coverage_by_p[j][3] = ds_coverage
    length_by_p[j][3] = ds_lengths

    coverage_by_p[j][4] = ds_coverage2
    length_by_p[j][4] = ds_lengths2

    coverage_by_p[j][5] = single_selective_coverage
    length_by_p[j][5] = single_selective_lengths

    coverage_by_p[j][6] = single_selective_coverage2
    length_by_p[j][6] = single_selective_lengths2

length = len(p_list)
def set_box_color(bp, color,linestyle):
    plt.setp(bp['boxes'], color=color,linestyle=linestyle,linewidth=2)
    plt.setp(bp['whiskers'], color=color,linestyle=linestyle,linewidth=2)
    plt.setp(bp['caps'], color=color,linewidth=2)
    plt.setp(bp['medians'], color=color,linewidth=2)

fig = plt.figure(figsize=(17,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
plt.sca(ax1)
first = plt.boxplot([coverage_by_p[j][0] for j in range(len(p_list))], positions=np.array(range(length)) * 3 , sym='', widths=0.3)
second = plt.boxplot([coverage_by_p[j][1] for j in range(len(p_list))], positions=np.array(range(length)) * 3 +.3, sym='', widths=0.3)
third = plt.boxplot([coverage_by_p[j][3] for j in range(len(p_list))], positions=np.array(range(length)) * 3 + .6, sym='', widths=0.3)
fourth = plt.boxplot([coverage_by_p[j][4] for j in range(len(p_list))], positions=np.array(range(length)) * 3 + 0.9, sym='', widths=0.3)
fifth = plt.boxplot([coverage_by_p[j][5] for j in range(len(p_list))], positions=np.array(range(length)) * 3 + 1.2, sym='',widths=0.3)
sixth = plt.boxplot([coverage_by_p[j][6] for j in range(len(p_list))], positions=np.array(range(length)) * 3 + 1.5, sym='',widths=0.3)
set_box_color(first, '#2b8cbe','solid')  # colors are from http://colorbrewer2.org/
set_box_color(second, '#6baed6','--')
set_box_color(third, '#238443','solid')
set_box_color(fourth, '#31a354','--')
set_box_color(fifth, '#fd8d3c','solid')
set_box_color(sixth,'#feb24c','--')
plt.xticks(range(1, (length) * 3 + 1, 3), p_list,fontsize=14)
plt.xlim(-1, (length - 1) * 3 + 3)
plt.plot([], c='#2b8cbe', label='MTL (0.7) + SI',linewidth=2.5)
plt.plot([], c='#6baed6', label='MTL (1.0) + SI',linestyle='--',linewidth=2.5)
plt.plot([], c='#238443', label='DS (0.67)',linewidth=2.5)
plt.plot([], c='#31a354', label='DS (0.5)',linestyle='--',linewidth=2.5)
plt.plot([], c='#fd8d3c', label='LASSO (0.7) + SI',linewidth=2.5)
plt.plot([], c='#feb24c', label='LASSO (1.0) + SI',linestyle='--',linewidth=2.5)
plt.legend()
plt.tight_layout()
plt.ylabel('Coverage per Simulation',fontsize=18)
plt.yticks(fontsize=14)

plt.sca(ax2)
first = plt.boxplot([length_by_p[j][0] for j in range(len(p_list))], positions=np.array(range(length)) * 3, sym='', widths=0.3)
second = plt.boxplot([length_by_p[j][1] for j in range(len(p_list))], positions=np.array(range(length)) * 3 +.3, sym='', widths=0.3)
third = plt.boxplot([length_by_p[j][3] for j in range(len(p_list))], positions=np.array(range(length)) * 3 + .6, sym='', widths=0.3)
fourth = plt.boxplot([length_by_p[j][4] for j in range(len(p_list))], positions=np.array(range(length)) * 3 + 0.9, sym='', widths=0.3)
fifth = plt.boxplot([length_by_p[j][5] for j in range(len(p_list))], positions=np.array(range(length)) * 3 + 1.2, sym='',widths=0.3)
sixth = plt.boxplot([length_by_p[j][6] for j in range(len(p_list))], positions=np.array(range(length)) * 3 + 1.5, sym='',widths=0.3)
set_box_color(first, '#2b8cbe','solid')  # colors are from http://colorbrewer2.org/
set_box_color(second, '#6baed6','--')
set_box_color(third, '#238443','solid')
set_box_color(fourth, '#31a354','--')
set_box_color(fifth, '#fd8d3c','solid')
set_box_color(sixth,'#feb24c','--')
plt.xticks(range(1, (length) * 3 + 1, 3), p_list,fontsize=14)
plt.xlim(-1, (length - 1) * 3 + 3)
plt.tight_layout()
plt.ylabel('Interval Length',fontsize=18)
plt.yticks(fontsize=14)

plt.sca(ax3)
first = plt.boxplot([f1_by_p[j][0] for j in range(len(p_list))], positions=np.array(range(length)) * 3, sym='', widths=0.3)
second = plt.boxplot([f1_by_p[j][1] for j in range(len(p_list))], positions=np.array(range(length)) * 3 +.3, sym='', widths=0.3)
third = plt.boxplot([f1_by_p[j][2] for j in range(len(p_list))], positions=np.array(range(length)) * 3 + .6, sym='', widths=0.3)
fourth = plt.boxplot([f1_by_p[j][3] for j in range(len(p_list))], positions=np.array(range(length)) * 3 + 0.9, sym='', widths=0.3)
fifth = plt.boxplot([f1_by_p[j][4] for j in range(len(p_list))], positions=np.array(range(length)) * 3 + 1.2, sym='',widths=0.3)
sixth = plt.boxplot([f1_by_p[j][5] for j in range(len(p_list))], positions=np.array(range(length)) * 3 + 1.5, sym='',widths=0.3)
set_box_color(first, '#2b8cbe','solid')  # colors are from http://colorbrewer2.org/
set_box_color(second, '#6baed6','--')
set_box_color(third, '#238443','solid')
set_box_color(fourth, '#31a354','--')
set_box_color(fifth, '#fd8d3c','solid')
set_box_color(sixth,'#feb24c','--')
plt.xticks(range(1, (length) * 3 + 1, 3), p_list,fontsize=14)
plt.xlim(-1, (length - 1) * 3 + 3)
plt.tight_layout()
plt.ylabel('f1 per Simulation',fontsize=18)
plt.yticks(fontsize=14)

ax1.set_title("Coverage", y = 1.01,fontsize=20)
ax2.set_title("Length", y = 1.01,fontsize=20)
ax3.set_title("Accuracy", y = 1.01,fontsize=20)


def common_format(ax):
    ax.grid(True, which='both',color='#f0f0f0')
    ax.set_xlabel('p', fontsize=18)
    return ax

common_format(ax1)
common_format(ax2)
common_format(ax3)

# add target coverage on the first plot
ax1.axhline(y=0.9, color='k', linestyle='--', linewidth=2)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ax1.legend(loc='lower left', bbox_to_anchor=(0.6, -0.45),fontsize=18,ncol=3)
plt.savefig('vary_p2.png', bbox_inches='tight')


#fig = plt.figure(figsize=(25, 10))
#fig.tight_layout()
#fig.add_subplot(1, 2, 1)
#plt.plot(task_sparsity_list, random_multitask_sensitivity, c='#D7191C')
#plt.plot(task_sparsity_list, naive_multitask_sensitivity, c='#2b8cbe')
#plt.plot(task_sparsity_list, data_splitting_sensitivity, c='#31a354')
#plt.plot(task_sparsity_list, k_random_lasso_sensitivity, c='#c51b8a')
#plt.plot([], c='#D7191C', label='Randomized Multi-Task Lasso')
#plt.plot([], c='#2b8cbe', label='Multi-Task Lasso')
#plt.plot([], c='#31a354', label='Data Splitting')
#plt.plot([], c='#c51b8a', label='K Randomized Lassos')
#plt.plot([], c='#feb24c', label='One Randomized Lasso')
#plt.legend()
#plt.tight_layout()
#plt.ylabel('Average Sensitivity')
#plt.xlabel('Task Sparsity')
#plt.title('Sensitivity by Task Sparsity')
#fig.add_subplot(1, 2, 2)
#plt.plot(task_sparsity_list, random_multitask_specificity, c='#D7191C')
#plt.plot(task_sparsity_list, naive_multitask_specificity, c='#2b8cbe')
#plt.plot(task_sparsity_list, data_splitting_specificity, c='#31a354')
#plt.plot(task_sparsity_list, k_random_lasso_specificity, c='#c51b8a')
#plt.plot([], c='#D7191C', label='Randomized Multi-Task Lasso')
#plt.plot([], c='#2b8cbe', label='Multi-Task Lasso')
#plt.plot([], c='#31a354', label='Data Splitting')
#plt.plot([], c='#c51b8a', label='K Randomized Lassos')
#plt.plot([], c='#feb24c', label='One Randomized Lasso')
#plt.legend()
#plt.tight_layout()
#plt.ylabel('Average Specificity')
#plt.xlabel('Task Sparsity')
#plt.title('Specificity by Task Sparsity')
#plt.savefig('model_selection_compare_weak.png')