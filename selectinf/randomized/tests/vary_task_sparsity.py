import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
from selectinf.randomized.tests.test_multitask_lasso_2 import test_coverage

k=5
p=100
global_sparsity = 0.95
#task_sparsity = 0.4

length_path = 15
lambdamin = 0
lambdamax = 4.0
#weights = np.arange(np.log(lambdamin), np.log(lambdamax), (np.log(lambdamax) - np.log(lambdamin)) / (length_path))
#feature_weight_list = np.exp(weights)
feature_weight_list = np.arange(lambdamin, lambdamax,(lambdamax - lambdamin) / (length_path))
print(feature_weight_list)

df = pd.DataFrame(columns=['Task Sparsity', 'Method', 'Coverage', 'Length'])


sparsity_list = [0.0,0.2,0.4,0.6]
#sparsity_list = [0.85,0.90,0.95,0.99]
n_list = [100,100,100,100]
##n_list = [5,5,5,20,20]
coverage_by_ts = {j: [[], [], [], [], [], [], []] for j in range(len(sparsity_list))}
length_by_ts = {j: [[], [], [], [], [], [], []] for j in range(len(sparsity_list))}
f1_by_ts = {j: [[], [], [], [], [], []] for j in range(len(sparsity_list))}


for j in range(len(sparsity_list)):
    positive = (1.-global_sparsity)*(1.-sparsity_list[j])*k*p
    #positive = (1.-task_sparsity)*(1.-sparsity_list[j])*k*p
    negative = k*p - positive

    selective_lengths = []
    selective_lengths2 = []
    naive_lengths = []
    ds_lengths = []
    ds_lengths2 = []
    single_selective_lengths = []
    single_selective_lengths2 = []

    selective_coverage = []
    selective_coverage2 = []
    naive_coverage = []
    ds_coverage = []
    ds_coverage2 = []
    single_selective_coverage = []
    single_selective_coverage2 = []

    selective_sensitivity = []
    selective_sensitivity2 = []
    naive_sensitivity = []
    ds_sensitivity = []
    ds_sensitivity2 = []
    single_task_sensitivity = []
    single_task_sensitivity2 = []

    selective_specificity = []
    selective_specificity2 = []
    naive_specificity = []
    ds_specificity = []
    ds_specificity2 = []
    single_task_specificity = []
    single_task_specificity2 = []

    selective_error = []
    selective_error2 = []
    naive_error = []
    ds_error = []
    ds_error2 = []
    single_selective_error = []
    single_selective_error2 = []

    for i in range(len(feature_weight_list)):
        print((i,j),"(i,j)")
        sims = test_coverage(feature_weight_list[i],[2.5,5.0],sparsity_list[j],nsim=n_list[j])
        selective_coverage.append(sims[3])
        selective_coverage2.append(sims[4])
        naive_coverage.append(sims[5])
        ds_coverage.append(sims[6])
        ds_coverage2.append(sims[7])
        single_selective_coverage.append(sims[8])
        single_selective_coverage2.append(sims[9])

        selective_lengths.append(sims[10])
        selective_lengths2.append(sims[11])
        naive_lengths.append(sims[12])
        ds_lengths.append(sims[13])
        ds_lengths2.append(sims[14])
        single_selective_lengths.append(sims[15])
        single_selective_lengths2.append(sims[16])

        selective_sensitivity.append(sims[17])
        selective_sensitivity2.append(sims[18])
        naive_sensitivity.append(sims[19])
        ds_sensitivity.append(sims[20])
        ds_sensitivity2.append(sims[21])
        single_task_sensitivity.append(sims[22])
        single_task_sensitivity2.append(sims[23])

        selective_specificity.append(sims[24])
        selective_specificity2.append(sims[25])
        naive_specificity.append(sims[26])
        ds_specificity.append(sims[27])
        ds_specificity2.append(sims[28])
        single_task_specificity.append(sims[29])
        single_task_specificity2.append(sims[30])

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

    selective_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(selective_specificity)[idx_min_random_multitask, :][n]
                     for n in range(n_list[j])]), np.asarray(selective_sensitivity)[idx_min_random_multitask, :]]).T
    selective_f1 = np.asarray(
        [2.0 * selective_tp_fp_mat[n, 1] * positive / (2.0 * selective_tp_fp_mat[n, 1] * positive +
                                                       selective_tp_fp_mat[n, 0] * negative + (1.0 -
                                                       selective_tp_fp_mat[n, 1]) * positive) for n in range(n_list[j])])

    f1_by_ts[j][0] = selective_f1

    selective2_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(selective_specificity2)[idx_min_random_multitask2, :][n]
                     for n in range(n_list[j])]), np.asarray(selective_sensitivity2)[idx_min_random_multitask, :]]).T
    selective2_f1 = np.asarray(
        [2.0 * selective2_tp_fp_mat[n, 1] * positive / (2.0 * selective2_tp_fp_mat[n, 1] * positive +
                                                       selective2_tp_fp_mat[n, 0] * negative + (1.0 -
                                                       selective2_tp_fp_mat[n, 1]) * positive) for n in range(n_list[j])])

    f1_by_ts[j][1] = selective2_f1

    ds_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(ds_specificity)[idx_min_data_splitting, :][n]
                     for n in range(n_list[j])]), np.asarray(ds_sensitivity)[idx_min_data_splitting, :]]).T
    ds_f1 = np.asarray(
        [2.0 * ds_tp_fp_mat[n, 1] * positive / (2.0 * ds_tp_fp_mat[n, 1] * positive +
                                                       ds_tp_fp_mat[n, 0] * negative + (1.0 -
                                                        ds_tp_fp_mat[ n, 1]) * positive) for n in range(n_list[j])])

    f1_by_ts[j][2] = ds_f1

    ds2_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(ds_specificity2)[idx_min_data_splitting2, :][n]
                     for n in range(n_list[j])]), np.asarray(ds_sensitivity2)[idx_min_data_splitting2, :]]).T
    ds2_f1 = np.asarray(
        [2.0 * ds2_tp_fp_mat[n, 1] * positive / (2.0 * ds2_tp_fp_mat[n, 1] * positive +
                                                ds2_tp_fp_mat[n, 0] * negative + (1.0 -
                                                                                 ds2_tp_fp_mat[n, 1]) * positive) for n
         in range(n_list[j])])

    f1_by_ts[j][3] = ds2_f1

    single_task_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(single_task_specificity)[idx_min_k_random_lasso, :][n]
                     for n in range(n_list[j])]), np.asarray(single_task_sensitivity)[idx_min_k_random_lasso, :]]).T
    single_task_f1 = np.asarray(
        [2.0 * single_task_tp_fp_mat[n, 1] * positive / (2.0 * single_task_tp_fp_mat[n, 1] * positive +
                                                single_task_tp_fp_mat[n, 0] * negative + (1.0 -
                                                                                 single_task_tp_fp_mat[n, 1]) * positive) for n
         in range(n_list[j])])

    f1_by_ts[j][4] = single_task_f1

    single_task2_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(single_task_specificity2)[idx_min_k_random_lasso2, :][n]
                     for n in range(n_list[j])]), np.asarray(single_task_sensitivity2)[idx_min_k_random_lasso2, :]]).T

    single_task2_f1 = np.asarray(
        [2.0 * single_task2_tp_fp_mat[n, 1] * positive / (2.0 * single_task2_tp_fp_mat[n, 1] * positive +
                                                         single_task2_tp_fp_mat[n, 0] * negative + (1.0 -
                                                                                                   single_task2_tp_fp_mat[
                                                                                                       n, 1]) * positive)
         for n
         in range(n_list[j])])

    f1_by_ts[j][5] = single_task2_f1

    print(np.mean(selective_f1),np.mean(selective2_f1),np.mean(ds_f1),np.mean(ds2_f1),np.mean(single_task_f1),np.mean(single_task2_f1),"F1 score means")


    coverage_by_ts[j][0] = selective_coverage[idx_min_random_multitask]
    length_by_ts[j][0] = selective_lengths[idx_min_random_multitask]

    coverage_by_ts[j][1] = selective_coverage2[idx_min_random_multitask2]
    length_by_ts[j][1] = selective_lengths2[idx_min_random_multitask2]

    coverage_by_ts[j][2] = naive_coverage[idx_min_naive_multitask]
    length_by_ts[j][2] = naive_lengths[idx_min_naive_multitask]

    coverage_by_ts[j][3] = ds_coverage[idx_min_data_splitting]
    length_by_ts[j][3] = ds_lengths[idx_min_data_splitting]

    coverage_by_ts[j][4] = ds_coverage2[idx_min_data_splitting2]
    length_by_ts[j][4] = ds_lengths2[idx_min_data_splitting2]

    coverage_by_ts[j][5] = single_selective_coverage[idx_min_k_random_lasso]
    length_by_ts[j][5] = single_selective_lengths[idx_min_k_random_lasso]

    coverage_by_ts[j][6] = single_selective_coverage2[idx_min_k_random_lasso2]
    length_by_ts[j][6] = single_selective_lengths2[idx_min_k_random_lasso2]

length = len(sparsity_list)
def set_box_color(bp, color,linestyle):
    plt.setp(bp['boxes'], color=color,linestyle=linestyle)
    plt.setp(bp['whiskers'], color=color,linestyle=linestyle)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

fig = plt.figure(figsize=(17,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
plt.sca(ax1)
first = plt.boxplot([coverage_by_ts[j][2] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3, sym='', widths=0.3)
second = plt.boxplot([coverage_by_ts[j][0] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + .3, sym='', widths=0.3)
third = plt.boxplot([coverage_by_ts[j][1] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 +.6, sym='', widths=0.3)
fourth = plt.boxplot([coverage_by_ts[j][3] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + .9, sym='', widths=0.3)
fifth = plt.boxplot([coverage_by_ts[j][4] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 1.2, sym='', widths=0.3)
sixth = plt.boxplot([coverage_by_ts[j][5] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 1.5, sym='',widths=0.3)
seventh = plt.boxplot([coverage_by_ts[j][6] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 1.8, sym='',widths=0.3)
set_box_color(first, '#D7191C','solid')
set_box_color(second, '#2b8cbe','solid')  # colors are from http://colorbrewer2.org/
set_box_color(third, '#6baed6','--')
set_box_color(fourth, '#238443','solid')
set_box_color(fifth, '#31a354','--')
set_box_color(sixth, '#fd8d3c','solid')
set_box_color(seventh,'#feb24c','--')
plt.xticks(range(1, (length) * 3 + 1, 3), [round(num, 1) for num in sparsity_list])
plt.xlim(-1, (length - 1) * 3 + 3)
plt.plot([], c='#D7191C', label='Naive',linewidth=2.5)
plt.plot([], c='#2b8cbe', label='Randomized Multi-Task Lasso 0.7',linewidth=2.5)
plt.plot([], c='#6baed6', label='Randomized Multi-Task Lasso 1.0',linestyle='--',linewidth=2.5)
plt.plot([], c='#238443', label='Data Splitting 67/33',linewidth=2.5)
plt.plot([], c='#31a354', label='Data Splitting 50/50',linestyle='--',linewidth=2.5)
plt.plot([], c='#fd8d3c', label='K Randomized Lassos 0.7',linewidth=2.5)
plt.plot([], c='#feb24c', label='K Randomized Lassos 1.0',linestyle='--',linewidth=2.5)
plt.legend()
plt.tight_layout()
plt.ylabel('Coverage per Simulation',fontsize=12)

plt.sca(ax2)
first = plt.boxplot([length_by_ts[j][0] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3, sym='', widths=0.3)
second = plt.boxplot([length_by_ts[j][1] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 +.3, sym='', widths=0.3)
fourth = plt.boxplot([length_by_ts[j][3] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + .6, sym='', widths=0.3)
fifth = plt.boxplot([length_by_ts[j][4] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 0.9, sym='', widths=0.3)
sixth = plt.boxplot([length_by_ts[j][5] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 1.2, sym='',widths=0.3)
seventh = plt.boxplot([length_by_ts[j][6] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 1.5, sym='',widths=0.3)
set_box_color(first, '#2b8cbe','solid')  # colors are from http://colorbrewer2.org/
set_box_color(second, '#6baed6','--')
set_box_color(fourth, '#238443','solid')
set_box_color(fifth, '#31a354','--')
set_box_color(sixth, '#fd8d3c','solid')
set_box_color(seventh,'#feb24c','--')
plt.xticks(range(1, (length) * 3 + 1, 3), [round(num, 1) for num in sparsity_list])
plt.xlim(-1, (length - 1) * 3 + 3)
plt.tight_layout()
plt.ylabel('Interval Length',fontsize=12)

plt.sca(ax3)
first = plt.boxplot([f1_by_ts[j][0] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3, sym='', widths=0.3)
second = plt.boxplot([f1_by_ts[j][1] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 +.3, sym='', widths=0.3)
fourth = plt.boxplot([f1_by_ts[j][2] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + .6, sym='', widths=0.3)
fifth = plt.boxplot([f1_by_ts[j][3] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 0.9, sym='', widths=0.3)
sixth = plt.boxplot([f1_by_ts[j][4] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 1.2, sym='',widths=0.3)
seventh = plt.boxplot([f1_by_ts[j][5] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 1.5, sym='',widths=0.3)
set_box_color(first, '#2b8cbe','solid')  # colors are from http://colorbrewer2.org/
set_box_color(second, '#6baed6','--')
set_box_color(fourth, '#238443','solid')
set_box_color(fifth, '#31a354','--')
set_box_color(sixth, '#fd8d3c','solid')
set_box_color(seventh,'#feb24c','--')
plt.xticks(range(1, (length) * 3 + 1, 3), [round(num, 1) for num in sparsity_list])
plt.xlim(-1, (length - 1) * 3 + 3)
plt.tight_layout()
plt.ylabel('f1 per Simulation',fontsize=12)


ax1.set_title("Coverage", y = 1.01)
ax2.set_title("Length", y = 1.01)
ax3.set_title("Accuracy", y = 1.01)
fig.suptitle("Regression Dimension p=100",fontsize=14)


def common_format(ax):
    ax.grid(True, which='both',color='#f0f0f0')
    ax.set_xlabel('Task Sparsity', fontsize=12)
    #ax.set_xlabel('Global Sparsity', fontsize=12)
    return ax

common_format(ax1)
common_format(ax2)
common_format(ax3)

# add target coverage on the first plot
ax1.axhline(y=0.9, color='k', linestyle='--', linewidth=2)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ax1.legend(loc='lower left', bbox_to_anchor=(0.6, -0.45),fontsize=14,ncol=3)
plt.savefig('vary_task_sparsity_p100.png', bbox_inches='tight')


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