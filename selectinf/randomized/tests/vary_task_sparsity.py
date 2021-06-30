import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
from selectinf.randomized.tests.test_multitask_lasso_2 import test_coverage

length_path = 10

lambdamin = 0.5
lambdamax = 4.0
#weights = np.arange(np.log(lambdamin), np.log(lambdamax), (np.log(lambdamax) - np.log(lambdamin)) / (length_path))
#feature_weight_list = np.exp(weights)
feature_weight_list = np.arange(lambdamin, lambdamax,(lambdamax - lambdamin) / (length_path))
print(feature_weight_list)

df = pd.DataFrame(columns=['Task Sparsity', 'Method', 'Coverage', 'Length'])


task_sparsity_list = [0.0, 0.2, 0.4, 0.6, 0.8]
n_list = [100,100,100,400,400]
##n_list = [5,5,5,20,20]
coverage_by_ts = {j: [[], [], [], [], [], [], []] for j in range(len(task_sparsity_list))}
length_by_ts = {j: [[], [], [], [], [], [], []] for j in range(len(task_sparsity_list))}
sensitivity_by_ts = {j: [[], [], [], [], [], [], []] for j in range(len(task_sparsity_list))}
specificity_by_ts = {j: [[], [], [], [], [], [], []] for j in range(len(task_sparsity_list))}

for j in range(len(task_sparsity_list)):
    coverage = {i: [[], [], [], [], [], [], []] for i in range(length_path)}
    length = {i: [[], [], [], [], [], [], []] for i in range(length_path)}
    sensitivity = {i: [[], [], [], [], [], [], []] for i in range(length_path)}
    specificity = {i: [[], [], [], [], [], [], []] for i in range(length_path)}
    error = {i: [[], [], [], [], [], [], []] for i in range(length_path)}

    for i in range(len(feature_weight_list)):
        print((i,j),"(i,j)")
        sims = test_coverage(feature_weight_list[i],[2.0,5.0],ts=task_sparsity_list[j],nsim=n_list[j])
        coverage[i][0].extend(sims[3])
        coverage[i][1].extend(sims[4])
        coverage[i][2].extend(sims[5])
        coverage[i][3].extend(sims[6])
        coverage[i][4].extend(sims[7])
        coverage[i][5].extend(sims[8])
        coverage[i][6].extend(sims[9])
        length[i][0].extend(sims[10])
        length[i][1].extend(sims[11])
        length[i][2].extend(sims[12])
        length[i][3].extend(sims[13])
        length[i][4].extend(sims[14])
        length[i][5].extend(sims[15])
        length[i][6].extend(sims[16])
        sensitivity[i][0].extend(sims[17])
        sensitivity[i][1].extend(sims[18])
        sensitivity[i][2].extend(sims[19])
        sensitivity[i][3].extend(sims[20])
        sensitivity[i][4].extend(sims[21])
        sensitivity[i][5].extend(sims[22])
        sensitivity[i][6].extend(sims[23])
        specificity[i][0].extend(sims[24])
        specificity[i][1].extend(sims[25])
        specificity[i][2].extend(sims[26])
        specificity[i][3].extend(sims[27])
        specificity[i][4].extend(sims[28])
        specificity[i][5].extend(sims[29])
        specificity[i][6].extend(sims[30])
        error[i][0].append(sims[31])
        error[i][1].append(sims[32])
        error[i][2].append(sims[33])
        error[i][3].append(sims[34])
        error[i][4].append(sims[35])
        error[i][5].append(sims[36])
        error[i][6].append(sims[37])

    selective_lengths = [length[i][0] for i in range(length_path)]
    selective_lengths2 = [length[i][1] for i in range(length_path)]
    naive_lengths = [length[i][2] for i in range(length_path)]
    ds_lengths = [length[i][3] for i in range(length_path)]
    ds_lengths2 = [length[i][4] for i in range(length_path)]
    single_selective_lengths = [length[i][5] for i in range(length_path)]
    single_selective_lengths2 = [length[i][6] for i in range(length_path)]

    selective_coverage = [coverage[i][0] for i in range(length_path)]
    selective_coverage2 = [coverage[i][1] for i in range(length_path)]
    naive_coverage = [coverage[i][2] for i in range(length_path)]
    ds_coverage = [coverage[i][3] for i in range(length_path)]
    ds_coverage2 = [coverage[i][4] for i in range(length_path)]
    single_selective_coverage = [coverage[i][5] for i in range(length_path)]
    single_selective_coverage2 = [coverage[i][6] for i in range(length_path)]

    selective_sensitivity = [sensitivity[i][0] for i in range(length_path)]
    selective_sensitivity2 = [sensitivity[i][1] for i in range(length_path)]
    naive_sensitivity = [sensitivity[i][2] for i in range(length_path)]
    ds_sensitivity = [sensitivity[i][3] for i in range(length_path)]
    ds_sensitivity2 = [sensitivity[i][4] for i in range(length_path)]
    single_task_sensitivity = [sensitivity[i][5] for i in range(length_path)]
    single_task_sensitivity2 = [sensitivity[i][6] for i in range(length_path)]

    selective_specificity = [specificity[i][0] for i in range(length_path)]
    selective_specificity2 = [specificity[i][1] for i in range(length_path)]
    naive_specificity = [specificity[i][2] for i in range(length_path)]
    ds_specificity = [specificity[i][3] for i in range(length_path)]
    ds_specificity2 = [specificity[i][4] for i in range(length_path)]
    single_task_specificity = [specificity[i][5] for i in range(length_path)]
    single_task_specificity2 = [specificity[i][6] for i in range(length_path)]

    selective_error = [error[i][0] for i in range(length_path)]
    selective_error2 = [error[i][1] for i in range(length_path)]
    naive_error = [error[i][2] for i in range(length_path)]
    ds_error = [error[i][3] for i in range(length_path)]
    ds_error2 = [error[i][4] for i in range(length_path)]
    single_selective_error = [error[i][5] for i in range(length_path)]
    single_selective_error2 = [error[i][6] for i in range(length_path)]

    idx_min_random_multitask = np.argmin(selective_error)
    idx_min_random_multitask2 = np.argmin(selective_error2)
    idx_min_naive_multitask = np.argmin(naive_error)
    idx_min_data_splitting = np.argmin(ds_error)
    idx_min_data_splitting2 = np.argmin(ds_error2)
    idx_min_k_random_lasso = np.argmin(single_selective_error)
    idx_min_k_random_lasso2 = np.argmin(single_selective_error2)

    coverage_by_ts[j][0] = selective_coverage[idx_min_random_multitask]
    length_by_ts[j][0] = selective_lengths[idx_min_random_multitask]
    sensitivity_by_ts[j][0] = selective_sensitivity[idx_min_random_multitask]
    specificity_by_ts[j][0] = selective_specificity[idx_min_random_multitask]
    coverage_by_ts[j][1] = selective_coverage2[idx_min_random_multitask2]
    length_by_ts[j][1] = selective_lengths2[idx_min_random_multitask2]
    sensitivity_by_ts[j][1] = selective_sensitivity2[idx_min_random_multitask2]
    specificity_by_ts[j][1] = selective_specificity2[idx_min_random_multitask2]
    coverage_by_ts[j][2] = naive_coverage[idx_min_naive_multitask]
    length_by_ts[j][2] = naive_lengths[idx_min_naive_multitask]
    sensitivity_by_ts[j][2] = naive_sensitivity[idx_min_naive_multitask]
    specificity_by_ts[j][2] = naive_specificity[idx_min_naive_multitask]
    coverage_by_ts[j][3] = ds_coverage[idx_min_data_splitting]
    length_by_ts[j][3] = ds_lengths[idx_min_data_splitting]
    sensitivity_by_ts[j][3] = ds_sensitivity[idx_min_data_splitting]
    specificity_by_ts[j][3] = ds_specificity[idx_min_data_splitting]
    coverage_by_ts[j][4] = ds_coverage2[idx_min_data_splitting2]
    length_by_ts[j][4] = ds_lengths2[idx_min_data_splitting2]
    sensitivity_by_ts[j][4] = ds_sensitivity2[idx_min_data_splitting2]
    specificity_by_ts[j][4] = ds_specificity2[idx_min_data_splitting2]
    coverage_by_ts[j][5] = single_selective_coverage[idx_min_k_random_lasso]
    length_by_ts[j][5] = single_selective_lengths[idx_min_k_random_lasso]
    sensitivity_by_ts[j][5] = single_task_sensitivity[idx_min_k_random_lasso]
    specificity_by_ts[j][5] = single_task_specificity[idx_min_k_random_lasso]
    coverage_by_ts[j][6] = single_selective_coverage2[idx_min_k_random_lasso2]
    length_by_ts[j][6] = single_selective_lengths2[idx_min_k_random_lasso2]
    sensitivity_by_ts[j][6] = single_task_sensitivity2[idx_min_k_random_lasso2]
    specificity_by_ts[j][6] = single_task_specificity2[idx_min_k_random_lasso2]

length = len(task_sparsity_list)
def set_box_color(bp, color,linestyle):
    plt.setp(bp['boxes'], color=color,linestyle=linestyle)
    plt.setp(bp['whiskers'], color=color,linestyle=linestyle)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

fig = plt.figure(figsize=(17,14))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

plt.sca(ax1)
first = plt.boxplot([coverage_by_ts[j][0] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3, sym='', widths=0.3)
second = plt.boxplot([coverage_by_ts[j][1] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 +.3, sym='', widths=0.3)
third = plt.boxplot([coverage_by_ts[j][2] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 + .6, sym='', widths=0.3)
fourth = plt.boxplot([coverage_by_ts[j][3] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 + .9, sym='', widths=0.3)
fifth = plt.boxplot([coverage_by_ts[j][4] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 + 1.2, sym='', widths=0.3)
sixth = plt.boxplot([coverage_by_ts[j][5] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 + 1.5, sym='',widths=0.3)
seventh = plt.boxplot([coverage_by_ts[j][6] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 + 1.8, sym='',widths=0.3)
set_box_color(first, '#2b8cbe','solid')  # colors are from http://colorbrewer2.org/
set_box_color(second, '#6baed6','--')
set_box_color(third, '#D7191C','solid')
set_box_color(fourth, '#238443','solid')
set_box_color(fifth, '#31a354','--')
set_box_color(sixth, '#fd8d3c','solid')
set_box_color(seventh,'#feb24c','--')
plt.xticks(xrange(1, (length) * 3 + 1, 3), [round(num, 1) for num in task_sparsity_list])
plt.xlim(-1, (length - 1) * 3 + 3)
plt.tight_layout()
plt.ylabel('Mean Coverage per Simulation',fontsize=12)

plt.sca(ax2)
first = plt.boxplot([length_by_ts[j][0] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3, sym='', widths=0.3)
second = plt.boxplot([length_by_ts[j][1] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 +.3, sym='', widths=0.3)
third = plt.boxplot([length_by_ts[j][2] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 + .6, sym='', widths=0.3)
fourth = plt.boxplot([length_by_ts[j][3] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 + .9, sym='', widths=0.3)
fifth = plt.boxplot([length_by_ts[j][4] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 + 1.2, sym='', widths=0.3)
sixth = plt.boxplot([length_by_ts[j][5] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 + 1.5, sym='',widths=0.3)
seventh = plt.boxplot([length_by_ts[j][6] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 + 1.8, sym='',widths=0.3)
set_box_color(first, '#2b8cbe','solid')  # colors are from http://colorbrewer2.org/
set_box_color(second, '#6baed6','--')
set_box_color(third, '#D7191C','solid')
set_box_color(fourth, '#238443','solid')
set_box_color(fifth, '#31a354','--')
set_box_color(sixth, '#fd8d3c','solid')
set_box_color(seventh,'#feb24c','--')
plt.xticks(xrange(1, (length) * 3 + 1, 3), [round(num, 1) for num in task_sparsity_list])
plt.xlim(-1, (length - 1) * 3 + 3)
plt.tight_layout()
plt.ylabel('Interval Length',fontsize=12)

print([np.std(sensitivity_by_ts[j][0]) for j in range(len(task_sparsity_list))])
plt.sca(ax3)
plt.errorbar(task_sparsity_list, [np.mean(sensitivity_by_ts[j][0]) for j in range(len(task_sparsity_list))], yerr=[np.std(sensitivity_by_ts[j][0])/np.sqrt(n_list[j]) for j in range(len(task_sparsity_list))] , c='#2b8cbe')
plt.errorbar(task_sparsity_list, [np.mean(sensitivity_by_ts[j][1]) for j in range(len(task_sparsity_list))], yerr=[np.std(sensitivity_by_ts[j][1])/np.sqrt(n_list[j]) for j in range(len(task_sparsity_list))] , c='#6baed6',linestyle='--')
plt.errorbar(task_sparsity_list, [np.mean(sensitivity_by_ts[j][3]) for j in range(len(task_sparsity_list))], yerr=[np.std(sensitivity_by_ts[j][3])/np.sqrt(n_list[j]) for j in range(len(task_sparsity_list))] , c='#238443')
plt.errorbar(task_sparsity_list, [np.mean(sensitivity_by_ts[j][4]) for j in range(len(task_sparsity_list))], yerr=[np.std(sensitivity_by_ts[j][4])/np.sqrt(n_list[j]) for j in range(len(task_sparsity_list))] , c='#31a354',linestyle='--')
plt.errorbar(task_sparsity_list, [np.mean(sensitivity_by_ts[j][5]) for j in range(len(task_sparsity_list))], yerr=[np.std(sensitivity_by_ts[j][5])/np.sqrt(n_list[j]) for j in range(len(task_sparsity_list))] , c='#fd8d3c')
plt.errorbar(task_sparsity_list, [np.mean(sensitivity_by_ts[j][6]) for j in range(len(task_sparsity_list))], yerr=[np.std(sensitivity_by_ts[j][6])/np.sqrt(n_list[j]) for j in range(len(task_sparsity_list))] , c='#feb24c',linestyle='--')
plt.plot([], c='#D7191C', label='Multi-Task Lasso',linewidth=2.5)
plt.plot([], c='#fd8d3c', label='K Randomized Lassos 0.7',linewidth=2.5)
plt.plot([], c='#feb24c', label='K Randomized Lassos 1.0',linestyle='--',linewidth=2.5)
plt.plot([], c='#238443', label='Data Splitting 67/33',linewidth=2.5)
plt.plot([], c='#31a354', label='Data Splitting 50/50',linestyle='--',linewidth=2.5)
plt.plot([], c='#2b8cbe', label='Randomized Multi-Task Lasso 0.7',linewidth=2.5)
plt.plot([], c='#6baed6', label='Randomized Multi-Task Lasso 1.0',linestyle='--',linewidth=2.5)
plt.legend()
plt.tight_layout()
plt.ylabel('Mean Sensivity per Simulation',fontsize=12)

plt.sca(ax4)
plt.errorbar(task_sparsity_list, [np.mean(specificity_by_ts[j][0]) for j in range(len(task_sparsity_list))], yerr=[np.std(specificity_by_ts[j][0])/np.sqrt(n_list[j]) for j in range(len(task_sparsity_list))] , c='#2b8cbe')
plt.errorbar(task_sparsity_list, [np.mean(specificity_by_ts[j][1]) for j in range(len(task_sparsity_list))], yerr=[np.std(specificity_by_ts[j][1])/np.sqrt(n_list[j]) for j in range(len(task_sparsity_list))] , c='#6baed6',linestyle='--')
plt.errorbar(task_sparsity_list, [np.mean(specificity_by_ts[j][3]) for j in range(len(task_sparsity_list))], yerr=[np.std(specificity_by_ts[j][3])/np.sqrt(n_list[j]) for j in range(len(task_sparsity_list))] , c='#238443')
plt.errorbar(task_sparsity_list, [np.mean(specificity_by_ts[j][4]) for j in range(len(task_sparsity_list))], yerr=[np.std(specificity_by_ts[j][4])/np.sqrt(n_list[j]) for j in range(len(task_sparsity_list))] , c='#31a354',linestyle='--')
plt.errorbar(task_sparsity_list, [np.mean(specificity_by_ts[j][5]) for j in range(len(task_sparsity_list))], yerr=[np.std(specificity_by_ts[j][5])/np.sqrt(n_list[j]) for j in range(len(task_sparsity_list))] , c='#fd8d3c')
plt.errorbar(task_sparsity_list, [np.mean(specificity_by_ts[j][6]) for j in range(len(task_sparsity_list))], yerr=[np.std(specificity_by_ts[j][6])/np.sqrt(n_list[j]) for j in range(len(task_sparsity_list))] , c='#feb24c')
plt.ylim((.98,1.001))
plt.tight_layout()
plt.ylabel('Mean Specificty per Simulation',fontsize=12)

ax1.set_title("Coverage", y = 1.01)
ax2.set_title("Length", y = 1.01)
ax3.set_title("Sensivity", y = 1.01)
ax4.set_title("Specificity", y = 1.01)


ax3.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.6),fontsize=14)

def common_format(ax):
    ax.grid(True, which='both',color='#f0f0f0')
    ax.set_xlabel('Task Sparsity', fontsize=12)
    return ax

common_format(ax1)
common_format(ax2)
common_format(ax3)
common_format(ax4)

# add target coverage on the first plot
ax1.axhline(y=0.9, color='k', linestyle='--', linewidth=2)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('cov_len_by_ts_mixed_2_5_n100_400.png', bbox_inches='tight')


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