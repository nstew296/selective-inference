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
coverage_by_ts = {j: [[], [], [], []] for j in range(len(task_sparsity_list))}
length_by_ts = {j: [[], [], [], []] for j in range(len(task_sparsity_list))}
sensitivity_by_ts = {j: [[], [], [], []] for j in range(len(task_sparsity_list))}
specificity_by_ts = {j: [[], [], [], []] for j in range(len(task_sparsity_list))}

for j in range(len(task_sparsity_list)):
    coverage = {i: [[], [], [], []] for i in range(length_path)}
    length = {i: [[], [], [], []] for i in range(length_path)}
    sensitivity = {i: [[], [], [], []] for i in range(length_path)}
    specificity = {i: [[], [], [], []] for i in range(length_path)}
    error = {i: [[], [], [], []] for i in range(length_path)}

    for i in range(len(feature_weight_list)):
        print((i,j),"(i,j)")
        sims = test_coverage(feature_weight_list[i],[3.0,5.0],ts=task_sparsity_list[j],nsim=30)
        coverage[i][0].extend(sims[3])
        coverage[i][1].extend(sims[4])
        coverage[i][2].extend(sims[5])
        coverage[i][3].extend(sims[6])
        length[i][0].extend(sims[7])
        length[i][1].extend(sims[8])
        length[i][2].extend(sims[9])
        length[i][3].extend(sims[10])
        sensitivity[i][0].append(sims[11])
        sensitivity[i][1].append(sims[12])
        sensitivity[i][2].append(sims[13])
        sensitivity[i][3].append(sims[14])
        specificity[i][0].append(sims[15])
        specificity[i][1].append(sims[16])
        specificity[i][2].append(sims[17])
        specificity[i][3].append(sims[18])
        error[i][0].append(sims[19])
        error[i][1].append(sims[20])
        error[i][2].append(sims[21])
        error[i][3].append(sims[22])

    selective_lengths = [length[i][0] for i in range(length_path)]
    naive_lengths = [length[i][1] for i in range(length_path)]
    ds_lengths = [length[i][2] for i in range(length_path)]
    single_selective_lengths = [length[i][3] for i in range(length_path)]

    selective_coverage = [coverage[i][0] for i in range(length_path)]
    naive_coverage = [coverage[i][1] for i in range(length_path)]
    ds_coverage = [coverage[i][2] for i in range(length_path)]
    single_selective_coverage = [coverage[i][3] for i in range(length_path)]

    selective_sensitivity = [sensitivity[i][0] for i in range(length_path)]
    naive_sensitivity = [sensitivity[i][1] for i in range(length_path)]
    ds_sensitivity = [sensitivity[i][2] for i in range(length_path)]
    single_task_sensitivity = [sensitivity[i][3] for i in range(length_path)]

    selective_specificity = [specificity[i][0] for i in range(length_path)]
    naive_specificity = [specificity[i][1] for i in range(length_path)]
    ds_specificity = [specificity[i][2] for i in range(length_path)]
    single_task_specificity = [specificity[i][3] for i in range(length_path)]

    selective_error = [error[i][0] for i in range(length_path)]
    naive_error = [error[i][1] for i in range(length_path)]
    ds_error = [error[i][2] for i in range(length_path)]
    single_selective_error = [error[i][3] for i in range(length_path)]

    idx_min_random_multitask = np.argmin(selective_error)
    idx_min_naive_multitask = np.argmin(naive_error)
    idx_min_data_splitting = np.argmin(ds_error)
    idx_min_k_random_lasso = np.argmin(single_selective_error)

    coverage_by_ts[j][0] = selective_coverage[idx_min_random_multitask]
    length_by_ts[j][0] = selective_lengths[idx_min_random_multitask]
    sensitivity_by_ts[j][0] = selective_sensitivity[idx_min_random_multitask]
    specificity_by_ts[j][0] = selective_specificity[idx_min_random_multitask]
    coverage_by_ts[j][1] = naive_coverage[idx_min_naive_multitask]
    length_by_ts[j][1] = naive_lengths[idx_min_naive_multitask]
    sensitivity_by_ts[j][1] = naive_sensitivity[idx_min_naive_multitask]
    specificity_by_ts[j][1] = naive_specificity[idx_min_naive_multitask]
    coverage_by_ts[j][2] = ds_coverage[idx_min_data_splitting]
    length_by_ts[j][2] = ds_lengths[idx_min_data_splitting]
    sensitivity_by_ts[j][2] = ds_sensitivity[idx_min_data_splitting]
    specificity_by_ts[j][2] = ds_specificity[idx_min_data_splitting]
    coverage_by_ts[j][3] = single_selective_coverage[idx_min_k_random_lasso]
    length_by_ts[j][3] = single_selective_lengths[idx_min_k_random_lasso]
    sensitivity_by_ts[j][3] = single_task_sensitivity[idx_min_k_random_lasso]
    specificity_by_ts[j][3] = single_task_specificity[idx_min_k_random_lasso]

length = len(task_sparsity_list)
def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

fig = plt.figure(figsize=(16,7))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
plt.sca(ax2)
first = plt.boxplot([length_by_ts[j][0] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3, sym='', widths=0.3)
second = plt.boxplot([length_by_ts[j][1] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 + .3, sym='', widths=0.3)
third = plt.boxplot([length_by_ts[j][2] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 + .6, sym='', widths=0.3)
fourth = plt.boxplot([length_by_ts[j][3] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 + 0.9, sym='',widths=0.3)
set_box_color(first, '#2b8cbe')  # colors are from http://colorbrewer2.org/
set_box_color(second, '#D7191C')
set_box_color(third, '#31a354')
set_box_color(fourth, '#feb24c')
plt.plot([], c='#2b8cbe', label='Randomized Multi-Task Lasso')
plt.plot([], c='#D7191C', label='Multi-Task Lasso')
plt.plot([], c='#31a354', label='Data Splitting')
plt.plot([], c='#feb24c', label='K Randomized Lassos')
plt.legend()
plt.xticks(xrange(1, (length) * 3 + 1, 3), [round(num, 1) for num in task_sparsity_list])
plt.xlim(-1, (length - 1) * 3 + 3)
plt.tight_layout()
plt.ylabel('Interval Length',fontsize=12)
plt.xlabel('Lambda Value')
plt.title('Interval Length Along Lambda Path')

print(coverage_by_ts)
plt.sca(ax1)
first = plt.boxplot([coverage_by_ts[j][0] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3, sym='', widths=0.3)
second = plt.boxplot([coverage_by_ts[j][1] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 + .3, sym='', widths=0.3)
third = plt.boxplot([coverage_by_ts[j][2] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 + .6, sym='', widths=0.3)
fourth = plt.boxplot([coverage_by_ts[j][3] for j in range(len(task_sparsity_list))], positions=np.array(xrange(length)) * 3 + 0.9, sym='',widths=0.3)
set_box_color(first, '#2b8cbe')  # colors are from http://colorbrewer2.org/
set_box_color(second, '#D7191C')
set_box_color(third, '#31a354')
set_box_color(fourth, '#feb24c')
plt.plot([], c='#2b8cbe', label='Randomized Multi-Task Lasso')
plt.plot([], c='#D7191C', label='Multi-Task Lasso')
plt.plot([], c='#31a354', label='Data Splitting')
plt.plot([], c='#feb24c', label='K Randomized Lassos')
plt.legend()
plt.xticks(xrange(1, (length) * 3 + 1, 3), [round(num, 1) for num in task_sparsity_list])
plt.xlim(-1, (length - 1) * 3 + 3)
plt.tight_layout()
plt.ylabel('Coverage',fontsize=12)
plt.xlabel('Lambda Value')
plt.title('Interval Length Along Lambda Path')

ax1.set_title("Coverage", y = 1.01)
ax2.set_title("Length", y = 1.01)

ax2.legend_.remove()
ax1.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.3))

def common_format(ax):
    ax.grid(True, which='both',color='#f0f0f0')
    ax.set_xlabel('Task Sparsity', fontsize=12)
    return ax

common_format(ax1)
common_format(ax2)

# add target coverage on the first plot
ax1.axhline(y=0.9, color='k', linestyle='--', linewidth=2)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('cov_len_by_ts_strong.png', bbox_inches='tight')


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