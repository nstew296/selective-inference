import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from selectinf.randomized.tests.test_multitask_lasso_2 import test_coverage

length_path = 10

lambdamin = 1.5
lambdamax = 3.75
#weights = np.arange(np.log(lambdamin), np.log(lambdamax), (np.log(lambdamax) - np.log(lambdamin)) / (length_path))
#feature_weight_list = np.exp(weights)
feature_weight_list = np.arange(lambdamin, lambdamax,(lambdamax - lambdamin) / (length_path))
print(feature_weight_list)

df = pd.DataFrame(columns=['Task Sparsity', 'Method', 'Coverage', 'Length'])

random_multitask_coverage = []
naive_multitask_coverage = []
data_splitting_coverage = []
k_random_lasso_coverage = []

random_multitask_length = []
naive_multitask_length = []
data_splitting_length = []
k_random_lasso_length = []

random_multitask_sensitivity = []
naive_multitask_sensitivity = []
data_splitting_sensitivity = []
k_random_lasso_sensitivity = []

random_multitask_specificity = []
naive_multitask_specificity = []
data_splitting_specificity = []
k_random_lasso_specificity = []

task_sparsity_list = [0.0, 0.2, 0.4, 0.6, 0.8]
for task_sparsity in task_sparsity_list:
    coverage = {i: [[], [], [], []] for i in range(length_path)}
    length = {i: [[], [], [], []] for i in range(length_path)}
    sensitivity = {i: [[], [], [], []] for i in range(length_path)}
    specificity = {i: [[], [], [], []] for i in range(length_path)}
    error = {i: [[], [], [], []] for i in range(length_path)}

    for i in range(len(feature_weight_list)):

        sims = test_coverage(feature_weight_list[i],[1.0,3.0],ts=task_sparsity,nsim=2)
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

    random_multitask_coverage.extend(selective_coverage[idx_min_random_multitask])
    random_multitask_length.extend(selective_lengths[idx_min_random_multitask])
    random_multitask_sensitivity.extend(selective_sensitivity[idx_min_random_multitask])
    random_multitask_specificity.extend(selective_specificity[idx_min_random_multitask])
    naive_multitask_coverage.extend(naive_coverage[idx_min_naive_multitask])
    naive_multitask_length.extend(naive_lengths[idx_min_naive_multitask])
    naive_multitask_sensitivity.extend(naive_sensitivity[idx_min_naive_multitask])
    naive_multitask_specificity.extend(naive_specificity[idx_min_naive_multitask])
    data_splitting_coverage.extend(ds_coverage[idx_min_data_splitting])
    data_splitting_length.extend(ds_lengths[idx_min_data_splitting])
    data_splitting_sensitivity.extend(ds_sensitivity[idx_min_data_splitting])
    data_splitting_specificity.extend(ds_specificity[idx_min_data_splitting])
    k_random_lasso_coverage.extend(single_selective_coverage[idx_min_k_random_lasso])
    k_random_lasso_length.extend(single_selective_lengths[idx_min_k_random_lasso])
    k_random_lasso_sensitivity.extend(single_task_sensitivity[idx_min_k_random_lasso])
    k_random_lasso_specificity.extend(single_task_specificity[idx_min_k_random_lasso])
    if random_multitask_coverage!=[]:
        print("here")
        df = df.append(pd.DataFrame(np.array([[task_sparsity,'random_multitask',random_multitask_coverage[i],random_multitask_length[i]] for i in range(len(random_multitask_coverage))]),
                   columns=['Task Sparsity', 'Method', 'Coverage', 'Length']))
    if naive_multitask_coverage !=[]:
        df = df.append(pd.DataFrame(np.array(
        [[task_sparsity, 'naive_multitask', naive_multitask_coverage[i], naive_multitask_length[i]] for i in
         range(len(naive_multitask_coverage))]), columns=['Task Sparsity', 'Method', 'Coverage', 'Length']))
    if data_splitting_coverage!=[]:
        df = df.append(pd.DataFrame(np.array(
        [[task_sparsity, 'data_splitting', data_splitting_coverage[i], data_splitting_length[i]] for i in
         range(len(data_splitting_coverage))]), columns=['Task Sparsity', 'Method', 'Coverage', 'Length']))
    if k_random_lasso_coverage!=[]:
        df = df.append(pd.DataFrame(np.array(
        [[task_sparsity, 'k_random_lasso', k_random_lasso_coverage[i], k_random_lasso_length[i]] for i in
         range(len(k_random_lasso_coverage))]), columns=['Task Sparsity', 'Method', 'Coverage', 'Length']))

print(df)

cols = ['#2b8cbe', '#D7191C', '#31a354', '#feb24c']
order = ['random_multitask','naive_multitask','data_splitting','k_random_lasso']

sns.set(font_scale=2) # fond size
sns.set_style("white", {'axes.facecolor': 'white',
                        'axes.grid': True,
                        'axes.linewidth': 2.0,
                        'grid.linestyle': u'--',
                        'grid.linewidth': 4.0,
                        'xtick.major.size': 5.0,
                       })

fig = plt.figure(figsize=(11,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

sns.pointplot(x="Task Sparsity", y="Coverage", hue_order=order, markers='o', hue="Method", data=df, ax=ax1, palette=cols)
sns.pointplot(x="Task Sparsity", y="Length",   hue_order=order,  markers='o', hue="Method", data=df, ax=ax2, palette=cols)

ax1.set_title("Coverage", y = 1.01)
ax2.set_title("Length", y = 1.01)

ax1.legend_.remove()
ax2.legend_.remove()

ax1.set_ylim(0,1)

def common_format(ax):
    ax.grid(True, which='both')
    ax.set_xlabel('', fontsize=22)
    # ax.yaxis.label.set_size(22)
    ax.set_ylabel('', fontsize=22)
    return ax

common_format(ax1)
common_format(ax2)
fig.text(0.5, -0.04, 'Task Sparsity (Percentage)', fontsize=22, ha='center')

# add target coverage on the first plot
ax1.axhline(y=0.9, color='k', linestyle='--', linewidth=2)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('cov_len_by_ts.png', bbox_inches='tight')


fig = plt.figure(figsize=(25, 10))
fig.tight_layout()
fig.add_subplot(1, 2, 1)
plt.plot(task_sparsity_list, random_multitask_sensitivity, c='#D7191C')
plt.plot(task_sparsity_list, naive_multitask_sensitivity, c='#2b8cbe')
plt.plot(task_sparsity_list, data_splitting_sensitivity, c='#31a354')
plt.plot(task_sparsity_list, k_random_lasso_sensitivity, c='#c51b8a')
plt.plot([], c='#D7191C', label='Randomized Multi-Task Lasso')
plt.plot([], c='#2b8cbe', label='Multi-Task Lasso')
plt.plot([], c='#31a354', label='Data Splitting')
plt.plot([], c='#c51b8a', label='K Randomized Lassos')
plt.plot([], c='#feb24c', label='One Randomized Lasso')
plt.legend()
plt.tight_layout()
plt.ylabel('Average Sensitivity')
plt.xlabel('Task Sparsity')
plt.title('Sensitivity by Task Sparsity')
fig.add_subplot(1, 2, 2)
plt.plot(task_sparsity_list, random_multitask_specificity, c='#D7191C')
plt.plot(task_sparsity_list, naive_multitask_specificity, c='#2b8cbe')
plt.plot(task_sparsity_list, data_splitting_specificity, c='#31a354')
plt.plot(task_sparsity_list, k_random_lasso_specificity, c='#c51b8a')
plt.plot([], c='#D7191C', label='Randomized Multi-Task Lasso')
plt.plot([], c='#2b8cbe', label='Multi-Task Lasso')
plt.plot([], c='#31a354', label='Data Splitting')
plt.plot([], c='#c51b8a', label='K Randomized Lassos')
plt.plot([], c='#feb24c', label='One Randomized Lasso')
plt.legend()
plt.tight_layout()
plt.ylabel('Average Specificity')
plt.xlabel('Task Sparsity')
plt.title('Specificity by Task Sparsity')
plt.savefig('model_selection_compare_weak.png')