import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from selectinf.randomized.tests.test_multitask_lasso import test_inference_mixed_ts

############# new imports ##############
from constants import *
from compile_results import *

extract_results_tarball(tar_filename, tar_extract_folder)
########################################

k = 4
p = 500
global_sparsity = 0.85

length_path = 6
lambdamin_ds = 1.0
lambdamax_ds = 3.5
feature_weight_list_ds = np.linspace(lambdamin_ds, lambdamax_ds, length_path)

lambdamin_si = 1.5
lambdamax_si = 4.0
feature_weight_list_si = np.linspace(lambdamin_si, lambdamax_si, length_path)
print(feature_weight_list_si)

# sparsity_list = [0, 0.25, 0.5, 0.75]
sparsity_list = np.array([[0.25, 0.25], [0.25, 0.5], [0.5, 0.5]])
n_list = [100, 100, 100]
# track coverage, length, and F1 score for each level of sparsity
coverage_by_ts = {j: [[], [], [], []] for j in range(len(sparsity_list))}
length_by_ts = {j: [[], [], [], []] for j in range(len(sparsity_list))}
f1_by_ts = {j: [[], [], [], []] for j in range(len(sparsity_list))}

for j in range(len(sparsity_list)):
    positive = (1. - global_sparsity) * (1. - sparsity_list[j, 1]) * k * int(np.floor(p / 2)) + \
               (1. - global_sparsity) * (1. - sparsity_list[j, 0]) * k * int(p - np.floor(p / 2))
    negative = k * p - positive

    # Create lists to track length for each method by lambda at given sparsity level
    selective_lengths, ds_lengths, ds_lengths2, single_selective_lengths = ([] for _ in range(4))

    # Create lists to track coverage for each method by lambda at given sparsity level
    selective_coverage, ds_coverage, ds_coverage2, single_selective_coverage = ([] for _ in range(4))

    # Create lists to track sensitivity for each method by lambda at given sparsity level
    selective_sensitivity, ds_sensitivity, ds_sensitivity2, single_task_sensitivity = ([] for _ in range(4))

    # Create lists to track specificity for each method by lambda at given sparsity level
    selective_specificity, ds_specificity, ds_specificity2, single_task_specificity = ([] for _ in range(4))

    # Create lists to track validation error for each method by lambda at given sparsity level
    selective_error, ds_error, ds_error2, single_selective_error = ([] for _ in range(4))

    for i in range(length_path):
        print((i, j), "(i,j)")
        weight = [feature_weight_list_si[i]]
        weight.extend([feature_weight_list_ds[i]] * 2)
        weight.extend([feature_weight_list_si[i]])
        sims = test_inference_mixed_ts(weight, [1.0, 3.0], p, sparsity_list[j, :], global_sparsity, nsim=n_list[j])
        # sims = test_inference_from_saved_results(j, i)
        selective_coverage.append(sims["MTL_SI_07_coverage"])
        ds_coverage.append(sims["DS_67_coverage"])
        ds_coverage2.append(sims["DS_50_coverage"])
        single_selective_coverage.append(sims["LASSO_SI_07_coverage"])

        selective_lengths.append(sims["MTL_SI_07_length"])
        ds_lengths.append(sims["DS_67_length"])
        ds_lengths2.append(sims["DS_50_length"])
        single_selective_lengths.append(sims["LASSO_SI_07_length"])

        selective_sensitivity.append(sims["MTL_SI_07_sensitivity"])
        ds_sensitivity.append(sims["DS_67_sensitivity"])
        ds_sensitivity2.append(sims["DS_50_sensitivity"])
        single_task_sensitivity.append(sims["LASSO_SI_07_sensitivity"])

        selective_specificity.append(sims["MTL_SI_07_specificity"])
        ds_specificity.append(sims["DS_67_specificity"])
        ds_specificity2.append(sims["DS_50_specificity"])
        single_task_specificity.append(sims["LASSO_SI_07_specificity"])

        selective_error.append(sims["MTL_SI_07_error"])
        ds_error.append(sims["DS_67_error"])
        ds_error2.append(sims["DS_50_error"])
        single_selective_error.append(sims["LASSO_SI_07_error"])

    idx_min_random_multitask = np.argmin(selective_error)
    print(idx_min_random_multitask)
    idx_min_data_splitting = np.argmin(ds_error)
    print(idx_min_data_splitting)
    idx_min_data_splitting2 = np.argmin(ds_error2)
    print(idx_min_data_splitting2)
    idx_min_k_random_lasso = np.argmin(single_selective_error)
    print(idx_min_k_random_lasso)

    # Convert sensitivity and specificity at optimal tuning parameter to true positive and false positive rates
    # First column gives false positive rate and second column gives true positive rate
    selective_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(selective_specificity)[idx_min_random_multitask, :][n]
                     for n in range(n_list[j])]), np.asarray(selective_sensitivity)[idx_min_random_multitask, :]]).T

    # Use true positive and false positive counts to find f1
    # f1 = (2TP/(2TP + FP + FN) https://en.wikipedia.org/wiki/F-score
    selective_f1 = np.asarray(
        [2.0 * selective_tp_fp_mat[n, 1] * positive / (2.0 * selective_tp_fp_mat[n, 1] * positive +
                                                       selective_tp_fp_mat[n, 0] * negative +
                                                       (1.0 - selective_tp_fp_mat[n, 1]) * positive)
         for n in range(n_list[j])])

    f1_by_ts[j][0] = selective_f1

    ds_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(ds_specificity)[idx_min_data_splitting, :][n]
                     for n in range(n_list[j])]), np.asarray(ds_sensitivity)[idx_min_data_splitting, :]]).T
    ds_f1 = np.asarray(
        [2.0 * ds_tp_fp_mat[n, 1] * positive / (2.0 * ds_tp_fp_mat[n, 1] * positive +
                                                ds_tp_fp_mat[n, 0] * negative +
                                                (1.0 - ds_tp_fp_mat[n, 1]) * positive) for n in range(n_list[j])])

    f1_by_ts[j][1] = ds_f1

    ds2_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(ds_specificity2)[idx_min_data_splitting2, :][n]
                     for n in range(n_list[j])]), np.asarray(ds_sensitivity2)[idx_min_data_splitting2, :]]).T
    ds2_f1 = np.asarray(
        [2.0 * ds2_tp_fp_mat[n, 1] * positive / (2.0 * ds2_tp_fp_mat[n, 1] * positive +
                                                 ds2_tp_fp_mat[n, 0] * negative +
                                                 (1.0 - ds2_tp_fp_mat[n, 1]) * positive) for n in range(n_list[j])])

    f1_by_ts[j][2] = ds2_f1

    single_task_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(single_task_specificity)[idx_min_k_random_lasso, :][n]
                     for n in range(n_list[j])]), np.asarray(single_task_sensitivity)[idx_min_k_random_lasso, :]]).T
    single_task_f1 = np.asarray(
        [2.0 * single_task_tp_fp_mat[n, 1] * positive / (2.0 * single_task_tp_fp_mat[n, 1] * positive +
                                                         single_task_tp_fp_mat[n, 0] * negative +
                                                         (1.0 - single_task_tp_fp_mat[n, 1]) * positive)
         for n in range(n_list[j])])

    f1_by_ts[j][3] = single_task_f1

    print(np.mean(selective_f1), np.mean(ds_f1), np.mean(ds2_f1), np.mean(single_task_f1), "F1 score means")

    # Record coverage and length at optimal tuning parameter for given sparsity level
    coverage_by_ts[j][0] = selective_coverage[idx_min_random_multitask]
    length_by_ts[j][0] = selective_lengths[idx_min_random_multitask]

    coverage_by_ts[j][1] = ds_coverage[idx_min_data_splitting]
    length_by_ts[j][1] = ds_lengths[idx_min_data_splitting]

    coverage_by_ts[j][2] = ds_coverage2[idx_min_data_splitting2]
    length_by_ts[j][2] = ds_lengths2[idx_min_data_splitting2]

    coverage_by_ts[j][3] = single_selective_coverage[idx_min_k_random_lasso]
    length_by_ts[j][3] = single_selective_lengths[idx_min_k_random_lasso]

# Visualize results
length = len(sparsity_list)


def set_boxplot_style(bp, color, linestyle):
    plt.setp(bp['boxes'], color=color, linestyle=linestyle, linewidth=2)
    plt.setp(bp['whiskers'], color=color, linestyle=linestyle, linewidth=2)
    plt.setp(bp['caps'], color=color, linestyle=linestyle, linewidth=2)
    plt.setp(bp['medians'], color=color, linestyle=linestyle, linewidth=2)


fig = plt.figure(figsize=(17, 5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
plt.sca(ax1)
first = plt.boxplot([coverage_by_ts[j][0] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3,
                    sym='', widths=0.3)
third = plt.boxplot([coverage_by_ts[j][1] for j in range(len(sparsity_list))],
                    positions=np.array(range(length)) * 3 + 0.5, sym='', widths=0.3)
fourth = plt.boxplot([coverage_by_ts[j][2] for j in range(len(sparsity_list))],
                     positions=np.array(range(length)) * 3 + 1, sym='', widths=0.3)
fifth = plt.boxplot([coverage_by_ts[j][3] for j in range(len(sparsity_list))],
                    positions=np.array(range(length)) * 3 + 1.5, sym='', widths=0.3)
set_boxplot_style(first, '#2b8cbe', 'solid')
set_boxplot_style(third, '#238443', 'dotted')
set_boxplot_style(fourth, '#31a354', 'dashed')
set_boxplot_style(fifth, '#fd8d3c', 'dashdot')
plt.xticks(range(1, (length) * 3 + 1, 3), [num for num in np.mean(sparsity_list, axis=1)], fontsize=14)
plt.xlim(-1, (length - 1) * 3 + 3)
plt.plot([], c='#2b8cbe', label='MTL (0.7) + SI', linewidth=2.5)
plt.plot([], c='#238443', label='DS (0.67)', linestyle='dotted', linewidth=2.5)
plt.plot([], c='#31a354', label='DS (0.5)', linestyle='dashed', linewidth=2.5)
plt.plot([], c='#fd8d3c', label='LASSO (0.7) + SI', linestyle='dashdot', linewidth=2.5)
plt.legend()
plt.tight_layout()
plt.ylabel('Coverage per Simulation', fontsize=18)
plt.yticks(fontsize=14)

plt.sca(ax2)
first = plt.boxplot([length_by_ts[j][0] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3,
                    sym='', widths=0.3)
third = plt.boxplot([length_by_ts[j][1] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + .5,
                    sym='', widths=0.3)
fourth = plt.boxplot([length_by_ts[j][2] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 1,
                     sym='', widths=0.3)
fifth = plt.boxplot([length_by_ts[j][3] for j in range(len(sparsity_list))],
                    positions=np.array(range(length)) * 3 + 1.5, sym='', widths=0.3)
set_boxplot_style(first, '#2b8cbe', 'solid')
set_boxplot_style(third, '#238443', 'dotted')
set_boxplot_style(fourth, '#31a354', 'dashed')
set_boxplot_style(fifth, '#fd8d3c', 'dashdot')
plt.xticks(range(1, (length) * 3 + 1, 3), [num for num in np.mean(sparsity_list, axis=1)], fontsize=14)
plt.xlim(-1, (length - 1) * 3 + 3)
plt.tight_layout()
plt.ylabel('Interval Lengths', fontsize=18)
plt.yticks(fontsize=14)

plt.sca(ax3)
first = plt.boxplot([f1_by_ts[j][0] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3, sym='',
                    widths=0.3)
third = plt.boxplot([f1_by_ts[j][1] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + .5,
                    sym='', widths=0.3)
fourth = plt.boxplot([f1_by_ts[j][2] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 1,
                     sym='', widths=0.3)
fifth = plt.boxplot([f1_by_ts[j][3] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 1.5,
                    sym='', widths=0.3)
set_boxplot_style(first, '#2b8cbe', 'solid')
set_boxplot_style(third, '#238443', 'dotted')
set_boxplot_style(fourth, '#31a354', 'dashed')
set_boxplot_style(fifth, '#fd8d3c', 'dashdot')
plt.xticks(range(1, (length) * 3 + 1, 3), [num for num in np.mean(sparsity_list, axis=1)], fontsize=14)
plt.xlim(-1, (length - 1) * 3 + 3)
plt.tight_layout()
plt.ylabel('F1 per Simulation', fontsize=18)
plt.yticks(fontsize=14)


# ax1.set_title("Coverage", y = 1.01,fontsize=20)
# ax2.set_title("Length", y = 1.01,fontsize=20)
# ax3.set_title("F1 Score", y = 1.01,fontsize=20)


def common_format(ax):
    ax.grid(True, which='both', color='#f0f0f0')
    ax.set_xlabel('Task Sparsity', fontsize=18)
    return ax


common_format(ax1)
common_format(ax2)
common_format(ax3)

# add target coverage on the first plot
ax1.axhline(y=0.9, color='k', linestyle='--', linewidth=2)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.2)
ax1.legend(loc='lower left', bbox_to_anchor=(0.32, -0.35), fontsize=20, ncol=4)
plt.savefig('vary_task_sparsity_p100.png', bbox_inches='tight')

# Plot sensitivity, specificity
# fig = plt.figure(figsize=(25, 10))
# fig.tight_layout()
# fig.add_subplot(1, 2, 1)
# plt.plot(task_sparsity_list, random_multitask_sensitivity, c='#D7191C')
# plt.plot(task_sparsity_list, naive_multitask_sensitivity, c='#2b8cbe')
# plt.plot(task_sparsity_list, data_splitting_sensitivity, c='#31a354')
# plt.plot(task_sparsity_list, k_random_lasso_sensitivity, c='#c51b8a')
# plt.plot([], c='#D7191C', label='Randomized Multi-Task Lasso')
# plt.plot([], c='#2b8cbe', label='Multi-Task Lasso')
# plt.plot([], c='#31a354', label='Data Splitting')
# plt.plot([], c='#c51b8a', label='K Randomized Lassos')
# plt.plot([], c='#feb24c', label='One Randomized Lasso')
# plt.legend()
# plt.tight_layout()
# plt.ylabel('Average Sensitivity')
# plt.xlabel('Task Sparsity')
# plt.title('Sensitivity by Task Sparsity')
# fig.add_subplot(1, 2, 2)
# plt.plot(task_sparsity_list, random_multitask_specificity, c='#D7191C')
# plt.plot(task_sparsity_list, naive_multitask_specificity, c='#2b8cbe')
# plt.plot(task_sparsity_list, data_splitting_specificity, c='#31a354')
# plt.plot(task_sparsity_list, k_random_lasso_specificity, c='#c51b8a')
# plt.plot([], c='#D7191C', label='Randomized Multi-Task Lasso')
# plt.plot([], c='#2b8cbe', label='Multi-Task Lasso')
# plt.plot([], c='#31a354', label='Data Splitting')
# plt.plot([], c='#c51b8a', label='K Randomized Lassos')
# plt.plot([], c='#feb24c', label='One Randomized Lasso')
# plt.legend()
# plt.tight_layout()
# plt.ylabel('Average Specificity')
# plt.xlabel('Task Sparsity')
# plt.title('Specificity by Task Sparsity')
# plt.savefig('model_selection_compare_weak.png')
