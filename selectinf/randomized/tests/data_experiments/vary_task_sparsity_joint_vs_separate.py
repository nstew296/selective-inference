import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from selectinf.randomized.tests.test_multitask_lasso import test_inference_one_vs_two_models

k = 4
p = 500
global_sparsity = 0.85

length_path = 6
lambdamin_si = 1.5
lambdamax_si = 4.0
feature_weight_list_si = np.linspace(lambdamin_si, lambdamax_si, length_path)
print(feature_weight_list_si)

sparsity_list = [0.5, 0.5]
task_sparsity_structure = ['joint', 'pairwise']
n_list = [100, 100]
# track coverage, length, and F1 score for each level of sparsity
coverage_by_ts = {j: [[], []] for j in range(len(sparsity_list))}
length_by_ts = {j: [[], []] for j in range(len(sparsity_list))}
f1_by_ts = {j: [[], []] for j in range(len(sparsity_list))}

for j in range(len(sparsity_list)):
    positive = (1. - global_sparsity) * (1. - sparsity_list[j]) * k * p
    negative = k * p - positive

    # Create lists to track length for each method by lambda at given sparsity level
    joint_lengths, separate_lengths = ([] for _ in range(2))

    # Create lists to track coverage for each method by lambda at given sparsity level
    joint_coverage, separate_coverage = ([] for _ in range(2))

    # Create lists to track sensitivity for each method by lambda at given sparsity level
    joint_sensitivity, separate_sensitivity = ([] for _ in range(2))

    # Create lists to track specificity for each method by lambda at given sparsity level
    joint_specificity, separate_specificity = ([] for _ in range(2))

    # Create lists to track validation error for each method by lambda at given sparsity level
    joint_error, separate_error = ([] for _ in range(2))

    for i in range(length_path):
        print((i, j), "(i,j)")
        weight = [feature_weight_list_si[i]]
        weight.extend([feature_weight_list_si[i]])
        sims = test_inference_one_vs_two_models(weight, [1.0, 3.0], p, task_sparsity_structure[j], global_sparsity,
                                                nsim=n_list[j])
        joint_coverage.append(sims["joint_coverage"])
        separate_coverage.append(sims["separate_coverage"])

        joint_lengths.append(sims["joint_length"])
        separate_lengths.append(sims["separate_length"])

        joint_sensitivity.append(sims["joint_sensitivity"])
        separate_sensitivity.append(sims["separate_sensitivity"])

        joint_specificity.append(sims["joint_specificity"])
        separate_specificity.append(sims["separate_specificity"])

        joint_error.append(sims["joint_error"])
        separate_error.append(sims["separate_error"])

    idx_min_joint = np.argmin(joint_error)
    print(idx_min_joint)
    idx_min_separate = np.argmin(separate_error)
    print(idx_min_separate)

    # Convert sensitivity and specificity at optimal tuning parameter to false positive and true positive rates
    # First column gives false positive rate and second column gives true positive rate
    joint_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(joint_specificity)[idx_min_joint, :][n]
                     for n in range(n_list[j])]), np.asarray(joint_sensitivity)[idx_min_joint, :]]).T

    # Use true positive and false positive counts to find f1
    # f1 = (2TP/(2TP + FP + FN) https://en.wikipedia.org/wiki/F-score
    joint_f1 = np.asarray(
        [2.0 * joint_tp_fp_mat[n, 1] * positive / (2.0 * joint_tp_fp_mat[n, 1] * positive +
                                                   joint_tp_fp_mat[n, 0] * negative +
                                                   (1.0 - joint_tp_fp_mat[n, 1]) * positive)
         for n in range(n_list[j])])

    f1_by_ts[j][0] = joint_f1

    separate_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(separate_specificity)[idx_min_separate, :][n]
                     for n in range(n_list[j])]), np.asarray(separate_sensitivity)[idx_min_separate, :]]).T
    separate_f1 = np.asarray(
        [2.0 * separate_tp_fp_mat[n, 1] * positive / (2.0 * separate_tp_fp_mat[n, 1] * positive +
                                                      separate_tp_fp_mat[n, 0] * negative +
                                                      (1.0 - separate_tp_fp_mat[n, 1]) * positive)
         for n in range(n_list[j])])

    f1_by_ts[j][1] = separate_f1

    print(np.mean(joint_f1), np.mean(separate_f1), "F1 score means")

    # Record coverage and length at optimal tuning parameter for given sparsity level
    coverage_by_ts[j][0] = joint_coverage[idx_min_joint]
    length_by_ts[j][0] = joint_lengths[idx_min_joint]

    coverage_by_ts[j][1] = separate_coverage[idx_min_separate]
    length_by_ts[j][1] = separate_lengths[idx_min_separate]

# Visualize results
length = len(sparsity_list)


def set_boxplot_style(bp, color, linestyle):
    plt.setp(bp['boxes'], color=color, linestyle=linestyle, linewidth=2)
    plt.setp(bp['whiskers'], color=color, linestyle=linestyle, linewidth=2)
    plt.setp(bp['caps'], color=color, linewidth=2)
    plt.setp(bp['medians'], color=color, linewidth=2)


fig = plt.figure(figsize=(17, 5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
plt.sca(ax1)
first = plt.boxplot([coverage_by_ts[j][0] for j in range(len(sparsity_list))],
                    positions=np.array(range(length)) * 3 + 0.75, sym='', widths=0.3)
second = plt.boxplot([coverage_by_ts[j][1] for j in range(len(sparsity_list))],
                     positions=np.array(range(length)) * 3 + 1.25, sym='', widths=0.3)
set_boxplot_style(first, '#2b8cbe', 'solid')
set_boxplot_style(second, '#2b8cbe', 'dotted')
plt.xticks(range(1, (length) * 3 + 1, 3), ['(a)', '(b)'], fontsize=17)
plt.xlim(-1, (length - 1) * 3 + 3)
plt.plot([], c='#2b8cbe', label='Joint', linewidth=2.5)
plt.plot([], c='#2b8cbe', label='Pairwise', linewidth=2.5, linestyle='dotted')
plt.legend()
plt.tight_layout()
plt.ylabel('Coverage per Simulation', fontsize=18)
plt.yticks(fontsize=14)

plt.sca(ax2)
first = plt.boxplot([length_by_ts[j][0] for j in range(len(sparsity_list))],
                    positions=np.array(range(length)) * 3 + 0.75, sym='', widths=0.3)
second = plt.boxplot([length_by_ts[j][1] for j in range(len(sparsity_list))],
                     positions=np.array(range(length)) * 3 + 1.2, sym='', widths=0.3)
set_boxplot_style(first, '#2b8cbe', 'solid')
set_boxplot_style(second, '#2b8cbe', 'dotted')
plt.xticks(range(1, (length) * 3 + 1, 3), ['(a)', '(b)'], fontsize=17)
plt.xlim(-1, (length - 1) * 3 + 3)
plt.tight_layout()
plt.ylabel('Interval Lengths', fontsize=18)
plt.yticks(fontsize=14)

plt.sca(ax3)
first = plt.boxplot([f1_by_ts[j][0] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 0.75,
                    sym='', widths=0.3)
second = plt.boxplot([f1_by_ts[j][1] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 1.25,
                     sym='', widths=0.3)
set_boxplot_style(first, '#2b8cbe', 'solid')
set_boxplot_style(second, '#2b8cbe', 'dotted')
plt.xticks(range(1, (length) * 3 + 1, 3), ['(a)', '(b)'], fontsize=17)
plt.xlim(-1, (length - 1) * 3 + 3)
plt.tight_layout()
plt.ylabel('F1 per Simulation', fontsize=18)
plt.yticks(fontsize=14)


# ax1.set_title("Coverage", y = 1.01,fontsize=20)
# ax2.set_title("Length", y = 1.01,fontsize=20)
# ax3.set_title("F1 Score", y = 1.01,fontsize=20)


def common_format(ax):
    ax.grid(True, which='both', color='#f0f0f0')
    ax.set_xlabel('Design of Shared Structure', fontsize=15)
    return ax


common_format(ax1)
common_format(ax2)
common_format(ax3)

# add target coverage on the first plot
ax1.axhline(y=0.9, color='k', linestyle='--', linewidth=2)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.2)
plt.rcParams['legend.title_fontsize'] = 18
ax1.legend(loc='lower left', bbox_to_anchor=(1.195, -0.44), ncol=2, fontsize=22, title='MTL(0.7) + SI:', borderpad=0.5)
plt.savefig('vary_task_sparsity_joint_vs_separate.png', bbox_inches='tight')
