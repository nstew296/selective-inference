import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from selectinf.randomized.tests.test_multitask_lasso import test_inference_error_comparison

############# new imports ##############
from constants import *
from compile_results import *

extract_results_tarball(tar_filename, tar_extract_folder)
########################################

k = 4
p = 500
global_sparsity = 0.85

length_path = 6
lambdamin_si = 100.5
lambdamax_si = 4.0
feature_weight_list_si = np.linspace(lambdamin_si, lambdamax_si, length_path)
print(feature_weight_list_si)

sparsity_list = [0, 0.25, 0.5]
n_list = [100, 100, 100]
# track coverage, length, and F1 score for each level of sparsity
coverage_by_ts = {j: [[], [], [], []] for j in range(len(sparsity_list))}

for j in range(len(sparsity_list)):
    # Create lists to track coverage for each method by lambda at given sparsity level
    gaussian_coverage, exponential_coverage, laplace_coverage = ([] for _ in range(3))

    # Create lists to track validation error for each method by lambda at given sparsity level
    gaussian_error, exponential_error, laplace_error = ([] for _ in range(3))

    for i in range(length_path):
        print((i, j), "(i,j)")
        weight = [feature_weight_list_si[i]]
        sims = test_inference_error_comparison(weight, [1.0, 3.0], p, sparsity_list[j], global_sparsity, nsim=n_list[j])
        gaussian_coverage.append(sims["Gaussian_coverage"])
        exponential_coverage.append(sims["Exponential_coverage"])
        laplace_coverage.append(sims["Laplace_coverage"])

        gaussian_error.append(sims["Gaussian_error"])
        exponential_error.append(sims["Exponential_error"])
        laplace_error.append(sims["Laplace_error"])

    idx_min_gaussian = np.argmin(gaussian_error)
    print(idx_min_gaussian)
    idx_min_exponential = np.argmin(exponential_error)
    print(idx_min_exponential)
    idx_min_laplace = np.argmin(laplace_error)
    print(idx_min_laplace)

    # Record coverage and length at optimal tuning parameter for given sparsity level
    coverage_by_ts[j][0] = gaussian_coverage[idx_min_gaussian]
    coverage_by_ts[j][1] = exponential_coverage[idx_min_exponential]
    coverage_by_ts[j][2] = laplace_coverage[idx_min_laplace]

# Visualize results
length = len(sparsity_list)


def set_boxplot_style(bp, color, linestyle):
    plt.setp(bp['boxes'], color=color, linestyle=linestyle, linewidth=2)
    plt.setp(bp['whiskers'], color=color, linestyle=linestyle, linewidth=2)
    plt.setp(bp['caps'], color=color, linewidth=2)
    plt.setp(bp['medians'], color=color, linewidth=2)


fig = plt.figure(figsize=(10, 7))
ax1 = fig.add_subplot(111)
first = plt.boxplot([coverage_by_ts[j][0] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3,
                    sym='', widths=0.3)
second = plt.boxplot([coverage_by_ts[j][1] for j in range(len(sparsity_list))],
                     positions=np.array(range(length)) * 3 + 0.5, sym='', widths=0.3)
third = plt.boxplot([coverage_by_ts[j][2] for j in range(len(sparsity_list))],
                    positions=np.array(range(length)) * 3 + 1, sym='', widths=0.3)
set_boxplot_style(first, '#2b8cbe', 'solid')
set_boxplot_style(second, '#2b8cbe', 'dashed')
set_boxplot_style(third, '#2b8cbe', 'dotted')
plt.xticks(np.arange(0.5, (length) * 3 + 0.5, 3), [round(num, 2) for num in sparsity_list], fontsize=14)
plt.xlim(-1, (length - 1) * 3 + 3)
plt.plot([], c='#2b8cbe', label='Gaussian Errors', linewidth=2.5)
plt.plot([], c='#2b8cbe', label='Exponential Errors', linestyle='dashed', linewidth=2.5)
plt.plot([], c='#2b8cbe', label='Laplacian Errors', linestyle='dotted', linewidth=2.5)
plt.legend(loc='lower left')
plt.tight_layout()
plt.ylabel('Coverage per Simulation', fontsize=18)
plt.yticks(fontsize=14)


def common_format(ax):
    ax.grid(True, which='both', color='#f0f0f0')
    ax.set_xlabel('Task Sparsity', fontsize=18)
    return ax


common_format(ax1)

# add target coverage on the first plot
ax1.axhline(y=0.9, color='k', linestyle='--', linewidth=2)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.2)
ax1.legend(loc='lower left', fontsize=14, ncol=1)
plt.savefig('vary_task_sparsity_error_comparison.png', bbox_inches='tight')
