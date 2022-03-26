import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from selectinf.randomized.tests.test_multitask_lasso import test_inference

k=5
#p=250
p = 100
#task_sparsity = 0.4
task_sparsity = 0.2

length_path = 10
lambdamin_ds = 0.5
lambdamax_ds = 4.5
feature_weight_list_ds = np.linspace(lambdamin_ds, lambdamax_ds,length_path)

lambdamin_si = 1.5
lambdamax_si = 5.5
feature_weight_list_si = np.linspace(lambdamin_si, lambdamax_si,length_path)

sparsity_list = [0.80,0.85,0.9,0.95]
#sparsity_list = [0.70,0.80,0.9,0.99]
n_list = [100,100,100,100]
##n_list = [5,5,5,20,20]
coverage_by_gs = {j: [[], [], [], [], [], []] for j in range(len(sparsity_list))}
length_by_gs = {j: [[], [], [], [], [], []] for j in range(len(sparsity_list))}
f1_by_gs = {j: [[], [], [], [], [], []] for j in range(len(sparsity_list))}


for j in range(len(sparsity_list)):
    positive = (1.-task_sparsity)*(1.-sparsity_list[j])*k*p
    negative = k*p - positive

    # Create empty lists to track length for each method by lambda at given sparsity level
    selective_lengths, selective_lengths2, ds_lengths, ds_lengths2, \
    single_selective_lengths, single_selective_lengths2 = ([] for _ in range(6))

    # Create empty lists to track coverage for each method by lambda at given sparsity level
    selective_coverage, selective_coverage2, ds_coverage, ds_coverage2, \
    single_selective_coverage, single_selective_coverage2 = ([] for _ in range(6))

    # Create empty lists to track sensitivity for each method by lambda at given sparsity level
    selective_sensitivity, selective_sensitivity2, ds_sensitivity, ds_sensitivity2, \
    single_task_sensitivity, single_task_sensitivity2 = ([] for _ in range(6))

    # Create empty lists to track specificity for each method by lambda at given sparsity level
    selective_specificity, selective_specificity2, ds_specificity, ds_specificity2, \
    single_task_specificity, single_task_specificity2 = ([] for _ in range(6))

    # Create empty lists to track validation error for each method by lambda at given sparsity level
    selective_error, selective_error2, ds_error, ds_error2, single_selective_error, \
    single_selective_error2 = ([] for _ in range(6))

    for i in range(length_path):
        print((i,j),"(i,j)")
        weight = [feature_weight_list_si[i]] * 2
        weight.extend([feature_weight_list_ds[i]] * 2)
        weight.extend([feature_weight_list_si[i]] * 2)
        sims = test_inference(weight,[1.0,3.0],p,task_sparsity,sparsity_list[j],nsim=n_list[j])
        selective_coverage.append(sims["MTL_SI_07_coverage"])
        selective_coverage2.append(sims["MTL_SI_1_coverage"])
        ds_coverage.append(sims["DS_67_coverage"])
        ds_coverage2.append(sims["DS_50_coverage"])
        single_selective_coverage.append(sims["LASSO_SI_07_coverage"])
        single_selective_coverage2.append(sims["LASSO_SI_1_coverage"])

        selective_lengths.append(sims["MTL_SI_07_length"])
        selective_lengths2.append(sims["MTL_SI_1_length"])
        ds_lengths.append(sims["DS_67_length"])
        ds_lengths2.append(sims["DS_50_length"])
        single_selective_lengths.append(sims["LASSO_SI_07_length"])
        single_selective_lengths2.append(sims["LASSO_SI_1_length"])

        selective_sensitivity.append(sims["MTL_SI_07_sensitivity"])
        selective_sensitivity2.append(sims["MTL_SI_1_sensitivity"])
        ds_sensitivity.append(sims["DS_67_sensitivity"])
        ds_sensitivity2.append(sims["DS_50_sensitivity"])
        single_task_sensitivity.append(sims["LASSO_SI_07_sensitivity"])
        single_task_sensitivity2.append(sims["LASSO_SI_1_sensitivity"])

        selective_specificity.append(sims["MTL_SI_07_specificity"])
        selective_specificity2.append(sims["MTL_SI_1_specificity"])
        ds_specificity.append(sims["DS_67_specificity"])
        ds_specificity2.append(sims["DS_50_specificity"])
        single_task_specificity.append(sims["LASSO_SI_07_specificity"])
        single_task_specificity2.append(sims["LASSO_SI_1_specificity"])

        selective_error.append(sims["MTL_SI_07_error"])
        selective_error2.append(sims["MTL_SI_1_error"])
        ds_error.append(sims["DS_67_error"])
        ds_error2.append(sims["DS_50_error"])
        single_selective_error.append(sims["LASSO_SI_07_error"])
        single_selective_error2.append(sims["LASSO_SI_1_error"])

    idx_min_random_multitask = np.argmin(selective_error)
    idx_min_random_multitask2 = np.argmin(selective_error2)
    idx_min_data_splitting = np.argmin(ds_error)
    idx_min_data_splitting2 = np.argmin(ds_error2)
    idx_min_k_random_lasso = np.argmin(single_selective_error)
    idx_min_k_random_lasso2 = np.argmin(single_selective_error2)

    #Convert sensitivity and specificity to true positive and false positive rates
    #First column gives false positive rate and second column gives true positive rate
    selective_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(selective_specificity)[idx_min_random_multitask, :][n]
                     for n in range(n_list[j])]), np.asarray(selective_sensitivity)[idx_min_random_multitask, :]]).T

    #Use true positive and false positive counts to find f1
    # f1 = (2TP/(2TP + FP + FN) https://en.wikipedia.org/wiki/F-score
    selective_f1 = np.asarray(
        [2.0 * selective_tp_fp_mat[n, 1] * positive / (2.0 * selective_tp_fp_mat[n, 1] * positive +
                                                       selective_tp_fp_mat[n, 0] * negative + (1.0 -
                                                       selective_tp_fp_mat[n, 1]) * positive) for n in range(n_list[j])])

    f1_by_gs[j][0] = selective_f1

    selective2_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(selective_specificity2)[idx_min_random_multitask2, :][n]
                     for n in range(n_list[j])]), np.asarray(selective_sensitivity2)[idx_min_random_multitask, :]]).T
    selective2_f1 = np.asarray(
        [2.0 * selective2_tp_fp_mat[n, 1] * positive / (2.0 * selective2_tp_fp_mat[n, 1] * positive +
                                                       selective2_tp_fp_mat[n, 0] * negative + (1.0 -
                                                       selective2_tp_fp_mat[n, 1]) * positive) for n in range(n_list[j])])

    f1_by_gs[j][1] = selective2_f1

    ds_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(ds_specificity)[idx_min_data_splitting, :][n]
                     for n in range(n_list[j])]), np.asarray(ds_sensitivity)[idx_min_data_splitting, :]]).T
    ds_f1 = np.asarray(
        [2.0 * ds_tp_fp_mat[n, 1] * positive / (2.0 * ds_tp_fp_mat[n, 1] * positive +
                                                       ds_tp_fp_mat[n, 0] * negative + (1.0 -
                                                        ds_tp_fp_mat[ n, 1]) * positive) for n in range(n_list[j])])

    f1_by_gs[j][2] = ds_f1

    ds2_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(ds_specificity2)[idx_min_data_splitting2, :][n]
                     for n in range(n_list[j])]), np.asarray(ds_sensitivity2)[idx_min_data_splitting2, :]]).T
    ds2_f1 = np.asarray(
        [2.0 * ds2_tp_fp_mat[n, 1] * positive / (2.0 * ds2_tp_fp_mat[n, 1] * positive +
                                                ds2_tp_fp_mat[n, 0] * negative + (1.0 -
                                                                                 ds2_tp_fp_mat[n, 1]) * positive) for n
         in range(n_list[j])])

    f1_by_gs[j][3] = ds2_f1

    single_task_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(single_task_specificity)[idx_min_k_random_lasso, :][n]
                     for n in range(n_list[j])]), np.asarray(single_task_sensitivity)[idx_min_k_random_lasso, :]]).T
    single_task_f1 = np.asarray(
        [2.0 * single_task_tp_fp_mat[n, 1] * positive / (2.0 * single_task_tp_fp_mat[n, 1] * positive +
                                                single_task_tp_fp_mat[n, 0] * negative + (1.0 -
                                                                                 single_task_tp_fp_mat[n, 1]) * positive) for n
         in range(n_list[j])])

    f1_by_gs[j][4] = single_task_f1

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

    f1_by_gs[j][5] = single_task2_f1

    print(np.mean(selective_f1),np.mean(selective2_f1),np.mean(ds_f1),np.mean(ds2_f1),np.mean(single_task_f1),np.mean(single_task2_f1),"F1 score means")


    coverage_by_gs[j][0] = selective_coverage[idx_min_random_multitask]
    length_by_gs[j][0] = selective_lengths[idx_min_random_multitask]

    coverage_by_gs[j][1] = selective_coverage2[idx_min_random_multitask2]
    length_by_gs[j][1] = selective_lengths2[idx_min_random_multitask2]

    coverage_by_gs[j][2] = ds_coverage[idx_min_data_splitting]
    length_by_gs[j][2] = ds_lengths[idx_min_data_splitting]

    coverage_by_gs[j][3] = ds_coverage2[idx_min_data_splitting2]
    length_by_gs[j][3] = ds_lengths2[idx_min_data_splitting2]

    coverage_by_gs[j][4] = single_selective_coverage[idx_min_k_random_lasso]
    length_by_gs[j][4] = single_selective_lengths[idx_min_k_random_lasso]

    coverage_by_gs[j][5] = single_selective_coverage2[idx_min_k_random_lasso2]
    length_by_gs[j][5] = single_selective_lengths2[idx_min_k_random_lasso2]

length = len(sparsity_list)
def set_boxplot_style(bp, color,linestyle):
    plt.setp(bp['boxes'], color=color,linestyle=linestyle, linewidth=2)
    plt.setp(bp['whiskers'], color=color,linestyle=linestyle,linewidth=2)
    plt.setp(bp['caps'], color=color,linewidth=2)
    plt.setp(bp['medians'], color=color,linewidth=2)

fig = plt.figure(figsize=(17,5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
plt.sca(ax1)
first = plt.boxplot([coverage_by_gs[j][0] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3, sym='', widths=0.3)
second = plt.boxplot([coverage_by_gs[j][1] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 0.3, sym='', widths=0.3)
third = plt.boxplot([coverage_by_gs[j][2] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 0.6, sym='', widths=0.3)
fourth = plt.boxplot([coverage_by_gs[j][3] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 0.9, sym='', widths=0.3)
fifth = plt.boxplot([coverage_by_gs[j][4] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 1.2, sym='',widths=0.3)
sixth = plt.boxplot([coverage_by_gs[j][5] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 1.5, sym='',widths=0.3)
set_boxplot_style(first, '#2b8cbe','solid')
set_boxplot_style(second, '#6baed6','--')
set_boxplot_style(third, '#238443','solid')
set_boxplot_style(fourth, '#31a354','--')
set_boxplot_style(fifth, '#fd8d3c','solid')
set_boxplot_style(sixth,'#feb24c','--')
plt.xticks(range(1, (length) * 3 + 1, 3), [round(num, 2) for num in sparsity_list],fontsize=14)
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
first = plt.boxplot([length_by_gs[j][0] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3, sym='', widths=0.3)
second = plt.boxplot([length_by_gs[j][1] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 +.3, sym='', widths=0.3)
third = plt.boxplot([length_by_gs[j][2] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + .6, sym='', widths=0.3)
fourth = plt.boxplot([length_by_gs[j][3] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 0.9, sym='', widths=0.3)
fifth = plt.boxplot([length_by_gs[j][4] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 1.2, sym='',widths=0.3)
sixth = plt.boxplot([length_by_gs[j][5] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 1.5, sym='',widths=0.3)
set_boxplot_style(first, '#2b8cbe','solid')
set_boxplot_style(second, '#6baed6','--')
set_boxplot_style(third, '#238443','solid')
set_boxplot_style(fourth, '#31a354','--')
set_boxplot_style(fifth, '#fd8d3c','solid')
set_boxplot_style(sixth,'#feb24c','--')
plt.xticks(range(1, (length) * 3 + 1, 3), [round(num, 2) for num in sparsity_list],fontsize=14)
plt.xlim(-1, (length - 1) * 3 + 3)
plt.tight_layout()
plt.ylabel('Interval Length',fontsize=18)
plt.yticks(fontsize=14)

plt.sca(ax3)
first = plt.boxplot([f1_by_gs[j][0] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3, sym='', widths=0.3)
second = plt.boxplot([f1_by_gs[j][1] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 +.3, sym='', widths=0.3)
third = plt.boxplot([f1_by_gs[j][2] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + .6, sym='', widths=0.3)
fourth = plt.boxplot([f1_by_gs[j][3] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 0.9, sym='', widths=0.3)
fifth = plt.boxplot([f1_by_gs[j][4] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 1.2, sym='',widths=0.3)
sixth = plt.boxplot([f1_by_gs[j][5] for j in range(len(sparsity_list))], positions=np.array(range(length)) * 3 + 1.5, sym='',widths=0.3)
set_boxplot_style(first, '#2b8cbe','solid')
set_boxplot_style(second, '#6baed6','--')
set_boxplot_style(third, '#238443','solid')
set_boxplot_style(fourth, '#31a354','--')
set_boxplot_style(fifth, '#fd8d3c','solid')
set_boxplot_style(sixth,'#feb24c','--')
plt.xticks(range(1, (length) * 3 + 1, 3), [round(num, 2) for num in sparsity_list],fontsize=14)
plt.xlim(-1, (length - 1) * 3 + 3)
plt.tight_layout()
plt.ylabel('F1 per Simulation',fontsize=18)
plt.yticks(fontsize=14)


ax1.set_title("Coverage", y = 1.01,fontsize=20)
ax2.set_title("Length", y = 1.01,fontsize=20)
ax3.set_title("Accuracy", y = 1.01,fontsize=20)


def common_format(ax):
    ax.grid(True, which='both',color='#f0f0f0')
    ax.set_xlabel('Global Sparsity', fontsize=18)
    return ax

common_format(ax1)
common_format(ax2)
common_format(ax3)

# add target coverage on the first plot
ax1.axhline(y=0.9, color='k', linestyle='--', linewidth=2)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.2)
ax1.legend(loc='lower left', bbox_to_anchor=(0.6, -0.45),fontsize=20,ncol=3)
plt.savefig('vary_global_sparsity_p100.png', bbox_inches='tight')


#Plot sensitivity, specificity
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