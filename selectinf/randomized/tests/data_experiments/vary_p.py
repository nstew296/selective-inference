import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from selectinf.randomized.tests.test_multitask_lasso import test_inference

k=5
#global_sparsity = [0.9167,0.9667,0.9833,0.9888]
#global_sparsity = [0.9375,0.975,0.9875,0.99167]
global_sparsity = [0.9375,0.975,0.9875,0.99375]
#global_sparsity = [0.83333,0.9667,0.9833]
#task_sparsity = 0.4
task_sparsity = 0.2

length_path = 15
lambdamin_ds = 1.0
lambdamax_ds = 4.0
feature_weight_list_ds = np.linspace(lambdamin_ds, lambdamax_ds,length_path)

lambdamin_si = 2.0
lambdamax_si = 5.0
feature_weight_list_si = np.linspace(lambdamin_si, lambdamax_si,length_path)

p_list = [100,250,500,1000]
#n_list = [100,100,100]
#p_list = [100,250,500,750]
n_list = [100,100,100,100]

coverage_by_p = {j: [[], [], [], [], [], [], []] for j in range(len(p_list))}
length_by_p = {j: [[], [], [], [], [], [], []] for j in range(len(p_list))}
f1_by_p = {j: [[], [], [], [], [], []] for j in range(len(p_list))}


for j in range(len(p_list)):
    positive = (1.-global_sparsity[j])*(1.-task_sparsity)*k*p_list[j]
    negative = k*p_list[j] - positive

    selective_error = []
    selective_error2 = []
    ds_error = []
    ds_error2 = []
    single_selective_error = []
    single_selective_error2 = []

    for i in range(length_path):
        print((i,j),"(i,j)")
        weight = [feature_weight_list_si[i]]*2
        weight.extend([feature_weight_list_ds[i]]*2)
        weight.extend([feature_weight_list_si[i]]*2)
        print(weight)
        sims = test_inference(weight,[1.0,3.0],p_list[j],task_sparsity,global_sparsity[j],nsim=10,seed=5)
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

    feature_weight_list2 = [feature_weight_list_si[idx_min_random_multitask],feature_weight_list_si[idx_min_random_multitask2],
                           feature_weight_list_ds[idx_min_data_splitting],
                           feature_weight_list_ds[idx_min_data_splitting2],feature_weight_list_si[idx_min_k_random_lasso],
                           feature_weight_list_si[idx_min_k_random_lasso2]]


    sims = test_inference(feature_weight_list2,[1.0,3.0],p_list[j],task_sparsity,global_sparsity[j],nsim=n_list[j]+10,seed=5)

    selective_coverage = coverage_by_p[j][0] = sims["MTL_SI_07_coverage"][10:]
    selective_coverage2 = coverage_by_p[j][1] = sims["MTL_SI_1_coverage"][10:]
    ds_coverage = coverage_by_p[j][3] = sims["DS_67_coverage"][10:]
    ds_coverage2 = coverage_by_p[j][4] = sims["DS_50_coverage"][10:]
    single_selective_coverage = coverage_by_p[j][5] = sims["LASSO_SI_07_coverage"][10:]
    single_selective_coverage2 = coverage_by_p[j][6] = sims["LASSO_SI_1_coverage"][10:]

    selective_lengths = length_by_p[j][0] = sims["MTL_SI_07_length"][10:]
    selective_lengths2 = length_by_p[j][1] = sims["MTL_SI_1_length"][10:]
    ds_lengths = length_by_p[j][3] = sims["DS_67_length"][10:]
    ds_lengths2 = length_by_p[j][4] = sims["DS_50_length"][10:]
    single_selective_lengths = length_by_p[j][5] = sims["LASSO_SI_07_length"][10:]
    single_selective_lengths2 = length_by_p[j][6] = sims["LASSO_SI_1_length"][10:]

    selective_sensitivity = sims["MTL_SI_07_sensitivity"][10:]
    selective_sensitivity2 = sims["MTL_SI_1_sensitivity"][10:]
    ds_sensitivity = sims["DS_67_sensitivity"][10:]
    ds_sensitivity2 = sims["DS_50_sensitivity"][10:]
    single_task_sensitivity = sims["LASSO_SI_07_sensitivity"][10:]
    single_task_sensitivity2 = sims["LASSO_SI_1_sensitivity"][10:]

    selective_specificity = sims["MTL_SI_07_specificity"][10:]
    selective_specificity2 = sims["MTL_SI_1_specificity"][10:]
    ds_specificity = sims["DS_67_specificity"][10:]
    ds_specificity2 = sims["DS_50_specificity"][10:]
    single_task_specificity = sims["LASSO_SI_07_specificity"][10:]
    single_task_specificity2 = sims["LASSO_SI_1_specificity"][10:]


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

length = len(p_list)
def set_boxplot_style(bp, color,linestyle):
    plt.setp(bp['boxes'], color=color,linestyle=linestyle,linewidth=2)
    plt.setp(bp['whiskers'], color=color,linestyle=linestyle,linewidth=2)
    plt.setp(bp['caps'], color=color,linewidth=2)
    plt.setp(bp['medians'], color=color,linewidth=2)

def common_format(ax):
    ax.grid(True, which='both',color='#f0f0f0')
    ax.set_xlabel('p', fontsize=18)
    return ax

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
set_boxplot_style(first, '#2b8cbe','solid')
set_boxplot_style(second, '#6baed6','--')
set_boxplot_style(third, '#238443','solid')
set_boxplot_style(fourth, '#31a354','--')
set_boxplot_style(fifth, '#fd8d3c','solid')
set_boxplot_style(sixth,'#feb24c','--')
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
set_boxplot_style(first, '#2b8cbe','solid')
set_boxplot_style(second, '#6baed6','--')
set_boxplot_style(third, '#238443','solid')
set_boxplot_style(fourth, '#31a354','--')
set_boxplot_style(fifth, '#fd8d3c','solid')
set_boxplot_style(sixth,'#feb24c','--')
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
set_boxplot_style(first, '#2b8cbe','solid')
set_boxplot_style(second, '#6baed6','--')
set_boxplot_style(third, '#238443','solid')
set_boxplot_style(fourth, '#31a354','--')
set_boxplot_style(fifth, '#fd8d3c','solid')
set_boxplot_style(sixth,'#feb24c','--')
plt.xticks(range(1, (length) * 3 + 1, 3), p_list,fontsize=14)
plt.xlim(-1, (length - 1) * 3 + 3)
plt.tight_layout()
plt.ylabel('F1 per Simulation',fontsize=18)
plt.yticks(fontsize=14)

ax1.set_title("Coverage", y = 1.01,fontsize=20)
ax2.set_title("Mean Length", y = 1.01,fontsize=20)
ax3.set_title("Accuracy", y = 1.01,fontsize=20)

common_format(ax1)
common_format(ax2)
common_format(ax3)

# add target coverage on the first plot
ax1.axhline(y=0.9, color='k', linestyle='--', linewidth=2)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ax1.legend(loc='lower left', bbox_to_anchor=(0.7, -0.45),fontsize=18,ncol=3)
plt.savefig('vary_p_n500_ts2.png', bbox_inches='tight')
