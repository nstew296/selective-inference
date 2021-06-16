import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from selectinf.randomized.tests.test_multitask_lasso_2 import test_coverage

length_path = 10

lambdamin = 0.5
lambdamax = 5.0
#weights = np.arange(np.log(lambdamin), np.log(lambdamax), (np.log(lambdamax) - np.log(lambdamin)) / (length_path))
#feature_weight_list = np.exp(weights)
feature_weight_list = np.arange(lambdamin, lambdamax,(lambdamax - lambdamin) / (length_path))
print(feature_weight_list)

random_multitask_sensitivity = []
naive_multitask_sensitivity = []
data_splitting_sensitivity = []
k_random_lasso_sensitivity = []
one_random_lasso_sensitivity = []

random_multitask_specificity = []
naive_multitask_specificity = []
data_splitting_specificity = []
k_random_lasso_specificity = []
one_random_lasso_specificity = []

task_sparsity_list = [0.0, 0.25, 0.5]
for task_sparsity in task_sparsity_list:
    print(task_sparsity)

    sensitivity = {i: [[], [], [], [], []] for i in range(length_path)}
    specificity = {i: [[], [], [], [], []] for i in range(length_path)}
    error = {i: [[], [], [], [], []] for i in range(length_path)}

    for i in range(len(feature_weight_list)):

        sims = test_coverage(feature_weight_list[i],[3.0,5.0],task_sparsity,150)
        sensitivity[i][0].append(sims[13])
        sensitivity[i][1].append(sims[14])
        sensitivity[i][2].append(sims[15])
        sensitivity[i][3].append(sims[16])
        sensitivity[i][4].append(sims[17])
        specificity[i][0].append(sims[18])
        specificity[i][1].append(sims[19])
        specificity[i][2].append(sims[20])
        specificity[i][3].append(sims[21])
        specificity[i][4].append(sims[22])
        error[i][0].append(sims[23])
        error[i][1].append(sims[24])
        error[i][2].append(sims[25])
        error[i][3].append(sims[26])
        error[i][4].append(sims[27])

    selective_sensitivity = [sensitivity[i][0] for i in range(length_path)]
    naive_sensitivity = [sensitivity[i][1] for i in range(length_path)]
    ds_sensitivity = [sensitivity[i][2] for i in range(length_path)]
    single_task_sensitivity = [sensitivity[i][3] for i in range(length_path)]
    one_lasso_sensitivity = [sensitivity[i][4] for i in range(length_path)]

    selective_specificity = [specificity[i][0] for i in range(length_path)]
    naive_specificity = [specificity[i][1] for i in range(length_path)]
    ds_specificity = [specificity[i][2] for i in range(length_path)]
    single_task_specificity = [specificity[i][3] for i in range(length_path)]
    one_lasso_specificity = [specificity[i][4] for i in range(length_path)]

    selective_error = [error[i][0] for i in range(length_path)]
    naive_error = [error[i][1] for i in range(length_path)]
    ds_error = [error[i][2] for i in range(length_path)]
    single_selective_error = [error[i][3] for i in range(length_path)]
    one_lasso_error = [error[i][4] for i in range(length_path)]

    print(selective_error,naive_error,ds_error,single_selective_error,one_lasso_error)

    idx_min_random_multitask = np.argmin(selective_error)
    idx_min_naive_multitask = np.argmin(naive_error)
    idx_min_data_splitting = np.argmin(ds_error)
    idx_min_k_random_lasso = np.argmin(single_selective_error)
    idx_min_one_lasso = np.argmin(one_lasso_error)

    print(idx_min_random_multitask, idx_min_naive_multitask, idx_min_data_splitting, idx_min_k_random_lasso, idx_min_one_lasso)

    print(selective_sensitivity[idx_min_random_multitask])

    random_multitask_sensitivity.extend(selective_sensitivity[idx_min_random_multitask])
    random_multitask_specificity.extend(selective_specificity[idx_min_random_multitask])
    naive_multitask_sensitivity.extend(naive_sensitivity[idx_min_naive_multitask])
    naive_multitask_specificity.extend(naive_specificity[idx_min_naive_multitask])
    data_splitting_sensitivity.extend(ds_sensitivity[idx_min_data_splitting])
    data_splitting_specificity.extend(ds_specificity[idx_min_data_splitting])
    k_random_lasso_sensitivity.extend(single_task_sensitivity[idx_min_k_random_lasso])
    k_random_lasso_specificity.extend(single_task_specificity[idx_min_k_random_lasso])
    one_random_lasso_sensitivity.extend(one_lasso_sensitivity[idx_min_one_lasso])
    one_random_lasso_specificity.extend(one_lasso_specificity[idx_min_one_lasso])

    print(random_multitask_sensitivity,naive_multitask_sensitivity,data_splitting_sensitivity,k_random_lasso_sensitivity,one_random_lasso_sensitivity)


fig = plt.figure(figsize=(25, 10))
fig.tight_layout()
fig.add_subplot(1, 2, 1)
plt.plot(task_sparsity_list, random_multitask_sensitivity, c='#D7191C')
plt.plot(task_sparsity_list, naive_multitask_sensitivity, c='#2b8cbe')
plt.plot(task_sparsity_list, data_splitting_sensitivity, c='#31a354')
plt.plot(task_sparsity_list, k_random_lasso_sensitivity, c='#c51b8a')
plt.plot(task_sparsity_list, one_random_lasso_sensitivity, c='#feb24c')
plt.plot([], c='#D7191C', label='Randomized Multi-Task Lasso')
plt.plot([], c='#2b8cbe', label='Naive Multi-Task Lasso')
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
plt.plot(task_sparsity_list, one_random_lasso_specificity, c='#feb24c')
plt.plot([], c='#D7191C', label='Randomized Multi-Task Lasso')
plt.plot([], c='#2b8cbe', label='Naive Multi-Task Lasso')
plt.plot([], c='#31a354', label='Data Splitting')
plt.plot([], c='#c51b8a', label='K Randomized Lassos')
plt.plot([], c='#feb24c', label='One Randomized Lasso')
plt.legend()
plt.tight_layout()
plt.ylabel('Average Specificity')
plt.xlabel('Task Sparsity')
plt.title('Specificity by Task Sparsity')
plt.savefig('model_selection_compare.png')