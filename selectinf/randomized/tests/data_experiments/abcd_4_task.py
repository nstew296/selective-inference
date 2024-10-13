from scipy.stats import t as tdist
import seaborn as sns
from abcd_functions import *

np.random.seed(5)


def _noise(n, df=np.inf):
    if df == np.inf:
        return np.random.standard_normal(n)
    else:
        sd_t = np.std(tdist.rvs(df, size=50000))
    return tdist.rvs(df, size=n) / sd_t


# Load fmri and ABCD task data (skip header)
predictors_train = np.genfromtxt('train.csv', delimiter=',')[1:, :-12]
predictors_validate = np.genfromtxt('validate.csv', delimiter=',')[1:, :-12]
predictors_test = np.genfromtxt('test.csv', delimiter=',')[1:, :-12]

responses_train = {}
responses_validate = {}
responses_test = {}
task_index = [-5, -11, -2, -9]
ntask = len(task_index)

# Scale response variables
for i in range(ntask):
    responses_train[i] = np.genfromtxt('train.csv', delimiter=',')[1:, task_index[i]]
    scale = np.std(responses_train[i])
    responses_train[i] /= scale
    responses_validate[i] = np.genfromtxt('validate.csv', delimiter=',')[1:, task_index[i]]
    responses_validate[i] /= scale
    responses_test[i] = np.genfromtxt('test.csv', delimiter=',')[1:, task_index[i]]
    responses_test[i] /= scale

# PC loadings and standard deviation of PC scores (singular values)
V = np.genfromtxt('V.csv', delimiter=',')[1:, :]
D = np.genfromtxt('D.csv', delimiter=',')[1:]

# Generate randomization variables to use across methods
noise = _noise(predictors_train.shape[1] * ntask)

################################################
# Data Splitting 67/33
################################################
sample_sizes = predictors_train.shape[0]
samples = np.arange(int(sample_sizes))
selection = np.random.choice(samples, size=int(0.67 * sample_sizes), replace=False)
inference = np.setdiff1d(samples, selection)

responses_selection = {j: responses_train[j][selection] for j in range(ntask)}
predictors_selection = predictors_train[selection, :]
responses_inference = {j: responses_train[j][inference] for j in range(ntask)}
predictors_inference = predictors_train[inference, :]

weight_list = np.linspace(0.5, 3.0, 10)
active, pred_r_validate = ds_multi_task_tune(predictors_selection, predictors_inference, predictors_validate,
                                             responses_selection, responses_inference, responses_validate, weight_list)
final_weight = np.argmax([np.sum(pred_r_validate[x]) / ntask for x in weight_list])

colors = ['red', 'green', 'orange', 'blue']
markers = ['|', '*', 'd', '.']
for i in range(ntask):
    print([np.sum(active[x] != 0) for x in weight_list])
    plt.plot([np.sum(active[x] != 0) / ntask for x in weight_list], [pred_r_validate[x][i] for x in weight_list],
             color=colors[i], marker=markers[i], markersize=5)
    plt.ylim(0.15, 0.4)
plt.savefig("data_splitting_test_67_30.png")
plt.clf()

final_estimates_ds67, final_intervals_ds67, ds67_interval_lengths, all_variables_ds67, significant_variables_ds67, \
    final_err_ds67, pred_r_ds67, coefs_var_ds67 = ds_multi_task_selection_inference(predictors_selection,
                                                                                    predictors_inference,
                                                                                    predictors_test,
                                                                                    responses_selection,
                                                                                    responses_inference,
                                                                                    responses_test,
                                                                                    weight=weight_list[final_weight])

print(final_err_ds67, "Average testing error per task, data split 67/33")
print(pred_r_ds67, "Predictive r, data split 67/33")
print(np.mean(ds67_interval_lengths), "Mean interval length, data split 67/33")
print(np.std(ds67_interval_lengths), "Sd interval length, data split 67/33")
print(np.sum([len(all_variables_ds67[i]) for i in range(len(all_variables_ds67))]), "Sum of selected across tasks")
print(np.sum([len(significant_variables_ds67[i]) for i in range(len(significant_variables_ds67))]),
      "Sum of significant across tasks")

###############################################
# Data Splitting 50/50
###############################################

sample_sizes = predictors_train.shape[0]
samples = np.arange(int(sample_sizes))
selection = np.random.choice(samples, size=int(0.5 * sample_sizes), replace=False)
inference = np.setdiff1d(samples, selection)

responses_selection = {j: responses_train[j][selection] for j in range(ntask)}
predictors_selection = predictors_train[selection, :]
responses_inference = {j: responses_train[j][inference] for j in range(ntask)}
predictors_inference = predictors_train[inference, :]

weight_list = np.linspace(0.6, 2.7, 10)
active, pred_r_validate = ds_multi_task_tune(predictors_selection, predictors_inference, predictors_validate,
                                             responses_selection, responses_inference, responses_validate, weight_list)
final_weight = np.argmax([np.sum(pred_r_validate[x]) / ntask for x in weight_list])

colors = ['red', 'green', 'orange', 'blue']
markers = ['|', '*', 'd', '.']
for i in range(ntask):
    print([np.sum(active[x] != 0) for x in weight_list])
    plt.plot([np.sum(active[x] != 0) / ntask for x in weight_list], [pred_r_validate[x][i] for x in weight_list],
             color=colors[i], marker=markers[i], markersize=5)
    plt.ylim(0.15, 0.4)
plt.savefig("data_splitting_test_50_50.png")

final_estimates_ds50, final_intervals_ds50, ds50_interval_lengths, all_variables_ds50, significant_variables_ds50, \
    final_err_ds50, pred_r_ds50, coefs_var_ds50 = ds_multi_task_selection_inference(predictors_selection,
                                                                                    predictors_inference,
                                                                                    predictors_test,
                                                                                    responses_selection,
                                                                                    responses_inference,
                                                                                    responses_test,
                                                                                    weight=weight_list[final_weight])

print(final_err_ds50, "Average testing error per task, data split 50/50")
print(pred_r_ds50, "Predictive r, data split 50/50")
print(np.mean(ds50_interval_lengths), "Mean interval length, data split 50/50")
print(np.std(ds50_interval_lengths), "Sd interval length, data split 50/50")
print(np.sum([len(all_variables_ds50[i]) for i in range(len(all_variables_ds50))]), "Sum of selected across tasks")
print(np.sum([len(significant_variables_ds50[i]) for i in range(len(significant_variables_ds50))]),
      "Sum of significant PCs in total")

###############################################
# Joint MTL+SI (0.7)
###############################################

weight_list = standardize_lambda_path_randomized_mtl(predictors_train, responses_train, noise, start_size=170,
                                                     stop_size=15)
active, pred_r_validate = rand_multi_task_tune(predictors_train, predictors_validate, responses_train,
                                               responses_validate, weight_list, noise, rand_scale=0.7)
final_weight = np.argmax([np.sum(pred_r_validate[x]) / ntask for x in weight_list])

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
plt.sca(ax1)
colors = ['red', 'green', 'orange', 'blue']
markers = ['.', '*', 'd', '|']
markersizes = [6, 5, 4.5, 6.5]
for i in range(ntask):
    print([np.sum(active[x] != 0) for x in weight_list])
    plt.plot([np.sum(active[x] != 0) / ntask for x in weight_list], [pred_r_validate[x][i] for x in weight_list],
             color=colors[i], marker=markers[i], markersize=markersizes[i], zorder=3)

final_estimates_joint, final_intervals_joint, interval_lengths_joint, all_variables_joint, significant_variables_joint, \
    final_err_joint, pred_r_joint, coefs_var_joint = rand_multi_task_selection_inference(predictors_train,
                                                                                         predictors_test,
                                                                                         responses_train,
                                                                                         responses_test,
                                                                                         weight_list[final_weight],
                                                                                         noise,
                                                                                         rand_scale=0.7)

print(final_err_joint, "Average testing error per task, rand scale 0.7")
print(pred_r_joint, "Predictive r, rand scale 0.7")
print(np.mean(interval_lengths_joint), "Mean interval length, rand scale 0.7")
print(np.std(interval_lengths_joint), "Sd interval length, rand scale 0.7")
print(np.sum([len(all_variables_joint[i]) for i in range(len(all_variables_joint))]), "Sum of selected in total")
print(np.sum([len(significant_variables_joint[i]) for i in range(len(significant_variables_joint))]),
      "Sum of significant in total")
print(significant_variables_joint, "Significant PCs by task")
print(all_variables_joint, "All variables")

###############################################
# Comparison to data-splitting
###############################################

# Create dictionary to store lengths by task
interval_len_by_task = {}
start = 0
for i in range(ntask):
    interval_len_by_task[i] = interval_lengths_joint[start:start + len(all_variables_joint[i])]
    start += len(all_variables_joint[i])

interval_len_by_task2 = {}
start2 = 0
for i in range(ntask):
    interval_len_by_task2[i] = ds67_interval_lengths[start2:start2 + len(all_variables_ds67[i])]
    start2 += len(all_variables_ds67[i])

interval_len_by_task3 = {}
start3 = 0
for i in range(ntask):
    interval_len_by_task3[i] = ds50_interval_lengths[start3:start3 + len(all_variables_ds50[i])]
    start3 += len(all_variables_ds50[i])

common_67 = {i: np.intersect1d(all_variables_joint[i], all_variables_ds67[i]) for i in range(ntask)}
common_significant_67 = {i: np.intersect1d(significant_variables_joint[i], significant_variables_ds67[i]) for i in
                         range(ntask)}

common_50 = {i: np.intersect1d(all_variables_joint[i], all_variables_ds50[i]) for i in range(ntask)}
common_significant_50 = {i: np.intersect1d(significant_variables_joint[i], significant_variables_ds50[i]) for i in
                         range(ntask)}

print("Common variables between MTL + SI and DS 67/33", common_67)
print("Common significant variables between MTL + SI and DS 67/33", common_significant_67)
print("Common variables between MTL + SI and DS 50/50", common_50)
print("Common significant variables between MTL + SI and DS 50/50", common_significant_50)

# Compute length ratio for shared parameters
common_lengths_67 = []
for i in range(ntask):
    for predictor in common_67[i]:
        ratio_length = interval_len_by_task2[i][np.argwhere(all_variables_ds67[i] == predictor)[0][0]] / \
                       interval_len_by_task[i][np.argwhere(all_variables_joint[i] == predictor)[0][0]]
        common_lengths_67.append(ratio_length)

common_lengths_50 = []
for i in range(ntask):
    for predictor in common_50[i]:
        ratio_length = interval_len_by_task3[i][np.argwhere(all_variables_ds50[i] == predictor)[0][0]] / \
                       interval_len_by_task[i][np.argwhere(all_variables_joint[i] == predictor)[0][0]]
        common_lengths_50.append(ratio_length)

# Estimate coefficients in original feature space
# Since I scaled the PCs by their sds (i.e. the singular values of the data) before performing model selection,
# I have divided the regression coefficients by the singular values before projecting back to the original space
running_counter = 0
original_coef_approx = np.zeros((np.shape(V)[0], ntask))
original_coef_approx_scaled = np.zeros((np.shape(V)[0], ntask))

for i in range(ntask):
    singular_values = D[all_variables_joint[i]]
    original_coef_approx[:, i] = V[:, all_variables_joint[i]].dot(
        np.divide(final_estimates_joint[running_counter:running_counter + len(all_variables_joint[i])],
                  singular_values))

    running_counter += len(all_variables_joint[i])

np.savetxt("original_approx07.csv", original_coef_approx, delimiter=",")

###############################################
# Pairwise MTL + SI - crystallized tasks
###############################################

task_index = [-5, -11]
ntask = len(task_index)
responses_train = {}
responses_validate = {}

for i in range(ntask):
    responses_train[i] = np.genfromtxt('train.csv', delimiter=',')[1:, task_index[i]]
    scale = np.std(responses_train[i])
    responses_train[i] /= scale
    responses_validate[i] = np.genfromtxt('validate.csv', delimiter=',')[1:, task_index[i]]
    responses_validate[i] /= scale
    responses_test[i] = np.genfromtxt('test.csv', delimiter=',')[1:, task_index[i]]
    responses_test[i] /= scale
    print("here")

weight_list = standardize_lambda_path_randomized_mtl(predictors_train, responses_train,
                                                     noise[:predictors_train.shape[1] * ntask],
                                                     start_size=170, stop_size=15)
active, pred_r_validate = rand_multi_task_tune(predictors_train, predictors_validate, responses_train,
                                               responses_validate, weight_list,
                                               noise[:predictors_train.shape[1] * ntask], rand_scale=0.7)
final_weight = np.argmax([np.sum(pred_r_validate[x]) / ntask for x in weight_list])

colors = ['red', 'green']
markers = ['.', '*']
markersizes = [6, 5]
for i in range(ntask):
    print([np.sum(active[x] != 0) for x in weight_list])
    plt.plot([np.sum(active[x] != 0) / ntask for x in weight_list], [pred_r_validate[x][i] for x in weight_list],
             linestyle='--', color=colors[i], marker=markers[i], markersize=markersizes[i], zorder=2)

final_estimates_crystallized, final_intervals_crystallized, interval_lengths_crystallized, all_variables_crystallized,\
    significant_variables_crystallized, final_err_crystallized, pred_r_crystallized, coefs_var_crystallized = \
    rand_multi_task_selection_inference(predictors_train,
                                        predictors_test,
                                        responses_train,
                                        responses_test,
                                        weight_list[final_weight],
                                        noise[:predictors_train.shape[1] * ntask],
                                        rand_scale=0.7)

print(final_err_crystallized, "Average testing error per task, rand scale 0.7")
print(pred_r_crystallized, "Predictive r, rand scale 0.7")
print(np.mean(interval_lengths_crystallized), "Mean interval length, rand scale 0.7")
print(np.std(interval_lengths_crystallized), "Sd interval length, rand scale 0.7")
print(np.sum([len(all_variables_crystallized[i]) for i in range(len(all_variables_crystallized))]),
      "Sum of selected in total")
print(np.sum([len(significant_variables_crystallized[i]) for i in range(len(significant_variables_crystallized))]),
      "Sum of significant in total")
print(significant_variables_crystallized, "Significant PCs by task")

separate_significant_variables_rand07 = significant_variables_crystallized
separate_all_variables_rand07 = all_variables_crystallized
separate_estimates_rand07 = final_estimates_crystallized
separate_intervals_rand07 = interval_lengths_crystallized

###############################################
# Pairwise MTL+SI - Fluid tasks
###############################################

task_index = [-2, -9]
ntask = len(task_index)
responses_train = {}
responses_validate = {}
responses_test = {}

for i in range(ntask):
    responses_train[i] = np.genfromtxt('train.csv', delimiter=',')[1:, task_index[i]]
    scale = np.std(responses_train[i])
    responses_train[i] /= scale
    responses_validate[i] = np.genfromtxt('validate.csv', delimiter=',')[1:, task_index[i]]
    responses_validate[i] /= scale
    responses_test[i] = np.genfromtxt('test.csv', delimiter=',')[1:, task_index[i]]
    responses_test[i] /= scale
    print("here")

weight_list = standardize_lambda_path_randomized_mtl(predictors_train, responses_train,
                                                     noise[predictors_train.shape[1] * ntask:],
                                                     start_size=170, stop_size=15)
active, pred_r_validate = rand_multi_task_tune(predictors_train, predictors_validate, responses_train,
                                               responses_validate, weight_list,
                                               noise[predictors_train.shape[1] * ntask:], rand_scale=0.7)
final_weight = np.argmax([np.sum(pred_r_validate[x]) / ntask for x in weight_list])

colors = ['orange', 'blue']
markers = ['d', '|']
markersizes = [4.5, 6.5]
for i in range(ntask):
    print([np.sum(active[x] != 0) for x in weight_list])
    plt.plot([np.sum(active[x] != 0) / ntask for x in weight_list], [pred_r_validate[x][i] for x in weight_list],
             linestyle='--', color=colors[i], marker=markers[i], markersize=markersizes[i], zorder=1)
plt.legend(['RC - Joint', 'PV - Joint', 'Matrix - Joint', 'LS - Joint',
            'RC - Pairwise', 'PV - Pairwise', 'Matrix - Pairwise', 'LS - Pairwise'], ncol=2)
plt.ylim(0.175, 0.4)
plt.ylabel("Predictive r on Validation Data", fontsize=16)
plt.xlabel("Average model size", fontsize=14)
plt.tight_layout()
common_format(ax1)

final_estimates_fluid, final_intervals_fluid, fluid_intervals, all_variables_fluid, significant_variables_fluid, \
    final_err_fluid, pred_r_fluid, info_mat_fluid, coefs_var_fluid = \
    rand_multi_task_selection_inference(predictors_train,
                                        predictors_test,
                                        responses_train,
                                        responses_test,
                                        weight_list[final_weight],
                                        noise[predictors_train.shape[1]
                                              * ntask:],
                                        rand_scale=0.7)
print(final_err_fluid, "Average testing error per task, rand scale 0.7")
print(pred_r_fluid, "Predictive r, rand scale 0.7")
print(np.mean(fluid_intervals), "Mean interval length, rand scale 0.7")
print(np.std(fluid_intervals), "Sd interval length, rand scale 0.7")
print(np.sum([len(all_variables_fluid[i]) for i in range(len(all_variables_fluid))]), "Sum of selected in total")
print(np.sum([len(significant_variables_fluid[i]) for i in range(len(significant_variables_fluid))]),
      "Sum of significant in total")
print(significant_variables_fluid, "Significant PCs by task")

# Add fluid results to dictionaries for separate approach
separate_significant_variables_rand07[2] = significant_variables_fluid[0]
separate_significant_variables_rand07[3] = significant_variables_fluid[1]
separate_all_variables_rand07[2] = all_variables_fluid[0]
separate_all_variables_rand07[3] = all_variables_fluid[1]
separate_estimates_rand07 = np.concatenate([separate_estimates_rand07, final_estimates_fluid])
separate_intervals_lengths = np.concatenate([separate_intervals_rand07, fluid_intervals])
print(separate_intervals_lengths)

# plt.clf()
plt.sca(ax2)
plt.boxplot(interval_lengths_joint, positions=[1])
plt.boxplot(separate_intervals_lengths, positions=[2])
plt.xticks([1, 2], labels=['Joint', 'Pairwise'], fontsize=14)
plt.ylabel("Confidence Interval Length", fontsize=16)
plt.tight_layout()
common_format(ax2)
plt.savefig("ci_length_comparison.png")
plt.clf()

jacard_matrix = np.zeros((4, 4))

for i in range(4):
    for j in range(4):
        jacard_matrix[i, j] = round(len(np.intersect1d(significant_variables_joint[i],
                                                       significant_variables_joint[j])) /
                                    len(np.union1d(significant_variables_joint[i],
                                                   significant_variables_joint[j])), 2)

print(jacard_matrix)

j_list = []
for i in range(4):
    for j in range(4):
        if j > i:
            j_list.append(jacard_matrix[i, j])

print(np.mean(j_list))
fig = plt.figure(figsize=(7, 5))
mat = sns.heatmap(jacard_matrix, vmin=0, vmax=1, cmap="viridis_r")
mat.set_xticklabels(['RC', 'PV', 'Matrix', 'LS'], rotation=90)
mat.set_yticklabels(['RC', 'PV', 'Matrix', 'LS'], rotation=0)
fig = mat.get_figure()
fig.tight_layout()
fig.savefig("jaccard_MTL_fluid_and_crystalized_together.png")
plt.clf()

jacard_matrix = np.zeros((4, 4))

for i in range(4):
    for j in range(4):
        jacard_matrix[i, j] = round(len(np.intersect1d(separate_significant_variables_rand07[i],
                                                       separate_significant_variables_rand07[j])) /
                                    len(np.union1d(separate_significant_variables_rand07[i],
                                                   separate_significant_variables_rand07[j])), 2)

print(jacard_matrix)

j_list = []
for i in range(4):
    for j in range(4):
        if j > i:
            j_list.append(jacard_matrix[i, j])

print(np.mean(j_list))
fig = plt.figure(figsize=(7, 5))
mat = sns.heatmap(jacard_matrix, vmin=0, vmax=1, cmap="viridis_r")
mat.set_xticklabels(['RC', 'PV', 'Matrix', 'LS'], rotation=90)
mat.set_yticklabels(['RC', 'PV', 'Matrix', 'LS'], rotation=0)
fig = mat.get_figure()
fig.tight_layout()
fig.savefig("jaccard_MTL_fluid_and_crystalized_separate.png")
plt.clf()

fig = plt.figure(figsize=(16, 5.5))
ax1 = fig.add_subplot(122)
plt.sca(ax1)
plt.boxplot([common_lengths_67], positions=[1], widths=0.4)
plt.boxplot([common_lengths_50], positions=[2], widths=0.4)
plt.xticks([1, 2], labels=['DS (0.67): MTL(0.7) + SI', 'DS(0.5): MTL(0.7) + SI'], fontsize=16)
plt.tight_layout()
plt.ylabel('Ratio of Lengths for Common Parameters', fontsize=16)
plt.yticks(fontsize=18)


common_format(ax1)
ax1.axhline(y=1.0, color='k', linestyle='--', linewidth=2.5)

ax2 = fig.add_subplot(121)
plt.sca(ax2)
plt.boxplot([interval_lengths_joint], positions=np.asarray([1]), widths=0.4)
plt.boxplot([ds67_interval_lengths], positions=np.asarray([1.5]), widths=0.4)
plt.boxplot([ds50_interval_lengths], positions=np.asarray([2.0]), widths=0.4)
plt.xticks([1, 1.5, 2.0], labels=['MTL (0.7) + SI', 'DS (0.67)', 'DS (0.5)'], fontsize=16)
plt.tight_layout()
plt.ylabel('Interval Length', fontsize=20)
plt.yticks(fontsize=16)
common_format(ax2)
plt.savefig('real_data_lengths_cv.png', bbox_inches='tight')
