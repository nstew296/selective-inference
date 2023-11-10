import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as tdist
from scipy.stats import norm as ndist
import seaborn as sns
import regreg.api as rr
from selectinf.randomized.randomization import randomization
from selectinf.randomized.multitask_lasso import multi_task_lasso

np.random.seed(5)

ntask = 4
# Load fmri and ABCD task data
predictors_train = np.genfromtxt('train.csv', delimiter=',')[1:, :-12]
predictors_validate = np.genfromtxt('validate.csv', delimiter=',')[1:, :-12]
predictors_test = np.genfromtxt('test.csv', delimiter=',')[1:, :-12]

responses_train = {}
responses_validate = {}
responses_test = {}
task_index = [-5, -11, -2, -9]

# Scale response variables
for i in range(ntask):
    responses_train[i] = np.genfromtxt('train.csv', delimiter=',')[1:, task_index[i]]
    scale = np.std(responses_train[i])
    responses_train[i] /= scale
    responses_validate[i] = np.genfromtxt('validate.csv', delimiter=',')[1:, task_index[i]]
    responses_validate[i] /= scale
    responses_test[i] = np.genfromtxt('test.csv', delimiter=',')[1:, task_index[i]]
    responses_test[i] /= scale

# PC loadings and singular values
V = np.genfromtxt('V.csv', delimiter=',')[1:, :]
sv = np.genfromtxt('lambda.csv', delimiter=',')[1:]


def _noise(n, df=np.inf):
    if df == np.inf:
        return np.random.standard_normal(n)
    else:
        sd_t = np.std(tdist.rvs(df, size=50000))
    return tdist.rvs(df, size=n) / sd_t


# Generate randomization variables to use across methods
noise = _noise(predictors_train.shape[1] * ntask)


def ds_multi_task_tune(predictor_vars_selection, predictor_vars_inference, predictor_vars_validate,
                       response_selection, response_inference, response_validate, weight_list):

    nfeatures = predictor_vars_selection.shape[1]
    sample_sizes_inference = predictor_vars_inference.shape[0]

    ridge_terms = np.zeros(ntask)
    randomizers = None

    # Estimate noise level (data is centered, saving 1 dof)
    noise_levels = []
    for j in range(ntask):
        noise_levels.append(np.sqrt(np.sum(np.asarray(response_inference[j] - predictor_vars_inference.dot(
            np.linalg.pinv(predictor_vars_inference).dot(response_inference[j]))) ** 2) /
                                    (sample_sizes_inference - nfeatures)))

    active_dict = {}
    pred_r_validate = {}

    # Fit MTL model (randomization/perturbation variable set to zero)
    for weight in weight_list:
        feature_weight = weight * np.ones(nfeatures)
        perturbations = np.zeros((nfeatures, ntask))
        loglikes = {j: rr.glm.gaussian(predictor_vars_selection, response_selection[j], coef=1., quadratic=None) for j
                    in range(ntask)}
        multi_lasso = multi_task_lasso(loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures, ntask,
                                       perturbations)
        active_signs = multi_lasso.fit(perturbations=perturbations)

        # Compute predictive r on validation data
        predictive_r = []
        if (active_signs != 0).sum() > 0:
            for j in range(ntask):
                if (active_signs[:, j] != 0).sum() == 0:
                    predictive_r.append(None)
                else:
                    X = predictor_vars_inference
                    y = response_inference[j]
                    observed_target = np.linalg.pinv(X[:, (active_signs[:, j] != 0)]).dot(y)
                    predictive_r.append(
                        np.corrcoef(response_validate[j], predictor_vars_validate[:, (active_signs[:, j] != 0)].dot(
                            observed_target))[0, 1])
                    print(predictive_r)

        active_dict[weight] = active_signs
        pred_r_validate[weight] = predictive_r

    return active_dict, pred_r_validate


def ds_multi_task_selection_inference(predictor_vars_selection, predictor_vars_inference,
                                      predictor_vars_test, response_selection, response_inference,
                                      response_test, weight):
    ridge_terms = np.zeros(ntask)
    randomizers = None

    nfeatures = predictor_vars_selection.shape[1]
    sample_sizes_inference = predictor_vars_inference.shape[0]
    sample_sizes_test = predictor_vars_test.shape[0]

    # Estimate noise level (data is centered, saving 1 dof)
    noise_levels = []
    for j in range(ntask):
        noise_levels.append(np.sqrt(np.sum(np.asarray(response_inference[j] - predictor_vars_inference.dot(
            np.linalg.pinv(predictor_vars_inference).dot(response_inference[j]))) ** 2) / (
                                            sample_sizes_inference - nfeatures)))
    dispersions = [noise_levels[j] ** 2 for j in range(len(noise_levels))]

    # Fit MTL model (randomization/perturbation variable set to zero)
    feature_weight = weight * np.ones(nfeatures)
    perturbations = np.zeros((nfeatures, ntask))
    loglikes = {j: rr.glm.gaussian(predictor_vars_selection, response_selection[j], coef=1., quadratic=None) for j in
                range(ntask)}
    multi_lasso = multi_task_lasso(loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures, ntask,
                                   perturbations)
    active_signs = multi_lasso.fit(perturbations=perturbations)

    estimate = []
    CIs = [[0, 0]]  # [0,0] is a placeholder that will be discarded
    CV = []
    predictive_r = []

    # Conduct inference (estimate target and construct confidence intervals)
    if (active_signs != 0).sum() > 0:
        error = 0
        for j in range(ntask):
            X = predictor_vars_inference
            y = response_inference[j]
            observed_target = np.linalg.pinv(X[:, (active_signs[:, j] != 0)]).dot(y)
            estimate = np.concatenate([estimate, observed_target])

            Qfeat = np.linalg.inv(X[:, (active_signs[:, j] != 0)].T.dot(X[:, (active_signs[:, j] != 0)]))
            cov_target = Qfeat * dispersions[j]
            CV = np.concatenate([CV, np.sqrt(np.diag(cov_target)) / np.abs(observed_target)])

            alpha = 1. - 0.90
            quantile = ndist.ppf(1 - alpha / 2.)
            intervals = np.vstack([observed_target - quantile * np.sqrt(np.diag(cov_target)),
                                   observed_target + quantile * np.sqrt(np.diag(cov_target))]).T
            CIs = np.vstack([CIs, intervals])

            # Compute predictive r on hold-out data
            idx_new = np.sum(active_signs[:, j] != 0)
            if idx_new == 0:
                error += np.sqrt(np.sum(np.square(response_test[j])) / sample_sizes_test)
                predictive_r.append(None)
            else:
                error += np.sqrt(np.sum(np.square(
                    response_test[j] - predictor_vars_test[:, (active_signs[:, j] != 0)].dot(
                        observed_target))) / sample_sizes_test)
                predictive_r.append(
                    np.corrcoef(response_test[j], predictor_vars_test[:, (active_signs[:, j] != 0)].dot(
                        observed_target))[0, 1])

    else:
        error = 0
        for j in range(ntask):
            error += np.sqrt(np.linalg.norm(response_test[j], 2) ** 2 / sample_sizes_test)
        CIs = np.asarray([[0, 0], [np.nan, np.nan]])

    final_error = error / ntask
    final_estimates = estimate
    final_intervals = CIs[1:, ]
    final_coefs_var = CV

    # Evaluate significance (variables with confidence intervals that do not contain zero)
    significant = [final_intervals[j, 0] > 0 or final_intervals[j, 1] < 0 for j in range(np.shape(final_intervals)[0])]
    significant_variables = {}
    all_variables_ds = {}
    placeholder = 0
    for j in range(ntask):
        active_ = active_signs[:, j] != 0
        new_placeholder = np.sum(active_)
        # Retrieve predictors that are non-zero for given task. Note: np.nonzero returns tuple (non_zero_array,)
        all_variables_ds[j] = np.nonzero(active_)[0]
        significant_variables[j] = np.nonzero(active_)[0][significant[placeholder:placeholder + new_placeholder]]
        placeholder = placeholder + new_placeholder

    ds_interval_lengths = np.asarray(final_intervals[:, 1] - final_intervals[:, 0])

    return (final_estimates, final_intervals, ds_interval_lengths, all_variables_ds, significant_variables,
            final_error, predictive_r, final_coefs_var)


def rand_multi_task_tune(predictor_vars_train, predictor_vars_validate, response_train,
                         response_validate, weight_list, noise, rand_scale=0.7):

    nfeatures = predictor_vars_train.shape[1]
    sample_size = predictor_vars_train.shape[0]

    # Noise estimation and randomization variable setup
    noise_levels = []
    for j in range(ntask):
        noise_levels.append(np.sqrt(np.sum(np.array(response_train[j] - predictor_vars_train.dot(
            np.linalg.pinv(predictor_vars_train).dot(response_train[j]))) ** 2) / (sample_size - nfeatures)))
    dispersions = [noise_levels[j] ** 2 for j in range(len(noise_levels))]

    # Setup randomization variable
    # The MTL functions require the randomizer to be specified, but I will pass in the perturbation variable
    # rather than sampling from the randomizer. This allows the same noise data to be used across methods.
    # The randomizer is only used to obtain properties of the randomization variable, including the precision matrix
    ridge_terms = np.zeros(ntask)
    randomizer_scales = rand_scale * np.asarray([noise_levels[j] for j in range(ntask)])
    randomizers = {j: randomization.isotropic_gaussian((nfeatures,), randomizer_scales[j]) for j in range(ntask)}

    active_dict = {}
    pred_r_validate = {}

    # Fit MTL model and perform inference for given tuning parameter
    for weight in weight_list:
        feature_weight = weight * np.ones(nfeatures)
        perturbations = np.array(
            [randomizer_scales[j] * noise[j * nfeatures:(j + 1) * nfeatures] for j in range(ntask)]).T
        loglikes = {j: rr.glm.gaussian(predictor_vars_train, response_train[j], coef=1., quadratic=None)
                    for j in range(ntask)}
        multi_lasso = multi_task_lasso(
            loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures, ntask, perturbations)
        active_signs = multi_lasso.fit(perturbations=perturbations)
        estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(
            dispersions=dispersions)

        active_dict[weight] = active_signs
        predictive_r = []
        # Calculate predictive r on validation data
        if (active_signs != 0).sum() > 0:
            idx = 0
            for j in range(ntask):
                idx_new = np.sum(active_signs[:, j] != 0)
                if idx_new == 0:
                    predictive_r.append(None)
                else:
                    predictive_r.append(
                        np.corrcoef(response_validate[j], predictor_vars_validate[:, (active_signs[:, j] != 0)].dot(
                            estimate[idx:idx + idx_new]))[0, 1])
                idx = idx + idx_new
                print(predictive_r)

        pred_r_validate[weight] = predictive_r

    return active_dict, pred_r_validate


def rand_multi_task_selection_inference(predictor_vars_train, predictor_vars_test, response_train,
                                        response_test, weight, noise, rand_scale=0.7):

    nfeatures = predictor_vars_train.shape[1]
    sample_size = predictor_vars_train.shape[0]
    sample_size_test = predictor_vars_test.shape[0]

    # Noise estimation and randomization variable setup
    noise_levels = []
    for j in range(ntask):
        noise_levels.append(np.sqrt(np.sum(np.array(response_train[j] - predictor_vars_train.dot(
            np.linalg.pinv(predictor_vars_train).dot(response_train[j]))) ** 2) / (sample_size - nfeatures)))
    dispersions = [noise_levels[j] ** 2 for j in range(len(noise_levels))]

    # Setup randomization variable
    # The MTL functions require the randomizer to be specified, but I will pass in the perturbation variable
    # rather than sampling from the randomizer. This allows the same noise data to be used across methods.
    # The randomizer is only used to obtain properties of the randomization variable, including the precision matrix
    randomizer_scales = rand_scale * np.asarray([noise_levels[j] for j in range(ntask)])
    randomizers = {j: randomization.isotropic_gaussian((nfeatures,), randomizer_scales[j]) for j in range(ntask)}
    ridge_terms = np.zeros(ntask)

    # Fit MTL model and perform inference
    feature_weight = weight * np.ones(nfeatures)
    perturbations = np.array([randomizer_scales[j] * noise[j * nfeatures:(j + 1) * nfeatures] for j in range(ntask)]).T
    loglikes = {j: rr.glm.gaussian(predictor_vars_train, response_train[j], coef=1., quadratic=None)
                for j in range(ntask)}
    multi_lasso = multi_task_lasso(
        loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures, ntask, perturbations)
    active_signs = multi_lasso.fit(perturbations=perturbations)
    estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(
        dispersions=dispersions)
    coefs_var = np.sqrt(np.diag(observed_info_mean)) / np.abs(estimate)

    # Calculate final testing error and predictive r on test set
    if (active_signs != 0).sum() > 0:
        final_error = 0
        predictive_r = []
        idx = 0
        for j in range(ntask):
            idx_new = np.sum(active_signs[:, j] != 0)
            if idx_new == 0:
                final_error += np.sqrt(np.sum(np.square(response_test[j])) / sample_size_test)
                predictive_r.append(None)
            else:
                final_error += np.sqrt(np.sum(
                    np.square((response_test[j] - predictor_vars_test[:, (active_signs[:, j] != 0)].dot(
                        estimate[idx:idx + idx_new])))) / sample_size_test)
                predictive_r.append(np.corrcoef(response_test[j], predictor_vars_test[:, (active_signs[:, j] != 0)].dot(
                    estimate[idx:idx + idx_new]))[0, 1])
            idx = idx + idx_new

    else:
        final_error = 0
        predictive_r = []
        for j in range(ntask):
            final_error += np.sqrt(np.linalg.norm(response_test[j], 2) ** 2 / sample_size_test)

    # Average final testing error by task
    final_avg_error = final_error / ntask

    # Identify intervals that do not cover zero
    all_variables = {}
    significant = [intervals[j, 0] > 0 or intervals[j, 1] < 0 for j in range(np.shape(intervals)[0])]
    significant_variables = {}
    placeholder = 0
    for j in range(ntask):
        # Identify variables (by task) corresponding to the significant intervals
        active_ = active_signs[:, j] != 0
        new_placeholder = np.sum(active_)
        # Retrieve predictors that are non-zero for given task. Note: np.nonzero returns tuple (non_zero_array,)
        all_variables[j] = np.nonzero(active_)[0]
        significant_variables[j] = np.nonzero(active_)[0][significant[placeholder:placeholder + new_placeholder]]
        placeholder = placeholder + new_placeholder

    selective_interval_lengths = np.asarray(intervals[:, 1] - intervals[:, 0])

    return (estimate, intervals, selective_interval_lengths, all_variables, significant_variables,
            final_avg_error, predictive_r, coefs_var)


def common_format(ax):
    ax.grid(True, which='both', color='#f0f0f0')
    return ax


# Data Splitting 67/33
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

colors = ['red', 'green', 'orange', 'blue']
markers = ['|', '*', 'd', '.']
for i in range(ntask):
    print([np.sum(active[x] != 0) for x in weight_list])
    plt.plot([np.sum(active[x] != 0) / ntask for x in weight_list], [pred_r_validate[x][i] for x in weight_list],
             color=colors[i], marker=markers[i], markersize=5)
    plt.ylim(0.15, 0.4)
plt.savefig("data_splitting_test_67_30.png")
plt.clf()

final_weight = np.argmax([np.sum(pred_r_validate[x])/ntask for x in weight_list])

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

# Data Splitting 50/50
sample_sizes = predictors_train.shape[0]
samples = np.arange(int(sample_sizes))
selection = np.random.choice(samples, size=int(0.5 * sample_sizes), replace=False)
inference = np.setdiff1d(samples, selection)

responses_selection = {j: responses_train[j][selection] for j in range(ntask)}
predictors_selection = predictors_train[selection, :]
responses_inference = {j: responses_train[j][inference] for j in range(ntask)}
predictors_inference = predictors_train[inference, :]

weight_list = np.linspace(0.66, 2.5, 10)
active, pred_r_validate = ds_multi_task_tune(predictors_selection, predictors_inference, predictors_validate,
                                             responses_selection, responses_inference, responses_validate, weight_list)

colors = ['red', 'green', 'orange', 'blue']
markers = ['|', '*', 'd', '.']
for i in range(ntask):
    print([np.sum(active[x] != 0) for x in weight_list])
    plt.plot([np.sum(active[x] != 0) / ntask for x in weight_list], [pred_r_validate[x][i] for x in weight_list],
             color=colors[i], marker=markers[i], markersize=5)
    plt.ylim(0.15, 0.4)
plt.savefig("data_splitting_test_50_50.png")

final_weight = np.argmax([np.sum(pred_r_validate[x])/ntask for x in weight_list])

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
print(len(ds50_interval_lengths), "Number selected in total")
print(np.sum([len(significant_variables_ds50[i]) for i in range(len(significant_variables_ds50))]),
      "Sum of significant PCs in total")

weight_list = np.linspace(1.3, 4.0, num=10)
active, pred_r_validate = rand_multi_task_tune(predictors_train, predictors_validate, responses_train,
                                               responses_validate, weight_list, noise, rand_scale=0.7)

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

final_weight = np.argmax([np.sum(pred_r_validate[x])/ntask for x in weight_list])

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

# Create dictionary to store lengths by task
match_length_indx = {}
start = 0
for i in range(ntask):
    match_length_indx[i] = interval_lengths_joint[start:start + len(all_variables_joint[i])]
    start += len(all_variables_joint[i])

# Estimate coefficients in original feature space
# Since I standardized the principal components in R before doing the MTL regression (as per our procedure),
# I have scaled the regression coefficients by the singular values before projecting back to the original space
running_counter = 0
original_coef_approx = np.zeros((np.shape(V)[0], ntask))
for i in range(ntask):
    singular_values = sv[all_variables_joint[i]]
    original_coef_approx[:, i] = V[:, all_variables_joint[i]].dot(
        np.divide(final_estimates_joint[running_counter:running_counter + len(all_variables_joint[i])],
                  singular_values))
    running_counter += len(all_variables_joint[i])
np.savetxt("original_approx07.csv", original_coef_approx, delimiter=",")

match_length_indx2 = {}
start2 = 0
for i in range(ntask):
    match_length_indx2[i] = ds67_interval_lengths[start2:start2 + len(all_variables_ds67[i])]
    start2 += len(all_variables_ds67[i])

common_67 = {i: np.intersect1d(all_variables_joint[i], all_variables_ds67[i]) for i in range(ntask)}
print("Common variables between MTL + SI and DS 67/33", common_67)
common_significant_67 = {i: np.intersect1d(significant_variables_joint[i], significant_variables_ds67[i]) for i in
                         range(ntask)}
print("Common significant variables between MTL + SI and DS 67/33", common_significant_67)
common_lengths_67 = []
for i in range(ntask):
    for predictor in common_67[i]:
        ratio_length = match_length_indx2[i][np.argwhere(all_variables_ds67[i] == predictor)[0][0]] / \
                       match_length_indx[i][np.argwhere(all_variables_joint[i] == predictor)[0][0]]
        common_lengths_67.append(ratio_length)

match_length_indx2 = {}
start = 0
for i in range(ntask):
    match_length_indx2[i] = ds50_interval_lengths[start:start + len(all_variables_ds50[i])]
    start += len(all_variables_ds50[i])

common_50 = {i: np.intersect1d(all_variables_joint[i], all_variables_ds50[i]) for i in range(ntask)}
print("Common variables between MTL + SI and DS 50/50", common_50)
common_significant_50 = {i: np.intersect1d(significant_variables_joint[i], significant_variables_ds50[i]) for i in
                         range(ntask)}
print("Common significant variables between MTL + SI and DS 50/50", common_significant_50)
common_lengths_50 = []
# Compute length ratio for shared parameters
for i in range(ntask):
    for predictor in common_50[i]:
        ratio_length = match_length_indx2[i][np.argwhere(all_variables_ds50[i] == predictor)[0][0]] / \
                       match_length_indx[i][np.argwhere(all_variables_joint[i] == predictor)[0][0]]
        common_lengths_50.append(ratio_length)

# Pairwise MTL + SI
ntask = 2
task_index = [-5, -11]
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

weight_list = np.linspace(1.0, 4.0, num=10)
active, pred_r_validate = rand_multi_task_tune(predictors_train, predictors_validate, responses_train,
                                               responses_validate, weight_list,
                                               noise[:predictors_train.shape[1] * ntask], rand_scale=0.7)

colors = ['red', 'green']
markers = ['.', '*']
markersizes = [6, 5]
for i in range(ntask):
    print([np.sum(active[x] != 0) for x in weight_list])
    plt.plot([np.sum(active[x] != 0) / ntask for x in weight_list], [pred_r_validate[x][i] for x in weight_list],
             linestyle='--', color=colors[i], marker=markers[i], markersize=markersizes[i], zorder=2)

final_weight = np.argmax([np.sum(pred_r_validate[x])/ntask for x in weight_list])

final_estimates_crystallized, final_intervals_crystallized, interval_lengths_crystallized, all_variables_crystallized, \
    significant_variables_crystallized, final_err_crystallized, pred_r_crystallized, coefs_var_crystallized = \
    rand_multi_task_selection_inference(predictors_train, predictors_test, responses_train, responses_test,
                                        weight_list[final_weight], noise, rand_scale=0.7)
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

ntask = 2
task_index = [-2, -9]
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

weight_list = np.linspace(0.9, 3.25, num=10)
active, pred_r_validate = rand_multi_task_tune(predictors_train, predictors_validate, responses_train,
                                               responses_validate, weight_list,
                                               noise[predictors_train.shape[1] * ntask:], rand_scale=0.7)

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

final_weight = np.argmax([np.sum(pred_r_validate[x])/ntask for x in weight_list])

final_estimates_fluid, final_intervals_fluid, fluid_intervals, all_variables_fluid, significant_variables_fluid, \
    final_err_fluid, pred_r_fluid, coefs_var_fluid = rand_multi_task_selection_inference(predictors_train,
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


# Estimate coefficients in original feature space
# running_counter = 0
# separate_original_coef_approx = np.zeros((np.shape(V)[0], 4))
# for i in range(4):
#   singular_values = sv[separate_all_variables_rand07[i]]
#  separate_original_coef_approx[:, i] = V[:, separate_all_variables_rand07[i]].dot(
#     np.divide(separate_estimates_rand07[running_counter:running_counter+len(separate_all_variables_rand07[i])],
#              singular_values))
# running_counter += len(separate_all_variables_rand07[i])
# np.savetxt("separate_original_approx07.csv", separate_original_coef_approx, delimiter=",")


def set_boxplot_style(bp, color, linestyle):
    plt.setp(bp['boxes'], color=color, linestyle=linestyle, linewidth=3.5)
    plt.setp(bp['whiskers'], color=color, linestyle=linestyle, linewidth=3.5)
    plt.setp(bp['caps'], color=color, linewidth=2.5)
    plt.setp(bp['medians'], color=color, linewidth=2.5)


fig = plt.figure(figsize=(16, 5.5))
ax1 = fig.add_subplot(122)
plt.sca(ax1)
plt.boxplot([common_lengths_67], positions=[1], widths=0.4)
plt.boxplot([common_lengths_50], positions=[2], widths=0.4)
plt.xticks([1, 2], labels=['DS (0.67): MTL(0.7) + SI', 'DS(0.5): MTL(0.7) + SI'], fontsize=16)
plt.tight_layout()
plt.ylabel('Ratio of Lengths for Common Parameters', fontsize=16)
plt.yticks(fontsize=18)
# ax1.set_title("Ratio of Interval Lengths", y=1.01, fontsize=24)


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
# ax2.set_title("Distribution of Interval Lengths", y=1.01 ,fontsize=24)
common_format(ax2)
plt.savefig('real_data_lengths_cv.png', bbox_inches='tight')
