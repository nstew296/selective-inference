import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import t as tdist
from scipy.stats import norm as ndist
import regreg.api as rr
from selectinf.randomized.randomization import randomization
from selectinf.randomized.multitask_lasso import multi_task_lasso
np.random.seed(5)

print("hi")

response_train = {}
response_validate = {}
response_test = {}

X1 = np.genfromtxt('task1.csv', delimiter=',')[1:,:-1]
Y1 = np.genfromtxt('task1.csv', delimiter=',')[1:,-1]
print(Y1)
samples = np.arange(np.int(np.shape(X1)[0]))
train = np.random.choice(samples, size=np.int(0.8*np.shape(X1)[0]), replace=False)
validate = np.random.choice(np.setdiff1d(samples, train),size=np.int(0.1*np.shape(X1)[0]), replace=False)
test = np.setdiff1d(np.setdiff1d(samples, train),validate)
predictor_vars_train = X1[train,:]
predictor_vars_validate = X1[validate,:]
predictor_vars_test = X1[test,:]
response_train[0] = Y1[train]
response_validate[0] = Y1[validate]
response_test[0] = Y1[test]

Y2 = np.genfromtxt('task2.csv', delimiter=',')[1:,-1]
print(Y2)
response_train[1] = Y2[train]
response_validate[1] = Y2[validate]
response_test[1] = Y2[test]

Y3 = np.genfromtxt('task3.csv', delimiter=',')[1:,-1]
print(Y3)
response_train[2] = Y3[train]
response_validate[2] = Y3[validate]
response_test[2] = Y3[test]

Y4 = np.genfromtxt('task4.csv', delimiter=',')[1:,-1]
print(Y4)
response_train[3] = Y4[train]
response_validate[3] = Y4[validate]
response_test[3] = Y4[test]

Y5 = np.genfromtxt('task5.csv', delimiter=',')[1:,-1]
print(Y5)
response_train[4] = Y5[train]
response_validate[4] = Y5[validate]
response_test[4] = Y5[test]

Y6 = np.genfromtxt('task6.csv', delimiter=',')[1:,-1]
print(Y6)
response_train[5] = Y6[train]
response_validate[5] = Y6[validate]
response_test[5] = Y6[test]

Y7 = np.genfromtxt('task7.csv', delimiter=',')[1:,-1]
print(Y7)
response_train[6] = Y7[train]
response_validate[6] = Y7[validate]
response_test[6] = Y7[test]

Y8 = np.genfromtxt('task8.csv', delimiter=',')[1:,-1]
print(Y8)
response_train[7] = Y8[train]
response_validate[7] = Y8[validate]
response_test[7] = Y8[test]

Y9 = np.genfromtxt('task9.csv', delimiter=',')[1:,-1]
print(Y9)
response_train[8] = Y9[train]
response_validate[8] = Y9[validate]
response_test[8] = Y9[test]

Y10 = np.genfromtxt('task10.csv', delimiter=',')[1:,-1]
print(Y10)
response_train[9] = Y10[train]
response_validate[9] = Y10[validate]
response_test[9] = Y10[test]

Y11 = np.genfromtxt('task11.csv', delimiter=',')[1:,-1]
print(Y11)
response_train[10] = Y11[train]
response_validate[10] = Y11[validate]
response_test[10] = Y11[test]

ntask = 11

def _noise(n, df=np.inf):
    if df == np.inf:
        return np.random.standard_normal(n)
    else:
        sd_t = np.std(tdist.rvs(df, size=50000))
    return tdist.rvs(df, size=n) / sd_t

sample_sizes = predictor_vars_train.shape[0]
sample_sizes_validate = predictor_vars_validate.shape[0]
sample_sizes_test = predictor_vars_test.shape[0]
ridge_terms = np.zeros(ntask)
nfeatures = predictor_vars_train.shape[1]
estimates_dict = {}
intervals_dict = {}
active_dict = {}
error_list = []

#Post-selection intervals
noise_levels = []
for i in range(ntask):
    noise_levels.append(np.sqrt(np.sum(np.array(response_train[i] - (predictor_vars_train).dot(
        np.linalg.pinv((predictor_vars_train)).dot(response_train[i]))) ** 2) / (sample_sizes - nfeatures)))
dispersions = [noise_levels[i] ** 2 for i in range(len(noise_levels))]
randomizer_scales = 1.0 * np.asarray([noise_levels[i] for i in range(ntask)])
randomizers = {i: randomization.isotropic_gaussian((nfeatures,), randomizer_scales[i]) for i in range(ntask)}
perturbations = np.array([randomizer_scales[i] * _noise(nfeatures) for i in range(ntask)]).T
weight_list = np.arange(26,60,1.5)

#Perform inference for given tuning parameter
for weight in weight_list:
    feature_weight = weight * np.ones(nfeatures)
    loglikes = {
        i: rr.glm.gaussian(predictor_vars_train, response_train[i], coef=1., quadratic=None)
        for i in range(ntask)}
    multi_lasso = multi_task_lasso(loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures, ntask, None)
    active_signs = multi_lasso.fit(perturbations=perturbations)
    estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(dispersions=dispersions)
    estimates_dict[weight] = estimate
    intervals_dict[weight] = intervals
    active_dict[weight] = active_signs

    #Caculate error on hold out data
    if (active_signs != 0).sum() > 0:
        error = 0
        idx = 0
        for j in range(ntask):
            idx_new = np.sum(active_signs[:, j] != 0)
            if idx_new == 0:
                error += (0.5 * np.sum(np.square(response_validate[j]))) / sample_sizes_validate
            else:
                error += 0.5 * (np.sum(
                    np.square((response_validate[j] - (predictor_vars_validate)[:, (active_signs[:, j] != 0)].dot(
                        estimate[idx:idx + idx_new]))))) / sample_sizes_validate
            idx = idx + idx_new

    else:
        error = 0
        for j in range(ntask):
            error += (0.5 * np.linalg.norm(response_validate[j], 2) ** 2) / sample_sizes_validate

    error_list.append(error)
    print(error)

min_error = np.min(error_list)
se_error = np.std(error_list)/np.sqrt(len(error_list))
error_list = error_list[:np.argmin(error_list)]
lambda_1se = np.argmin(np.abs(error_list - min_error))

final_estimates = estimates_dict[weight_list[lambda_1se]]
final_intervals = intervals_dict[weight_list[lambda_1se]]

#Caculate final error on test set
if (active_dict[weight_list[lambda_1se]] != 0).sum() > 0:
    final_error = 0
    idx = 0
    for j in range(ntask):
        idx_new = np.sum(active_dict[weight_list[lambda_1se]][:, j] != 0)
        if idx_new == 0:
            final_error += (0.5 * np.sum(np.square(response_test[j]))) / sample_sizes_test
        else:
            final_error += 0.5 * (np.sum(
                np.square((response_test[j] - (predictor_vars_test)[:, (active_dict[weight_list[lambda_1se]][:, j] != 0)].dot(
                    estimates_dict[weight_list[lambda_1se]][idx:idx + idx_new]))))) / sample_sizes_test
        idx = idx + idx_new

else:
    final_error = 0
    for j in range(ntask):
        final_error += (0.5 * np.linalg.norm(response_test[j], 2) ** 2) / sample_sizes_test

significant = [final_intervals[j, 0] > 0 or final_intervals[j, 1] < 0 for j in range(np.shape(final_intervals)[0])]
ordered_variables = {}
all_variables = {}
placeholder = 0
for i in range(ntask):
    active_ = active_dict[weight_list[lambda_1se]][:, i] != 0
    new_placeholder = np.sum(active_)
    all_variables[i] = np.nonzero(active_)[0]
    ordered_variables[i] = np.nonzero(active_)[0][significant[placeholder:placeholder+new_placeholder]]
    placeholder = placeholder + new_placeholder
    variables = ordered_variables

selective1_intervals = np.asarray(final_intervals[:,1]-final_intervals[:,0])

match_length_indx = {}
start = 0
for i in range(ntask):
    match_length_indx[i] = selective1_intervals[start:start+len(all_variables[i])]
    start += len(all_variables[i])

print(final_error)
print(np.mean(final_intervals[:,1]-final_intervals[:,0]))
print(np.sd(final_intervals[:,1]-final_intervals[:,0]))
print(np.sum(significant))
print(all_variables)
print(variables)

#Data splitting

samples = np.arange(np.int(sample_sizes))
selection = np.random.choice(samples, size=np.int(0.5 * sample_sizes), replace=False)
inference = np.setdiff1d(samples, selection)
response_selection = {j: response_train[j][selection] for j in range(ntask)}
predictor_vars_selection = predictor_vars_train[selection]
response_inference = {j: response_train[j][inference] for j in range(ntask)}
predictor_vars_inference = predictor_vars_train[inference]

noise_levels = []
for i in range(ntask):
   noise_levels.append(np.sqrt(np.sum(np.asarray(response_selection[i] - predictor_vars_selection.dot(np.linalg.pinv(predictor_vars_selection).dot(response_selection[i])))**2)/(0.5*sample_sizes-nfeatures)))
dispersions = [noise_levels[i]**2 for i in range(len(noise_levels))]

weight_list = np.arange(12,37,1.5)
estimates_dict = {}
intervals_dict = {}
active_dict = {}
error_list = []

#Perform inference for given tuning parameter
for weight in weight_list:
    feature_weight = weight * np.ones(nfeatures)
    loglikes = {
        i: rr.glm.gaussian(predictor_vars_selection, response_selection[i], coef=1., quadratic=None) for i in range(ntask)}
    perturbations = np.zeros((nfeatures, ntask))
    multi_lasso = multi_task_lasso(loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures, ntask, perturbations)
    active_signs = multi_lasso.fit(perturbations=perturbations)
    CIs = [[0, 0]]
    estimate = []

    #Calculate error on holdout data
    if (active_signs != 0).sum() > 0:
        error = 0
        for i in range(ntask):
            X = predictor_vars_inference
            y = response_inference[i]
            Qfeat = np.linalg.inv(X[:, (active_signs[:, i] != 0)].T.dot(X[:, (active_signs[:, i] != 0)]))
            observed_target = np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(y)
            estimate = np.concatenate([estimate,observed_target])
            cov_target = Qfeat * dispersions[i]
            alpha = 1. - 0.90
            quantile = ndist.ppf(1 - alpha / 2.)
            intervals = np.vstack([observed_target - quantile * np.sqrt(np.diag(cov_target)),
                                   observed_target + quantile * np.sqrt(np.diag(cov_target))]).T
            CIs = np.vstack([CIs, intervals])

            idx_new = np.sum(active_signs[:, i] != 0)
            if idx_new == 0:
                error += (0.5 * np.sum(np.square(response_validate[i]))) / sample_sizes_validate
            else:
                error += (0.5 * np.sum(np.square(
                response_validate[i] - (predictor_vars_validate)[:, (active_signs[:, i] != 0)].dot(
                    observed_target)))) / sample_sizes_validate


    else:
        error = 0
        for j in range(ntask):
            error += (0.5 * np.linalg.norm(response_validate[j], 2) ** 2) / sample_sizes_validate
        CIs = np.asarray([[0, 0], [np.nan, np.nan]])

    estimates_dict[weight] = estimate
    intervals_dict[weight] = CIs
    active_dict[weight] = active_signs
    error_list.append(error)
    print(error)

min_error = np.min(error_list)
se_error = np.std(error_list) / np.sqrt(len(error_list))
error_list = error_list[:np.argmin(error_list)]
lambda_1se = np.argmin(np.abs(error_list - min_error))

final_estimates = estimates_dict[weight_list[lambda_1se]]
final_intervals = intervals_dict[weight_list[lambda_1se]][1:, ]

#Caculate final error on test set
if (active_dict[weight_list[lambda_1se]] != 0).sum() > 0:
    final_error = 0
    idx = 0
    for j in range(ntask):
        idx_new = np.sum(active_dict[weight_list[lambda_1se]][:, j] != 0)
        if idx_new == 0:
            final_error += (0.5 * np.sum(np.square(response_test[j]))) / sample_sizes_test
        else:
            final_error += 0.5 * (np.sum(
                np.square((response_test[j] - (predictor_vars_test)[:, (active_dict[weight_list[lambda_1se]][:, j] != 0)].dot(
                    estimates_dict[weight_list[lambda_1se]][idx:idx + idx_new]))))) / sample_sizes_test
        idx = idx + idx_new

else:
    final_error = 0
    for j in range(ntask):
        final_error += (0.5 * np.linalg.norm(response_test[j], 2) ** 2) / sample_sizes_test

significant = [final_intervals[j, 0] > 0 or final_intervals[j, 1] < 0 for j in range(np.shape(final_intervals)[0])]
ordered_variables = {}
all_variables_ds = {}
placeholder = 0
for i in range(ntask):
    active_ = active_dict[weight_list[lambda_1se]][:, i] != 0
    new_placeholder = np.sum(active_)
    all_variables_ds[i] = np.nonzero(active_)[0]
    ordered_variables[i] = np.nonzero(active_)[0][significant[placeholder:placeholder+new_placeholder]]
    placeholder = placeholder + new_placeholder
    variables = ordered_variables

ds50_intervals = np.asarray(final_intervals[:,1]-final_intervals[:,0])

print(final_error)
print(np.mean(final_intervals[:,1]-final_intervals[:,0]))
print(np.sd(final_intervals[:,1]-final_intervals[:,0]))
print(np.sum(significant))
print(all_variables_ds)
print(variables)

match_length_indx2 = {}
start2 = 0
for i in range(ntask):
    match_length_indx2[i] = ds50_intervals[start2:start2+len(all_variables_ds[i])]
    start2 += len(all_variables_ds[i])

common = {i:np.intersect1d(all_variables[i],all_variables_ds[i]) for i in range(ntask)}
print("common",common)
common_lengths = []
for i in range(ntask):
    for predictor in common[i]:
        diff_length = match_length_indx2[i][np.argwhere(all_variables_ds[i]==predictor)[0][0]]-match_length_indx[i][np.argwhere(all_variables[i]==predictor)[0][0]]
        common_lengths.append(diff_length)
print(common_lengths)
fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(common_lengths)
plt.savefig('real_data_lengths2.png', bbox_inches='tight')

sample_sizes = predictor_vars_train.shape[0]
sample_sizes_validate = predictor_vars_validate.shape[0]
sample_sizes_test = predictor_vars_test.shape[0]
ridge_terms = np.zeros(ntask)
nfeatures = predictor_vars_train.shape[1]
estimates_dict = {}
intervals_dict = {}
active_dict = {}
error_list = []

#Post-selection intervals
noise_levels = []
for i in range(ntask):
    noise_levels.append(np.sqrt(np.sum(np.array(response_train[i] - (predictor_vars_train).dot(
        np.linalg.pinv((predictor_vars_train)).dot(response_train[i]))) ** 2) / (sample_sizes - nfeatures)))
dispersions = [noise_levels[i] ** 2 for i in range(len(noise_levels))]
randomizer_scales = 0.7 * np.asarray([noise_levels[i] for i in range(ntask)])
randomizers = {i: randomization.isotropic_gaussian((nfeatures,), randomizer_scales[i]) for i in range(ntask)}
perturbations = np.array([randomizer_scales[i] * _noise(nfeatures) for i in range(ntask)]).T
weight_list = np.arange(30,57,1.5)

#Perform inference for given tuning parameter
for weight in weight_list:
    feature_weight = weight * np.ones(nfeatures)
    loglikes = {
        i: rr.glm.gaussian(predictor_vars_train, response_train[i], coef=1., quadratic=None)
        for i in range(ntask)}
    multi_lasso = multi_task_lasso(loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures, ntask, None)
    active_signs = multi_lasso.fit(perturbations=perturbations)
    estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(dispersions=dispersions)
    estimates_dict[weight] = estimate
    intervals_dict[weight] = intervals
    active_dict[weight] = active_signs

    #Caculate error on hold out data
    if (active_signs != 0).sum() > 0:
        error = 0
        idx = 0
        for j in range(ntask):
            idx_new = np.sum(active_signs[:, j] != 0)
            if idx_new == 0:
                error += (0.5 * np.sum(np.square(response_validate[j]))) / sample_sizes_validate
            else:
                error += 0.5 * (np.sum(
                    np.square((response_validate[j] - (predictor_vars_validate)[:, (active_signs[:, j] != 0)].dot(
                        estimate[idx:idx + idx_new]))))) / sample_sizes_validate
            idx = idx + idx_new

    else:
        error = 0
        for j in range(ntask):
            error += (0.5 * np.linalg.norm(response_validate[j], 2) ** 2) / sample_sizes_validate

    error_list.append(error)
    print(error)

min_error = np.min(error_list)
se_error = np.std(error_list)/np.sqrt(len(error_list))
error_list = error_list[:np.argmin(error_list)]
lambda_1se = np.argmin(np.abs(error_list - min_error))

final_estimates = estimates_dict[weight_list[lambda_1se]]
final_intervals = intervals_dict[weight_list[lambda_1se]]

#Caculate final error on test set
if (active_dict[weight_list[lambda_1se]] != 0).sum() > 0:
    final_error = 0
    idx = 0
    for j in range(ntask):
        idx_new = np.sum(active_dict[weight_list[lambda_1se]][:, j] != 0)
        if idx_new == 0:
            final_error += (0.5 * np.sum(np.square(response_test[j]))) / sample_sizes_test
        else:
            final_error += 0.5 * (np.sum(
                np.square((response_test[j] - (predictor_vars_test)[:, (active_dict[weight_list[lambda_1se]][:, j] != 0)].dot(
                    estimates_dict[weight_list[lambda_1se]][idx:idx + idx_new]))))) / sample_sizes_test
        idx = idx + idx_new

else:
    final_error = 0
    for j in range(ntask):
        final_error += (0.5 * np.linalg.norm(response_test[j], 2) ** 2) / sample_sizes_test

significant = [final_intervals[j, 0] > 0 or final_intervals[j, 1] < 0 for j in range(np.shape(final_intervals)[0])]
ordered_variables = {}
all_variables = {}
placeholder = 0
for i in range(ntask):
    active_ = active_dict[weight_list[lambda_1se]][:, i] != 0
    new_placeholder = np.sum(active_)
    all_variables[i] = np.nonzero(active_)[0]
    ordered_variables[i] = np.nonzero(active_)[0][significant[placeholder:placeholder+new_placeholder]]
    placeholder = placeholder + new_placeholder
    variables = ordered_variables

selective07_intervals = np.asarray(final_intervals[:,1]-final_intervals[:,0])

print(final_error)
print(np.mean(final_intervals[:,1]-final_intervals[:,0]))
print(np.sd(final_intervals[:,1]-final_intervals[:,0]))
print(np.sum(significant))
print(all_variables)
print(variables)

match_length_indx = {}
start = 0
for i in range(ntask):
    match_length_indx[i] = selective07_intervals[start:start+len(all_variables[i])]
    start += len(all_variables[i])

#Data splitting

samples = np.arange(np.int(sample_sizes))
selection = np.random.choice(samples, size=np.int(0.67 * sample_sizes), replace=False)
inference = np.setdiff1d(samples, selection)
response_selection = {j: response_train[j][selection] for j in range(ntask)}
predictor_vars_selection = predictor_vars_train[selection]
response_inference = {j: response_train[j][inference] for j in range(ntask)}
predictor_vars_inference = predictor_vars_train[inference]

noise_levels = []
for i in range(ntask):
   noise_levels.append(np.sqrt(np.sum(np.asarray(response_selection[i] - predictor_vars_selection.dot(np.linalg.pinv(predictor_vars_selection).dot(response_selection[i])))**2)/(0.67*sample_sizes-nfeatures)))
dispersions = [noise_levels[i]**2 for i in range(len(noise_levels))]

weight_list = np.arange(20,45,1.5)
estimates_dict = {}
intervals_dict = {}
active_dict = {}
error_list = []

#Perform inference for given tuning parameter
for weight in weight_list:
    feature_weight = weight * np.ones(nfeatures)
    loglikes = {
        i: rr.glm.gaussian(predictor_vars_selection, response_selection[i], coef=1., quadratic=None) for i in range(ntask)}
    perturbations = np.zeros((nfeatures, ntask))
    multi_lasso = multi_task_lasso(loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures, ntask, perturbations)
    active_signs = multi_lasso.fit(perturbations=perturbations)
    CIs = [[0, 0]]
    estimate = []

    #Calculate error on holdout data
    if (active_signs != 0).sum() > 0:
        error = 0
        for i in range(ntask):
            X = predictor_vars_inference
            y = response_inference[i]
            Qfeat = np.linalg.inv(X[:, (active_signs[:, i] != 0)].T.dot(X[:, (active_signs[:, i] != 0)]))
            observed_target = np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(y)
            estimate = np.concatenate([estimate,observed_target])
            cov_target = Qfeat * dispersions[i]
            alpha = 1. - 0.90
            quantile = ndist.ppf(1 - alpha / 2.)
            intervals = np.vstack([observed_target - quantile * np.sqrt(np.diag(cov_target)),
                                   observed_target + quantile * np.sqrt(np.diag(cov_target))]).T
            CIs = np.vstack([CIs, intervals])

            idx_new = np.sum(active_signs[:, i] != 0)
            if idx_new == 0:
                error += (0.5 * np.sum(np.square(response_validate[i]))) / sample_sizes_validate
            else:
                error += (0.5 * np.sum(np.square(
                response_validate[i] - (predictor_vars_validate)[:, (active_signs[:, i] != 0)].dot(
                    observed_target)))) / sample_sizes_validate


    else:
        error = 0
        for j in range(ntask):
            error += (0.5 * np.linalg.norm(response_validate[j], 2) ** 2) / sample_sizes_validate
        CIs = np.asarray([[0, 0], [np.nan, np.nan]])

    estimates_dict[weight] = estimate
    intervals_dict[weight] = CIs
    active_dict[weight] = active_signs
    error_list.append(error)
    print(error)

min_error = np.min(error_list)
se_error = np.std(error_list) / np.sqrt(len(error_list))
error_list = error_list[:np.argmin(error_list)]
lambda_1se = np.argmin(np.abs(error_list - min_error))

final_estimates = estimates_dict[weight_list[lambda_1se]]
final_intervals = intervals_dict[weight_list[lambda_1se]][1:, ]

#Caculate final error on test set
if (active_dict[weight_list[lambda_1se]] != 0).sum() > 0:
    final_error = 0
    idx = 0
    for j in range(ntask):
        idx_new = np.sum(active_dict[weight_list[lambda_1se]][:, j] != 0)
        if idx_new == 0:
            final_error += (0.5 * np.sum(np.square(response_test[j]))) / sample_sizes_test
        else:
            final_error += 0.5 * (np.sum(
                np.square((response_test[j] - (predictor_vars_test)[:, (active_dict[weight_list[lambda_1se]][:, j] != 0)].dot(
                    estimates_dict[weight_list[lambda_1se]][idx:idx + idx_new]))))) / sample_sizes_test
        idx = idx + idx_new

else:
    final_error = 0
    for j in range(ntask):
        final_error += (0.5 * np.linalg.norm(response_test[j], 2) ** 2) / sample_sizes_test

significant = [final_intervals[j, 0] > 0 or final_intervals[j, 1] < 0 for j in range(np.shape(final_intervals)[0])]
ordered_variables = {}
all_variables_ds = {}
placeholder = 0
for i in range(ntask):
    active_ = active_dict[weight_list[lambda_1se]][:, i] != 0
    new_placeholder = np.sum(active_)
    all_variables_ds[i] = np.nonzero(active_)[0]
    ordered_variables[i] = np.nonzero(active_)[0][significant[placeholder:placeholder+new_placeholder]]
    placeholder = placeholder + new_placeholder
    variables = ordered_variables

ds67_intervals = np.asarray(final_intervals[:,1]-final_intervals[:,0])

print(final_error)
print(np.mean(final_intervals[:,1]-final_intervals[:,0]))
print(np.sd(final_intervals[:,1]-final_intervals[:,0]))
print(np.sum(significant))
print(all_variables)
print(variables)

match_length_indx2 = {}
start2 = 0
for i in range(ntask):
    match_length_indx2[i] = ds67_intervals[start2:start2+len(all_variables_ds[i])]
    start2 += len(all_variables_ds[i])

common = {i:np.intersect1d(all_variables[i],all_variables_ds[i]) for i in range(ntask)}
print("common",common)
common_lengths_67 = []
for i in range(ntask):
    for predictor in common[i]:
        diff_length = match_length_indx2[i][np.argwhere(all_variables_ds[i]==predictor)[0][0]]-match_length_indx[i][np.argwhere(all_variables[i]==predictor)[0][0]]
        common_lengths_67.append(diff_length)

def set_box_color(bp, color, linestyle):
    plt.setp(bp['boxes'], color=color, linestyle=linestyle, linewidth=2.5)
    plt.setp(bp['whiskers'], color=color, linestyle=linestyle, linewidth=2.5)
    plt.setp(bp['caps'], color=color, linewidth=2.5)
    plt.setp(bp['medians'], color=color, linewidth=2.5)

fig = plt.figure(figsize=(17, 14))
ax1 = fig.add_subplot(111)

plt.sca(ax1)
first = plt.boxplot([common_lengths_67], positions=np.asarray([1]), sym='', widths=0.3)
second = plt.boxplot([common_lengths], positions=np.asarray([1.6]), sym='', widths=0.3)
set_box_color(first, '#2c7fb8', 'solid')  # colors are from http://colorbrewer2.org/
set_box_color(second, '#2c7fb8', '--')
plt.xlim(0.7, 1.9)
plt.tight_layout()
plt.plot([], c='#2c7fb8', label='57/33 Split')
plt.plot([], c='#2c7fb8', label='50/50 Split', linestyle='--', linewidth=2.5)
plt.legend()
plt.ylabel('Difference in Interval Length', fontsize=20)

ax1.set_title("Difference in Confidence Interval Length between Data Splitting and Selective Inference", y=1.01 ,fontsize=24)
ax1.legend(loc='lower left', bbox_to_anchor=(0.41, -0.125), fontsize=20)
ax1.set_xticklabels([])
ax1.set_xticks([])


def common_format(ax):
    ax.grid(True, which='both', color='#f0f0f0')
    ax.set_xlabel('Method', fontsize=20)
    return ax

common_format(ax1)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('real_data_lengths2.png', bbox_inches='tight')


fig = plt.figure(figsize=(17, 14))
ax1 = fig.add_subplot(111)

plt.sca(ax1)
first = plt.boxplot([selective07_intervals], positions=np.asarray([1]), sym='', widths=0.3)
second = plt.boxplot([selective1_intervals], positions=np.asarray([1.3]), sym='', widths=0.3)
fourth = plt.boxplot([ds67_intervals], positions=np.asarray([1.8]), sym='', widths=0.3)
fifth = plt.boxplot([ds50_intervals], positions=np.asarray([2.1]), sym='', widths=0.3)
set_box_color(first, '#2b8cbe', 'solid')  # colors are from http://colorbrewer2.org/
set_box_color(second, '#6baed6', '--')
set_box_color(fourth, '#238443', 'solid')
set_box_color(fifth, '#31a354', '--')
plt.xlim(0.7, 2.4)
plt.tight_layout()
plt.plot([], c='#2b8cbe', label='Randomized Multi-Task Lasso 0.7')
plt.plot([], c='#6baed6', label='Randomized Multi-Task Lasso 1.0', linestyle='--', linewidth=2.5)
plt.plot([], c='#238443', label='Data Splitting 67/33', linewidth=2.5)
plt.plot([], c='#31a354', label='Data Splitting 50/50', linestyle='--', linewidth=2.5)
plt.legend()
plt.ylabel('Interval Length', fontsize=20)

ax1.set_title("Distribution of Interval Lengths", y=1.01 ,fontsize=24)
ax1.legend(loc='lower left', bbox_to_anchor=(0.319, -0.225), fontsize=20)
ax1.set_xticklabels([])
ax1.set_xticks([])


def common_format(ax):
    ax.grid(True, which='both', color='#f0f0f0')
    ax.set_xlabel('Method', fontsize=20)
    return ax

common_format(ax1)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('real_data_lengths.png', bbox_inches='tight')

