import numpy as np
from scipy.stats import t as tdist
from scipy.stats import norm as ndist
import regreg.api as rr
from selectinf.randomized.randomization import randomization
from selectinf.randomized.multitask_lasso import multi_task_lasso
np.random.seed(5)

response_train = {}
response_test = {}

X1 = np.genfromtxt('task1.csv', delimiter=',')[1:,:-1]
Y1 = np.genfromtxt('task1.csv', delimiter=',')[1:,-1]
samples = np.arange(np.int(np.shape(X1)[0]))
train = np.random.choice(samples, size=np.int(0.8*np.shape(X1)[0]), replace=False)
test = np.setdiff1d(samples, train)
predictor_vars_train = X1[train]
predictor_vars_test = X1[test]
response_train[0] = Y1[train]
response_test[0] = Y1[test]

Y2 = np.genfromtxt('task2.csv', delimiter=',')[1:,-1]
response_train[1] = Y2[train]
response_test[1] = Y2[test]

Y3 = np.genfromtxt('task3.csv', delimiter=',')[1:,-1]
response_train[2] = Y3[train]
response_test[2] = Y3[test]

Y4 = np.genfromtxt('task4.csv', delimiter=',')[1:,-1]
response_train[3] = Y4[train]
response_test[3] = Y4[test]

Y5 = np.genfromtxt('task5.csv', delimiter=',')[1:,-1]
response_train[4] = Y5[train]
response_test[4] = Y5[test]

Y6 = np.genfromtxt('task6.csv', delimiter=',')[1:,-1]
response_train[5] = Y6[train]
response_test[5] = Y6[test]

Y7 = np.genfromtxt('task7.csv', delimiter=',')[1:,-1]
response_train[6] = Y7[train]
response_test[6] = Y7[test]

Y8 = np.genfromtxt('task8.csv', delimiter=',')[1:,-1]
response_train[7] = Y8[train]
response_test[7] = Y8[test]

Y9 = np.genfromtxt('task9.csv', delimiter=',')[1:,-1]
response_train[8] = Y9[train]
response_test[8] = Y9[test]

Y10 = np.genfromtxt('task10.csv', delimiter=',')[1:,-1]
response_train[9] = Y10[train]
response_test[9] = Y10[test]

Y11 = np.genfromtxt('task11.csv', delimiter=',')[1:,-1]
response_train[10] = Y11[train]
response_test[10] = Y11[test]

ntask = 11

def _noise(n, df=np.inf):
    if df == np.inf:
        return np.random.standard_normal(n)
    else:
        sd_t = np.std(tdist.rvs(df, size=50000))
    return tdist.rvs(df, size=n) / sd_t

sample_sizes = predictor_vars_train.shape[0]
sample_sizes_test = predictor_vars_test.shape[0]
ridge_terms = np.zeros(ntask)
nfeatures = predictor_vars_train.shape[1]
intervals_dict = {}
error_list = []

#Post-selection intervals
noise_levels = []
for i in range(ntask):
    noise_levels.append(np.sqrt(np.mean(np.array(response_train[i] - predictor_vars_train.dot(np.linalg.pinv(predictor_vars_train).dot(response_train[i])))**2)))
print(noise_levels)
dispersions = [noise_levels[i]**2 for i in range(len(noise_levels))]
randomizer_scales = 0.7 * np.asarray(noise_levels)
randomizers = {i: randomization.isotropic_gaussian((nfeatures,), randomizer_scales[i]) for i in range(ntask)}

weight_list = np.arange(20,35,1)
low_error = np.inf
#Perform inference for given tuning parameter
for weight in weight_list:
    feature_weight = weight * np.ones(nfeatures)
    loglikes = {
        i: rr.glm.gaussian(predictor_vars_train, response_train[i], coef=1., quadratic=None)
        for i in range(ntask)}
    perturbations = np.array([randomizer_scales[i] * _noise(nfeatures) for i in range(ntask)]).T
    multi_lasso = multi_task_lasso(loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures, ntask, perturbations)
    active_signs = multi_lasso.fit(perturbations=perturbations)
    estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(dispersions=dispersions)
    intervals_dict[weight] = intervals

    #Caculate error on hold out data
    if (active_signs != 0).sum() > 0:
        error = 0
        idx = 0
        for j in range(ntask):
            idx_new = np.sum(active_signs[:, j] != 0)
            if idx_new == 0:
                error += (0.5 * np.sum(np.square(response_test[j]))) / sample_sizes_test
                continue
            error += 0.5 * (np.sum(
                np.square((response_test[j] - (predictor_vars_test)[:, (active_signs[:, j] != 0)].dot(
                    estimate[idx:idx + idx_new]))))) / sample_sizes_test
            idx = idx + idx_new

    else:
        error = 0
        for j in range(ntask):
            error += (0.5 * np.linalg.norm(response_test[j], 2) ** 2) / sample_sizes_test

    error_list.append(error)
    print(error)

    #Track intervals and selected predictors for best model
    if error < low_error:
        low_error = error
        final_intervals = intervals
        significant = [intervals[j, 0] > 0 or intervals[j, 1] < 0 for j in range(np.shape(intervals)[0])]
        ordered_variables = {}
        placeholder = 0
        for i in range(ntask):
            active_ = active_signs[:, i] != 0
            new_placeholder = np.sum(active_)
            ordered_variables[i] = np.nonzero(active_)[0][significant[placeholder:placeholder+new_placeholder]]
            placeholder = placeholder + new_placeholder
        variables = ordered_variables

print(final_intervals)
print(np.mean(final_intervals[:,1]-final_intervals[:,0]))
print(variables)

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
    noise_levels.append(np.sqrt(np.mean(np.asarray(response_selection[i] - predictor_vars_selection.dot(np.linalg.pinv(predictor_vars_selection).dot(response_selection[i])))**2)))
dispersions = [noise_levels[i]**2 for i in range(len(noise_levels))]

weight_list = np.arange(5,18,1)
low_error = np.inf
#Perform inference for given tuning parameter
for weight in weight_list:
    feature_weight = weight * np.ones(nfeatures)
    loglikes = {
        i: rr.glm.gaussian(predictor_vars_selection, response_selection[i], coef=1., quadratic=None) for i in range(ntask)}
    perturbations = np.zeros((nfeatures, ntask))
    multi_lasso = multi_task_lasso(loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures, ntask, perturbations)
    active_signs = multi_lasso.fit(perturbations=perturbations)
    CIs = [[0, 0]]

    #Calculate error on holdout data
    if (active_signs != 0).sum() > 0:
        error = 0
        for i in range(ntask):
            X = predictor_vars_inference
            y = response_inference[i]
            Qfeat = np.linalg.inv(X[:, (active_signs[:, i] != 0)].T.dot(X[:, (active_signs[:, i] != 0)]))
            observed_target = np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(y)
            cov_target = Qfeat * dispersions[i]
            alpha = 1. - 0.90
            quantile = ndist.ppf(1 - alpha / 2.)
            intervals = np.vstack([observed_target - quantile * np.sqrt(np.diag(cov_target)),
                                   observed_target + quantile * np.sqrt(np.diag(cov_target))]).T
            CIs = np.vstack([CIs, intervals])

            idx_new = np.sum(active_signs[:, i] != 0)
            if idx_new == 0:
                error += (0.5 * np.sum(np.square(response_test[i]))) / sample_sizes_test
            else:
                error += (0.5 * np.sum(np.square(
                response_test[i] - (predictor_vars_test)[:, (active_signs[:, i] != 0)].dot(
                    observed_target)))) / sample_sizes_test
            #print(observed_target)

    else:
        error = 0
        for j in range(ntask):
            error += (0.5 * np.linalg.norm(response_test[j], 2) ** 2) / sample_sizes_test
        CIs = np.asarray([[0, 0], [np.nan, np.nan]])
        #print("partial",error)

    print(error)
    #Track intervals and selected predictors for best model
    if error < low_error:
        low_error = error
        final_intervals = CIs[1:, ]
        significant = [final_intervals[j, 0] > 0 or final_intervals[j, 1] < 0 for j in range(np.shape(final_intervals)[0])]
        ordered_variables = {}
        placeholder = 0
        for i in range(ntask):
            active_ = active_signs[:, i] != 0
            new_placeholder = np.sum(active_)
            ordered_variables[i] = np.nonzero(active_)[0][significant[placeholder:placeholder + new_placeholder]]
            placeholder = placeholder + new_placeholder
        variables = ordered_variables

print(final_intervals)
print(np.mean(final_intervals[:,1]-final_intervals[:,0]))
print(variables)