import numpy as np
from scipy.stats import t as tdist
from scipy.stats import norm as ndist
import regreg.api as rr
from selectinf.randomized.randomization import randomization
from selectinf.randomized.multitask_lasso import multi_task_lasso
np.random.seed(5)

predictor_vars_train = {}
predictor_vars_test = {}
response_train = {}
response_test = {}

X1 = np.genfromtxt('task1.csv', delimiter=',')[1:,:-1]
Y1 = np.genfromtxt('task1.csv', delimiter=',')[1:,-1]
samples1 = np.arange(np.int(np.shape(X1)[0]))
train1 = np.random.choice(samples1, size=np.int(0.9*np.shape(X1)[0]), replace=False)
test1 = np.setdiff1d(samples1, train1)
predictor_vars_train[0] = X1[train1]
predictor_vars_test[0] = X1[test1]
response_train[0] = Y1[train1]
response_test[0] = Y1[test1]

X2 = np.genfromtxt('task2.csv', delimiter=',')[1:,:-1]
Y2 = np.genfromtxt('task2.csv', delimiter=',')[1:,-1]
samples2 = np.arange(np.int(np.shape(X2)[0]))
train2 = np.random.choice(samples2, size=np.int(0.9*np.shape(X2)[0]), replace=False)
test2 = np.setdiff1d(samples2, train2)
predictor_vars_train[1] = X2[train2]
predictor_vars_test[1] = X2[test2]
response_train[1] = Y2[train2]
response_test[1] = Y2[test2]

X3 = np.genfromtxt('task3.csv', delimiter=',')[1:,:-1]
Y3 = np.genfromtxt('task3.csv', delimiter=',')[1:,-1]
samples3 = np.arange(np.int(np.shape(X3)[0]))
train3 = np.random.choice(samples3, size=np.int(0.9*np.shape(X3)[0]), replace=False)
test3 = np.setdiff1d(samples3, train3)
predictor_vars_train[2] = X3[train3]
predictor_vars_test[2] = X3[test3]
response_train[2] = Y3[train3]
response_test[2] = Y3[test3]

X4 = np.genfromtxt('task4.csv', delimiter=',')[1:,:-1]
Y4 = np.genfromtxt('task4.csv', delimiter=',')[1:,-1]
samples4 = np.arange(np.int(np.shape(X4)[0]))
train4 = np.random.choice(samples4, size=np.int(0.9*np.shape(X4)[0]), replace=False)
test4 = np.setdiff1d(samples4, train4)
predictor_vars_train[3] = X4[train4]
predictor_vars_test[3] = X4[test4]
response_train[3] = Y4[train4]
response_test[3] = Y4[test4]

X5 = np.genfromtxt('task5.csv', delimiter=',')[1:,:-1]
Y5 = np.genfromtxt('task5.csv', delimiter=',')[1:,-1]
samples5 = np.arange(np.int(np.shape(X5)[0]))
train5 = np.random.choice(samples5, size=np.int(0.9*np.shape(X5)[0]), replace=False)
test5 = np.setdiff1d(samples5, train5)
predictor_vars_train[4] = X5[train5]
predictor_vars_test[4] = X5[test5]
response_train[4] = Y5[train5]
response_test[4] = Y5[test5]

X6 = np.genfromtxt('task6.csv', delimiter=',')[1:,:-1]
Y6 = np.genfromtxt('task6.csv', delimiter=',')[1:,-1]
samples6 = np.arange(np.int(np.shape(X6)[0]))
train6 = np.random.choice(samples6, size=np.int(0.8*np.shape(X6)[0]), replace=False)
test6 = np.setdiff1d(samples6, train6)
predictor_vars_train[5] = X6[train6]
predictor_vars_test[5] = X6[test6]
response_train[5] = Y6[train6]
response_test[5] = Y6[test6]

#X7 = np.genfromtxt('task7.csv', delimiter=',')[1:,:-1]
#Y7 = np.genfromtxt('task7.csv', delimiter=',')[1:,-1]
#samples7 = np.arange(np.int(np.shape(X7)[0]))
#train7 = np.random.choice(samples7, size=np.int(0.75*np.shape(X7)[0]), replace=False)
#test7 = np.setdiff1d(samples7, train7)
#predictor_vars_train[6] = X7[train7]
#predictor_vars_test[6] = X7[test7]
#response_train[6] = Y7[train1]
#response_test[6] = Y7[test1]

#X8 = np.genfromtxt('task8.csv', delimiter=',')[1:,:-1]
#Y8 = np.genfromtxt('task8.csv', delimiter=',')[1:,-1]
#samples8 = np.arange(np.int(np.shape(X8)[0]))
#train8 = np.random.choice(samples8, size=np.int(0.75*np.shape(X8)[0]), replace=False)
#test8 = np.setdiff1d(samples8, train8)
#predictor_vars_train[7] = X8[train8]
#predictor_vars_test[7] = X8[test8]
#response_train[7] = Y8[train1]
#response_test[7] = Y8[test1]

#X9 = np.genfromtxt('task9.csv', delimiter=',')[1:,:-1]
#Y9 = np.genfromtxt('task9.csv', delimiter=',')[1:,-1]
#samples9 = np.arange(np.int(np.shape(X9)[0]))
#train9 = np.random.choice(samples9, size=np.int(0.75*np.shape(X9)[0]), replace=False)
#test9 = np.setdiff1d(samples9, train9)
#predictor_vars_train[8] = X9[train9]
#predictor_vars_test[8] = X9[test9]
#response_train[8] = Y9[train1]
#response_test[8] = Y9[test1]

#X10 = np.genfromtxt('task10.csv', delimiter=',')[1:,:-1]
#Y10 = np.genfromtxt('task10.csv', delimiter=',')[1:,-1]
#samples10 = np.arange(np.int(np.shape(X10)[0]))
#train10 = np.random.choice(samples10, size=np.int(0.75*np.shape(X10)[0]), replace=False)
#test10 = np.setdiff1d(samples10, train10)
#predictor_vars_train[9] = X10[train10]
#predictor_vars_test[9] = X10[test10]
#response_train[9] = Y10[train1]
#response_test[9] = Y10[test1]

#X11 = np.genfromtxt('task11.csv', delimiter=',')[1:,:-1]
#Y11 = np.genfromtxt('task11.csv', delimiter=',')[1:,-1]
#samples11 = np.arange(np.int(np.shape(X11)[0]))
#train11 = np.random.choice(samples11, size=np.int(0.75*np.shape(X11)[0]), replace=False)
#test11 = np.setdiff1d(samples11, train11)
#predictor_vars_train[10] = X11[train11]
#predictor_vars_test[10] = X11[test11]
#response_train[10] = Y11[train1]
#response_test[10] = Y11[test1]


ntask = 5

def _noise(n, df=np.inf):
    if df == np.inf:
        return np.random.standard_normal(n)
    else:
        sd_t = np.std(tdist.rvs(df, size=50000))
    return tdist.rvs(df, size=n) / sd_t

sample_sizes = np.asarray([predictor_vars_train[i].shape[0] for i in range(ntask)])
sample_sizes_test = np.asarray([predictor_vars_test[i].shape[0] for i in range(ntask)])
#scalings = {i: predictor_vars_train[i].std(0) for i in range(ntask)}
#predictor_vars_train = {i: predictor_vars_train[i] / (scalings[i][None, :]) for i in range(ntask)}
#scalings_test = {i: predictor_vars_test[i].std(0) for i in range(ntask)}
#predictor_vars_test = {i: predictor_vars_test[i] / (scalings[i][None, :]) for i in range(ntask)}

ridge_terms = np.zeros(ntask)
nfeatures = [predictor_vars_train[i].shape[1] for i in range(ntask)]
intervals_dict = {}
error_list = []

#samples = {j:np.arange(np.int(sample_sizes[j])) for j in range(ntask)}
#selection = {j: np.random.choice(samples[j], size=np.int(0.5 * sample_sizes[j]), replace=False) for j in range(ntask)}
#inference = {j:np.setdiff1d(samples[j], selection[j]) for j in range(ntask)}
#response_selection = {j: response_train[j][selection[j]] for j in range(ntask)}
#predictor_vars_selection = {j: predictor_vars_train[j][selection[j]] for j in range(ntask)}
#response_inference = {j: response_train[j][inference[j]] for j in range(ntask)}
#predictor_vars_inference = {j: predictor_vars_train[j][inference[j]] for j in range(ntask)}
samples = {j:np.arange(np.int(sample_sizes[j])) for j in range(ntask)}
selection = {j: np.random.choice(samples[j], size=np.int(0.67 * sample_sizes[j]), replace=False) for j in range(ntask)}
inference = {j:np.setdiff1d(samples[j], selection[j]) for j in range(ntask)}
response_selection = {j: response_train[j][selection[j]] for j in range(ntask)}
predictor_vars_selection = {j: predictor_vars_train[j][selection[j]] for j in range(ntask)}
response_inference = {j: response_train[j][inference[j]] for j in range(ntask)}
predictor_vars_inference = {j: predictor_vars_train[j][inference[j]] for j in range(ntask)}


noise_levels = []
print(sample_sizes)
print(nfeatures)
for i in range(ntask):
    print(np.sum(np.array(response_train[i] - (predictor_vars_train[i]).dot(
        np.linalg.pinv((predictor_vars_train[i])).dot(response_train[i])))** 2))
    print(sample_sizes[i] - nfeatures[i])
    #noise_levels.append(np.sqrt(np.sum(np.array(response_train[i] - (predictor_vars_train[i]).dot(
        #np.linalg.pinv((predictor_vars_train[i])).dot(response_train[i])))** 2) / (sample_sizes[i] - nfeatures[i])))
    noise_levels.append(np.std(response_train[i]))
print(noise_levels)
dispersions = [noise_levels[i] ** 2 for i in range(len(noise_levels))]
randomizer_scales = 0.7 * np.asarray([noise_levels[i] for i in range(ntask)])
randomizers = {i: randomization.isotropic_gaussian((nfeatures[i],), randomizer_scales[i]) for i in range(ntask)}
perturbations = np.array([randomizer_scales[i] * _noise(nfeatures[i]) for i in range(ntask)]).T
weight_list = np.arange(2.25,8,.05)

low_error = np.inf
for weight in weight_list:
    feature_weight = weight * np.ones(nfeatures[0])
    loglikes = {
        i: rr.glm.gaussian(predictor_vars_train[i], response_train[i], coef=1., quadratic=None)
        for i in range(ntask)}
    multi_lasso = multi_task_lasso(loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures[0], ntask, perturbations)
    active_signs = multi_lasso.fit(perturbations=perturbations)
    estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(dispersions=dispersions)
    intervals_dict[weight] = intervals
    if (active_signs != 0).sum() > 0:
        error = 0
        idx = 0
        for j in range(ntask):
            idx_new = np.sum(active_signs[:, j] != 0)
            if idx_new == 0:
                error += (0.5 * np.sum(np.square(response_test[j]))) / sample_sizes_test[0]
                continue
            error += 0.5 * (np.sum(
                np.square((response_test[j] - (predictor_vars_test[j])[:, (active_signs[:, j] != 0)].dot(
                    estimate[idx:idx + idx_new]))))) / sample_sizes_test[0]
            idx = idx + idx_new

    else:
        error = 0
        for j in range(ntask):
            error += (0.5 * np.linalg.norm(response_test[j], 2) ** 2) / sample_sizes_test[0]

    error_list.append(error)
    print(error)

    if error < low_error:
        low_error = error
        final_intervals = intervals
        significant = [intervals[j, 0] > 0 or intervals[j, 1] < 0 for j in range(np.shape(intervals)[0])]
        ordered_variables = {}
        all_variables = {}
        placeholder = 0
        for i in range(ntask):
            active_ = active_signs[:, i] != 0
            all_variables[i] = np.nonzero(active_)[0]
            new_placeholder = np.sum(active_)
            ordered_variables[i] = np.nonzero(active_)[0][significant[placeholder:placeholder+new_placeholder]]
            placeholder = placeholder + new_placeholder
        variables = ordered_variables
#lambda_1se = np.argmin(np.abs(error_list - np.min(error_list+(np.std(error_list)/np.sqrt(len(error_list))))))
#print(error_list)
#print(np.argmin(error_list))
#print((np.std(error_list)/np.sqrt(len(error_list))))
#print(np.abs(error_list - np.min(error_list+np.std(error_list)/np.sqrt(len(error_list)))))
#lambda_1se = np.argmin(error_list)
#print(lambda_1se)
#print(weight_list[lambda_1se])
#final_intervals = intervals_dict[weight_list[lambda_1se]]
        #ordered_variables = {}
        #for i in range(ntask):
            #active_ = active_signs[:, i] != 0
            #ordered_variables[i] = np.nonzero(active_)[0]
        #variables = ordered_variables

print(final_intervals)
print(np.mean(final_intervals[:,1]-final_intervals[:,0]))
#print(estimate)
print(all_variables)
print(variables)

noise_levels = []
for i in range(ntask):
    #print(np.sum(np.asarray(response_selection[i] - predictor_vars_selection[i].dot(np.linalg.pinv(predictor_vars_selection[i]).dot(response_selection[i])))**2)/(np.int(0.67 * sample_sizes[i]-nfeatures[i]))**2)
    #print((np.int(0.67 * sample_sizes[i]-nfeatures[i])))
    #noise_levels.append(np.sqrt(np.sum(np.asarray(response_selection[i] - predictor_vars_selection[i].dot(np.linalg.pinv(predictor_vars_selection[i]).dot(response_selection[i])))**2)/(np.int(0.5 * sample_sizes[i]-nfeatures[i]))))
    noise_levels.append(np.std(response_selection[i]))
dispersions = [noise_levels[i]**2 for i in range(len(noise_levels))]

weight_list = np.arange(0.5,5,.05)
low_error = np.inf
for weight in weight_list:
    feature_weight = weight * np.ones(nfeatures[0])
    loglikes = {
        i: rr.glm.gaussian(predictor_vars_selection[i], response_selection[i], coef=1., quadratic=None) for i in range(ntask)}
    perturbations = np.zeros((nfeatures[0], ntask))
    multi_lasso = multi_task_lasso(loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures[0], ntask, perturbations)
    active_signs = multi_lasso.fit(perturbations=perturbations)
    CIs = [[0, 0]]
    if (active_signs != 0).sum() > 0:
        error = 0
        for i in range(ntask):
            X = predictor_vars_inference[i]
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
                error += (0.5 * np.sum(np.square(response_test[i]))) / sample_sizes_test[i]
            else:
                error += (0.5 * np.sum(np.square(
                response_test[i] - (predictor_vars_test[i])[:, (active_signs[:, i] != 0)].dot(
                    observed_target)))) / sample_sizes_test[i]
            #print(observed_target)

    else:
        error = 0
        for j in range(ntask):
            error += (0.5 * np.linalg.norm(response_test[j], 2) ** 2) / sample_sizes_test[j]
        CIs = np.asarray([[0, 0], [np.nan, np.nan]])
        #print("partial",error)

    print(error)
    if error < low_error:
        low_error = error
        final_intervals = CIs[1:, ]
        significant = [final_intervals[j, 0] > 0 or final_intervals[j, 1] < 0 for j in range(np.shape(final_intervals)[0])]
        ordered_variables = {}
        all_variables = {}
        placeholder = 0
        for i in range(ntask):
            active_ = active_signs[:, i] != 0
            all_variables[i] = np.nonzero(active_)[0]
            new_placeholder = np.sum(active_)
            ordered_variables[i] = np.nonzero(active_)[0][significant[placeholder:placeholder + new_placeholder]]
            placeholder = placeholder + new_placeholder
        variables = ordered_variables

print(final_intervals)
print(np.mean(final_intervals[:,1]-final_intervals[:,0]))
print(all_variables)
print(variables)