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

responses_train = {}
responses_validate = {}
responses_test = {}

X = np.genfromtxt('task1.csv', delimiter=',')[1:,:-1]
Y1 = np.genfromtxt('task1.csv', delimiter=',')[1:,-1]

samples = np.arange(np.int(np.shape(X)[0]))
train = np.random.choice(samples, size=np.int(0.8*np.shape(X)[0]), replace=False)
validate = np.random.choice(np.setdiff1d(samples, train),size=np.int(0.1*np.shape(X)[0]), replace=False)
test = np.setdiff1d(np.setdiff1d(samples, train),validate)
print(np.intersect1d(train,validate))
print(np.intersect1d(train,test))
predictors_train = X[train,:]
predictors_validate = X[validate,:]
predictors_test = X[test,:]
responses_train[0] = Y1[train]
responses_validate[0] = Y1[validate]
responses_test[0] = Y1[test]

Y2 = np.genfromtxt('task2.csv', delimiter=',')[1:,-1]
responses_train[1] = Y2[train]
responses_validate[1] = Y2[validate]
responses_test[1] = Y2[test]

Y3 = np.genfromtxt('task3.csv', delimiter=',')[1:,-1]
print(Y3)
responses_train[2] = Y3[train]
responses_validate[2] = Y3[validate]
responses_test[2] = Y3[test]

Y4 = np.genfromtxt('task4.csv', delimiter=',')[1:,-1]
responses_train[3] = Y4[train]
responses_validate[3] = Y4[validate]
responses_test[3] = Y4[test]

Y5 = np.genfromtxt('task5.csv', delimiter=',')[1:,-1]
responses_train[4] = Y5[train]
responses_validate[4] = Y5[validate]
responses_test[4] = Y5[test]

Y6 = np.genfromtxt('task6.csv', delimiter=',')[1:,-1]
responses_train[5] = Y6[train]
responses_validate[5] = Y6[validate]
responses_test[5] = Y6[test]

Y7 = np.genfromtxt('task7.csv', delimiter=',')[1:,-1]
responses_train[6] = Y7[train]
responses_validate[6] = Y7[validate]
responses_test[6] = Y7[test]

Y8 = np.genfromtxt('task8.csv', delimiter=',')[1:,-1]
responses_train[7] = Y8[train]
responses_validate[7] = Y8[validate]
responses_test[7] = Y8[test]

Y9 = np.genfromtxt('task9.csv', delimiter=',')[1:,-1]
responses_train[8] = Y9[train]
responses_validate[8] = Y9[validate]
responses_test[8] = Y9[test]

Y10 = np.genfromtxt('task10.csv', delimiter=',')[1:,-1]
responses_train[9] = Y10[train]
responses_validate[9] = Y10[validate]
responses_test[9] = Y10[test]

Y11 = np.genfromtxt('task11.csv', delimiter=',')[1:,-1]
responses_train[10] = Y11[train]
responses_validate[10] = Y11[validate]
responses_test[10] = Y11[test]

glavaan = np.genfromtxt('general_g.csv', delimiter=',')[1:]

ntask = 11

def _noise(n, df=np.inf):
    if df == np.inf:
        return np.random.standard_normal(n)
    else:
        sd_t = np.std(tdist.rvs(df, size=50000))
    return tdist.rvs(df, size=n) / sd_t

def rand_multi_task_selection_inference(predictor_vars_train,predictor_vars_validate,predictor_vars_test,response_train,
                                        response_validate,response_test,weight_list,rand_scale=0.7):

    sample_sizes = predictor_vars_train.shape[0]
    sample_sizes_validate = predictor_vars_validate.shape[0]
    sample_sizes_test = predictor_vars_test.shape[0]
    ridge_terms = np.zeros(ntask)
    nfeatures = predictor_vars_train.shape[1]
    estimates_dict = {}
    coef_var_dict = {}
    intervals_dict = {}
    active_dict = {}
    error_list = []

    #Post-selection intervals
    noise_levels = []
    for i in range(ntask):
        noise_levels.append(np.sqrt(np.sum(np.array(response_train[i] - (predictor_vars_train).dot(
            np.linalg.pinv((predictor_vars_train)).dot(response_train[i]))) ** 2) / (sample_sizes - nfeatures -1)))
    dispersions = [noise_levels[i] ** 2 for i in range(len(noise_levels))]
    randomizer_scales = rand_scale * np.asarray([noise_levels[i] for i in range(ntask)])
    randomizers = {i: randomization.isotropic_gaussian((nfeatures,), randomizer_scales[i]) for i in range(ntask)}
    perturbations = np.array([randomizer_scales[i] * _noise(nfeatures) for i in range(ntask)]).T

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
        coef_var_dict[weight] = np.sqrt(np.diag(observed_info_mean)) / np.abs(estimate)
        intervals_dict[weight] = intervals
        active_dict[weight] = active_signs

        #Caculate error on hold out data
        if (active_signs != 0).sum() > 0:
            error = 0
            idx = 0
            for j in range(ntask):
                idx_new = np.sum(active_signs[:, j] != 0)
                if idx_new == 0:
                    error += np.sqrt(np.sum(np.square(response_validate[j])) / sample_sizes_validate)
                else:
                    error += np.sqrt(np.sum(
                        np.square((response_validate[j] - (predictor_vars_validate)[:, (active_signs[:, j] != 0)].dot(
                            estimate[idx:idx + idx_new])))) / sample_sizes_validate)
                idx = idx + idx_new

        else:
            error = 0
            for j in range(ntask):
                error += np.qrt((np.linalg.norm(response_validate[j], 2) ** 2) / sample_sizes_validate)

        error_list.append(error/ntask)

    min_error = np.min(error_list)
    error_list = error_list[:np.argmin(error_list)+1]
    lambda_1se = np.argmin(np.abs(error_list - min_error))
    final_estimates = estimates_dict[weight_list[lambda_1se]]
    final_intervals = intervals_dict[weight_list[lambda_1se]]
    final_coefs_var = coef_var_dict[weight_list[lambda_1se]]

    #Caculate final error on test set
    if (active_dict[weight_list[lambda_1se]] != 0).sum() > 0:
        final_error = 0
        predictive_r = []
        idx = 0
        for j in range(ntask):
            idx_new = np.sum(active_dict[weight_list[lambda_1se]][:, j] != 0)
            if idx_new == 0:
                final_error += np.sqrt(np.sum(np.square(response_test[j])) / sample_sizes_test)
            else:
                final_error += np.sqrt(np.sum(
                    np.square((response_test[j] - (predictor_vars_test)[:, (active_dict[weight_list[lambda_1se]][:, j] != 0)].dot(
                        estimates_dict[weight_list[lambda_1se]][idx:idx + idx_new])))) / sample_sizes_test)
                predictive_r.append(np.corrcoef(response_test[j],(predictor_vars_test)[:, (active_dict[weight_list[lambda_1se]][:, j] != 0)].dot(
                        estimates_dict[weight_list[lambda_1se]][idx:idx + idx_new]))[0,1])
            idx = idx + idx_new

    else:
        final_error = 0
        predictive_r = []
        for j in range(ntask):
            final_error += np.sqrt(np.linalg.norm(response_test[j], 2) ** 2 / sample_sizes_test)

    final_avg_error = final_error/ntask

    all_variables = {}
    significant = [final_intervals[j, 0] > 0 or final_intervals[j, 1] < 0 for j in range(np.shape(final_intervals)[0])]
    significant_variables = {}
    placeholder = 0
    for i in range(ntask):
        active_ = active_dict[weight_list[lambda_1se]][:, i] != 0
        new_placeholder = np.sum(active_)
        all_variables[i] = np.nonzero(active_)[0]
        significant_variables[i] = np.nonzero(active_)[0][significant[placeholder:placeholder+new_placeholder]]
        placeholder = placeholder + new_placeholder

    selective_interval_lengths = np.asarray(final_intervals[:,1]-final_intervals[:,0])

    return(final_estimates, final_intervals,selective_interval_lengths,all_variables,significant_variables,
           final_avg_error,predictive_r,final_coefs_var)


def ds_multi_task_selection_inference(predictor_vars_selection,predictor_vars_inference,predictor_vars_validate,
                                      predictor_vars_test,response_selection,response_inference,
                                        response_validate,response_test,weight_list,split=0.5):

    sample_sizes = predictor_vars_selection.shape[0]
    sample_sizes_validate = predictor_vars_validate.shape[0]
    sample_sizes_test = predictor_vars_test.shape[0]
    ridge_terms = np.zeros(ntask)
    nfeatures = predictor_vars_selection.shape[1]
    noise_levels = []
    for i in range(ntask):
       noise_levels.append(np.sqrt(np.sum(np.asarray(response_selection[i] - predictor_vars_selection.dot(np.linalg.pinv(predictor_vars_selection).dot(response_selection[i])))**2)/(np.int(sample_sizes)-nfeatures-1)))
    dispersions = [noise_levels[i]**2 for i in range(len(noise_levels))]
    randomizers = None
    estimates_dict = {}
    intervals_dict = {}
    coef_var_dict = {}
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
        CV = []

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
                CV = np.concatenate([CV,np.sqrt(np.diag(cov_target))/np.abs(observed_target)])
                alpha = 1. - 0.90
                quantile = ndist.ppf(1 - alpha / 2.)
                intervals = np.vstack([observed_target - quantile * np.sqrt(np.diag(cov_target)),
                                       observed_target + quantile * np.sqrt(np.diag(cov_target))]).T
                CIs = np.vstack([CIs, intervals])

                idx_new = np.sum(active_signs[:, i] != 0)
                if idx_new == 0:
                    error += np.sqrt(np.sum(np.square(response_validate[i])) / sample_sizes_validate)
                else:
                    error += np.sqrt(np.sum(np.square(
                    response_validate[i] - (predictor_vars_validate)[:, (active_signs[:, i] != 0)].dot(
                        observed_target))) / sample_sizes_validate)


        else:
            error = 0
            for j in range(ntask):
                error += np.sqrt(np.linalg.norm(response_validate[j], 2) ** 2 / sample_sizes_validate)
            CIs = np.asarray([[0, 0], [np.nan, np.nan]])

        estimates_dict[weight] = estimate
        intervals_dict[weight] = CIs
        coef_var_dict[weight] = CV
        active_dict[weight] = active_signs
        error_list.append(error/ntask)

    min_error = np.min(error_list)
    error_list = error_list[:np.argmin(error_list)+1]
    lambda_1se = np.argmin(np.abs(error_list - min_error))

    final_estimates = estimates_dict[weight_list[lambda_1se]]
    final_intervals = intervals_dict[weight_list[lambda_1se]][1:, ]
    final_coefs_var = coef_var_dict[weight_list[lambda_1se]]

    #Caculate final error on test set
    if (active_dict[weight_list[lambda_1se]] != 0).sum() > 0:
        final_error = 0
        predictive_r = []
        idx = 0
        for j in range(ntask):
            idx_new = np.sum(active_dict[weight_list[lambda_1se]][:, j] != 0)
            if idx_new == 0:
                final_error += np.sqrt(np.sum(np.square(response_test[j])) / sample_sizes_test)
            else:
                final_error += np.sqrt(np.sum(
                    np.square((response_test[j] - (predictor_vars_test)[:, (active_dict[weight_list[lambda_1se]][:, j] != 0)].dot(
                        estimates_dict[weight_list[lambda_1se]][idx:idx + idx_new])))) / sample_sizes_test)
                predictive_r.append(np.corrcoef(response_test[j], (predictor_vars_test)[:, (active_dict[weight_list[lambda_1se]][:, j] != 0)].dot(
                        estimates_dict[weight_list[lambda_1se]][idx:idx + idx_new]))[0,1])
            idx = idx + idx_new

    else:
        final_error = 0
        predictive_r = []
        for j in range(ntask):
            final_error += np.sqrt(np.linalg.norm(response_test[j], 2) ** 2 / sample_sizes_test)

    final_avg_error = final_error / ntask

    significant = [final_intervals[j, 0] > 0 or final_intervals[j, 1] < 0 for j in range(np.shape(final_intervals)[0])]
    significant_variables = {}
    all_variables_ds = {}
    placeholder = 0
    for i in range(ntask):
        active_ = active_dict[weight_list[lambda_1se]][:, i] != 0
        new_placeholder = np.sum(active_)
        all_variables_ds[i] = np.nonzero(active_)[0]
        significant_variables[i] = np.nonzero(active_)[0][significant[placeholder:placeholder+new_placeholder]]
        placeholder = placeholder + new_placeholder

    ds_interval_lengths = np.asarray(final_intervals[:,1]-final_intervals[:,0])

    return (final_estimates, final_intervals, ds_interval_lengths, all_variables_ds, significant_variables,
            final_avg_error, predictive_r, final_coefs_var)

final_estimates_rand1, final_intervals_rand1, selective1_intervals, all_variables_rand1, significant_variables_rand1, final_err_rand1, pred_r_rand1, coefs_var_rand1 = \
    rand_multi_task_selection_inference(predictors_train,predictors_validate,predictors_test, responses_train,
                                        responses_validate, responses_test,weight_list = np.arange(26,56,1.0),rand_scale=1.0)

print(final_err_rand1, "Average testing error per task, rand scale 1.0")
print(pred_r_rand1, "Predictive r, rand scale 1.0")
print(np.mean(selective1_intervals),"Mean interval length, rand scale 1.0")
print(np.std(selective1_intervals), "Sd interval length, rand scale 1.0")
print(len(selective1_intervals),"Number selected")
print(np.sum([len(significant_variables_rand1[i]) for i in range(len(significant_variables_rand1))]),"Sum of significant across tasks")


match_length_indx = {}
start = 0
for i in range(ntask):
    match_length_indx[i] = selective1_intervals[start:start+len(all_variables_rand1[i])]
    start += len(all_variables_rand1[i])

#Predict g
#Task scores
task_scores = []
start = 0
for i in range(ntask):
    task_scores.append(predictors_train[:, all_variables_rand1[i]].dot(final_estimates_rand1[start:start + len(all_variables_rand1[i])]))
    start += len(all_variables_rand1[i])

#Estimate coefficients
y = glavaan[train]
observed_target = (np.linalg.pinv(task_scores).T).dot(y)

#Predicted g
test_task_scores = []
start = 0
for i in range(ntask):
    test_task_scores.append(predictors_test[:, all_variables_rand1[i]].dot(final_estimates_rand1[start:start + len(all_variables_rand1[i])]))
    start += len(all_variables_rand1[i])

pred_g = test_task_scores.dot(observed_target)
pred_r_general = np.corrcoef(glavaan[test],pred_g)
print("general pred r, rand scale 1.0",pred_r_general)

#Data splitting
sample_sizes = predictors_train.shape[0]
samples = np.arange(np.int(sample_sizes))
selection = np.random.choice(samples, size=np.int(0.5 * sample_sizes), replace=False)
inference = np.setdiff1d(samples, selection)
print(selection)
print(inference)
responses_selection = {j: responses_train[j][selection] for j in range(ntask)}
predictors_selection = predictors_train[selection,:]
responses_inference = {j: responses_train[j][inference] for j in range(ntask)}
predictors_inference = predictors_train[inference,:]


final_estimates_ds50, final_intervals_ds50, ds50_intervals, all_variables_ds50, significant_variables_ds50, final_err_ds50, pred_r_ds50, coefs_var_ds50 = \
    ds_multi_task_selection_inference(predictors_selection,predictors_inference,predictors_validate,predictors_test, responses_selection, responses_inference,
                                        responses_validate, responses_test, weight_list = np.arange(12,37,1.5),split=0.5)

print(final_err_ds50, "Average testing error per task, data split 50/50")
print(pred_r_ds50, "Predictive r, data split 50/50")
print(np.mean(ds50_intervals),"Mean interval length, data split 50/50")
print(np.std(ds50_intervals), "Sd interval length, data split 50/50")
print(len(ds50_intervals),"Number selected")
print(np.sum([len(significant_variables_ds50[i]) for i in range(len(significant_variables_ds50))]),"Sum of significant across tasks")

match_length_indx2 = {}
start2 = 0
for i in range(ntask):
    match_length_indx2[i] = ds50_intervals[start2:start2+len(all_variables_ds50[i])]
    start2 += len(all_variables_ds50[i])

#Predict g
#Task scores
task_scores = []
start = 0
for i in range(ntask):
    task_scores.append(predictors_train[:, all_variables_ds50[i]].dot(final_estimates_ds50[start:start + len(all_variables_ds50[i])]))
    start += len(all_variables_ds50[i])

#Estimate coefficients
y = glavaan[train]
observed_target = (np.linalg.pinv(task_scores).T).dot(y)

#Predicted g
test_task_scores = []
start = 0
for i in range(ntask):
    test_task_scores.append(predictors_test[:, all_variables_ds50[i]].dot(final_estimates_ds50[start:start + len(all_variables_ds50[i])]))
    start += len(all_variables_ds50[i])

pred_g = test_task_scores.dot(observed_target)
pred_r_general = np.corrcoef(glavaan[test],pred_g)
print("general pred r, data split 50/50",pred_r_general)

common = {i:np.intersect1d(all_variables_rand1[i],all_variables_ds50[i]) for i in range(ntask)}
print("common",common)
common_lengths = []
for i in range(ntask):
    for predictor in common[i]:
        ratio_length = match_length_indx2[i][np.argwhere(all_variables_ds50[i]==predictor)[0][0]]/match_length_indx[i][np.argwhere(all_variables_rand1[i]==predictor)[0][0]]
        common_lengths.append(ratio_length)
print(common_lengths)

#----------------------------------------------------------------

final_estimates_rand07, final_intervals_rand07, selective07_intervals, all_variables_rand07, significant_variables_rand07, final_err_rand07, pred_r_rand07, coefs_var_rand07 = \
    rand_multi_task_selection_inference(predictors_train,predictors_validate,predictors_test, responses_train,
                                        responses_validate, responses_test,weight_list = np.arange(30,57,1.5),rand_scale=0.7)

print(final_err_rand07, "Average testing error per task, rand scale 0.7")
print(pred_r_rand07, "Predictive r, rand scale 0.7")
print(np.mean(selective07_intervals),"Mean interval length, rand scale 0.7")
print(np.std(selective07_intervals), "Sd interval length, rand scale 0.7")
print(np.sum([len(all_variables_rand07[i]) for i in range(len(all_variables_rand1))]),"Sum of selected across tasks")
print(np.sum([len(significant_variables_rand07[i]) for i in range(len(significant_variables_rand1))]),"Sum of "
                                                                                                         "significant across tasks")
match_length_indx = {}
start = 0
for i in range(ntask):
    match_length_indx[i] = selective07_intervals[start:start+len(all_variables_rand07[i])]
    start += len(all_variables_rand07[i])

#Predict g
#Task scores
task_scores = []
start = 0
for i in range(ntask):
    task_scores.append(predictors_train[:, all_variables_rand07[i]].dot(final_estimates_rand07[start:start + len(all_variables_rand07[i])]))
    start += len(all_variables_rand07[i])

#Estimate coefficients
y = glavaan[train]
observed_target = (np.linalg.pinv(task_scores).T).dot(y)

#Predicted g
test_task_scores = []
start = 0
for i in range(ntask):
    test_task_scores.append(predictors_test[:, all_variables_rand07[i]].dot(final_estimates_rand07[start:start + len(all_variables_rand07[i])]))
    start += len(all_variables_rand07[i])

pred_g = test_task_scores.dot(observed_target)
pred_r_general = np.corrcoef(glavaan[test],pred_g)
print("general pred r, rand scale 0.7",pred_r_general)

samples = np.arange(np.int(sample_sizes))
selection = np.random.choice(samples, size=np.int(0.67 * sample_sizes), replace=False)
inference = np.setdiff1d(samples, selection)
response_selection = {j: responses_train[j][selection] for j in range(ntask)}
predictor_vars_selection = predictors_train[selection,:]
response_inference = {j: responses_train[j][inference] for j in range(ntask)}
predictor_vars_inference = predictors_train[inference,:]

final_estimates_ds67, final_intervals_ds67, ds67_intervals, all_variables_ds67, significant_variables_ds67, final_err_ds67, pred_r_ds67, coefs_var_ds67 = \
    ds_multi_task_selection_inference(predictors_selection,predictors_inference,predictors_validate,predictors_test, responses_train,
                                        responses_validate, responses_test,weight_list = np.arange(20,45,1.5),split=0.67)

print(final_err_ds67, "Average testing error per task, data split 67/33")
print(pred_r_ds67, "Predictive r, data split 67/33")
print(np.mean(ds67_intervals),"Mean interval length, data split 67/33")
print(np.std(ds67_intervals), "Sd interval length, data split 67/33")
print(np.sum([len(all_variables_ds67[i]) for i in range(len(all_variables_ds67))]),"Sum of selected across tasks")
print(np.sum([len(significant_variables_ds67[i]) for i in range(len(significant_variables_ds67))]),"Sum of significant across tasks")

#Predict g
#Task scores
task_scores = []
start = 0
for i in range(ntask):
    task_scores.append(predictors_train[:, all_variables_ds67[i]].dot(final_estimates_ds67[start:start + len(all_variables_ds67[i])]))
    start += len(all_variables_ds67[i])

#Estimate coefficients
y = glavaan[train]
observed_target = (np.linalg.pinv(task_scores).T).dot(y)

#Predicted g
test_task_scores = []
start = 0
for i in range(ntask):
    test_task_scores.append(predictors_test[:, all_variables_ds67[i]].dot(final_estimates_ds67[start:start + len(all_variables_ds67[i])]))
    start += len(all_variables_ds67[i])

pred_g = test_task_scores.dot(observed_target)
pred_r_general = np.corrcoef(glavaan[test],pred_g)
print("general pred r, data split 67/33",pred_r_general)

match_length_indx2 = {}
start2 = 0
for i in range(ntask):
    match_length_indx2[i] = ds67_intervals[start2:start2+len(all_variables_ds67[i])]
    start2 += len(all_variables_ds67[i])

common = {i:np.intersect1d(all_variables_rand07[i],all_variables_ds67[i]) for i in range(ntask)}
print("common",common)
common_lengths_67 = []
for i in range(ntask):
    for predictor in common[i]:
        ratio_length = match_length_indx2[i][np.argwhere(all_variables_ds67[i]==predictor)[0][0]]/match_length_indx[i][np.argwhere(all_variables_rand07[i]==predictor)[0][0]]
        common_lengths_67.append(ratio_length)

def set_box_color(bp, color, linestyle):
    plt.setp(bp['boxes'], color=color, linestyle=linestyle, linewidth=3.5)
    plt.setp(bp['whiskers'], color=color, linestyle=linestyle, linewidth=3.5)
    plt.setp(bp['caps'], color=color, linewidth=2.5)
    plt.setp(bp['medians'], color=color, linewidth=2.5)

fig = plt.figure(figsize=(17, 14))
ax1 = fig.add_subplot(111)

plt.sca(ax1)
first = plt.boxplot([common_lengths_67], positions=np.asarray([1]), sym='', widths=0.3)
second = plt.boxplot([common_lengths], positions=np.asarray([1.6]), sym='', widths=0.3)
set_box_color(first, '#35978f', 'solid')  # colors are from http://colorbrewer2.org/
set_box_color(second, '#35978f', '--')
plt.xlim(0.7, 1.9)
plt.tight_layout()
plt.plot([], c='#35978f', label='67/33 Split: Selective Inference', linewidth=2.5)
plt.plot([], c='#35978f', label='50/50 Split: Selective Inference', linestyle='--', linewidth=2.5)
plt.legend()
plt.ylabel('Ratio of Interval Lengths for Common Parameters', fontsize=24)
plt.yticks(fontsize=20)

ax1.set_title("Ratio of Confidence Interval Lengths", y=1.01 ,fontsize=32)
ax1.legend(loc='lower left', bbox_to_anchor=(0.3225, -0.125), fontsize=20)
ax1.set_xticklabels([])
ax1.set_xticks([])

def common_format(ax):
    ax.grid(True, which='both', color='#f0f0f0')
    ax.set_xlabel('Method', fontsize=20)
    return ax

common_format(ax1)
ax1.axhline(y=1.0, color='k', linestyle='--', linewidth=2.5)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('real_data_lengths2.png', bbox_inches='tight')


fig = plt.figure(figsize=(17, 14))
ax1 = fig.add_subplot(111)

plt.sca(ax1)
first = plt.boxplot([selective07_intervals], positions=np.asarray([1]), sym='', widths=0.3)
second = plt.boxplot([selective1_intervals], positions=np.asarray([1.8]), sym='', widths=0.3)
fourth = plt.boxplot([ds67_intervals], positions=np.asarray([1.3]), sym='', widths=0.3)
fifth = plt.boxplot([ds50_intervals], positions=np.asarray([2.1]), sym='', widths=0.3)
set_box_color(first, '#2b8cbe', 'solid')  # colors are from http://colorbrewer2.org/
set_box_color(second, '#6baed6', '--')
set_box_color(fourth, '#238443', 'solid')
set_box_color(fifth, '#31a354', '--')
plt.xlim(0.7, 2.4)
plt.tight_layout()
plt.plot([], c='#2b8cbe', label='Randomized Multi-Task Lasso 0.7', linewidth=2.5)
plt.plot([], c='#238443', label='Data Splitting 67/33', linewidth=2.5)
plt.plot([], c='#6baed6', label='Randomized Multi-Task Lasso 1.0', linestyle='--', linewidth=2.5)
plt.plot([], c='#31a354', label='Data Splitting 50/50', linestyle='--', linewidth=2.5)
plt.legend()
plt.ylabel('Interval Length', fontsize=20)
plt.yticks(fontsize=18)

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



fig = plt.figure(figsize=(17, 14))
ax1 = fig.add_subplot(111)
plt.sca(ax1)
first = plt.boxplot([coefs_var_rand07], positions=np.asarray([1]), sym='', widths=0.3)
second = plt.boxplot([coefs_var_rand1], positions=np.asarray([1.8]), sym='', widths=0.3)
fourth = plt.boxplot([coefs_var_ds67], positions=np.asarray([1.3]), sym='', widths=0.3)
fifth = plt.boxplot([coefs_var_ds50], positions=np.asarray([2.1]), sym='', widths=0.3)
set_box_color(first, '#2b8cbe', 'solid')  # colors are from http://colorbrewer2.org/
set_box_color(second, '#6baed6', '--')
set_box_color(fourth, '#238443', 'solid')
set_box_color(fifth, '#31a354', '--')
plt.xlim(0.7, 2.4)
plt.tight_layout()
plt.plot([], c='#2b8cbe', label='Randomized Multi-Task Lasso 0.7', linewidth=2.5)
plt.plot([], c='#238443', label='Data Splitting 67/33', linewidth=2.5)
plt.plot([], c='#6baed6', label='Randomized Multi-Task Lasso 1.0', linestyle='--', linewidth=2.5)
plt.plot([], c='#31a354', label='Data Splitting 50/50', linestyle='--', linewidth=2.5)
plt.legend()
plt.ylabel('Coefficient of Variation', fontsize=20)
plt.yticks(fontsize=18)

ax1.set_title("Coefficient of Variation for Estimated Model Parameters", y=1.01 ,fontsize=24)
ax1.legend(loc='lower left', bbox_to_anchor=(0.319, -0.225), fontsize=20)
ax1.set_xticklabels([])
ax1.set_xticks([])

def common_format(ax):
    ax.grid(True, which='both', color='#f0f0f0')
    ax.set_xlabel('Method', fontsize=20)
    return ax

common_format(ax1)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig('real_data_cv.png', bbox_inches='tight')