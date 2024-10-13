import numpy as np
import matplotlib
matplotlib.use('agg')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t as tdist
from scipy.stats import norm as ndist
import regreg.api as rr
from selectinf.randomized.randomization import randomization
from selectinf.randomized.multitask_lasso import multi_task_lasso
from selectinf.randomized.lasso import lasso, selected_targets
np.random.seed(5)

ntask = 11

#Load fmri and cognitive task data
predictors_train = np.genfromtxt('train.csv', delimiter=',')[1:,:-12]
predictors_validate = np.genfromtxt('validate.csv', delimiter=',')[1:,:-12]
predictors_test = np.genfromtxt('test.csv', delimiter=',')[1:,:-12]

responses_train = {}
responses_validate = {}
responses_test = {}
#task_index = [-2, -5, -9, -11]

for i in range(ntask):
    responses_train[i] = np.genfromtxt('train.csv', delimiter=',')[1:, -ntask+i]
    scale = np.std(responses_train[i])
    responses_train[i] /= scale
    responses_validate[i] = np.genfromtxt('validate.csv', delimiter=',')[1:, -ntask+i]
    responses_validate[i] /=  scale
    responses_test[i] = np.genfromtxt('test.csv', delimiter=',')[1:, -ntask+i]
    responses_test[i] /= scale
    print("here")

#PC loadings and singular values
V = np.genfromtxt('V.csv', delimiter=',')[1:,:]
sv = np.genfromtxt('lambda.csv', delimiter=',')[1:]

#g factor
#g_train = np.genfromtxt('train.csv', delimiter=',')[1:,-12]
#g_test = np.genfromtxt('test.csv', delimiter=',')[1:,-12]

print("HI")

def _noise(n, df=np.inf):
    if df == np.inf:
        return np.random.standard_normal(n)
    else:
        sd_t = np.std(tdist.rvs(df, size=50000))
    return tdist.rvs(df, size=n) / sd_t

#Generate randomization variable
noise = _noise(predictors_train.shape[1]*ntask)

#Function to perform randomized model selection and conduct post-selection inference
#Takes as input the training data, validation data, testing data, lambda path (weight list), randomization variable, and randomizer scale
#Returns the selective MLE, post-selection intervals, and interval lengths
#Also returns a dictionary of selected features by task, a dictionary of significant features by task, and performance metrics on the testing data

def rand_multi_task_selection_inference(predictor_vars_train,predictor_vars_validate,predictor_vars_test,response_train,
                                        response_validate,response_test,weight_list,noise,rand_scale=0.7):

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

    #Setup for post-selection inference
    noise_levels = []
    for i in range(ntask):
        noise_levels.append(np.sqrt(np.sum(np.array(response_train[i] - predictor_vars_train.dot(
            np.linalg.pinv(predictor_vars_train).dot(response_train[i]))) ** 2) / (sample_sizes - nfeatures)))
    dispersions = [noise_levels[i] ** 2 for i in range(len(noise_levels))]
    randomizer_scales = rand_scale * np.asarray([noise_levels[i] for i in range(ntask)])
    randomizers = {i: randomization.isotropic_gaussian((nfeatures,), randomizer_scales[i]) for i in range(ntask)}
    perturbations = np.array([randomizer_scales[i] * noise[i*nfeatures:(i+1)*nfeatures] for i in range(ntask)]).T

    #Perform inference for given tuning parameter
    for weight in weight_list:
        feature_weight = weight * np.ones(nfeatures)
        loglikes = {
            i: rr.glm.gaussian(predictor_vars_train, response_train[i], coef=1., quadratic=None)
            for i in range(ntask)}
        multi_lasso = multi_task_lasso(loglikes, np.asarray(feature_weight), ridge_terms, randomizers, nfeatures, ntask, None)
        active_signs = multi_lasso.fit(perturbations=perturbations)
        estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(dispersions=dispersions)

        #Track the active variables, selective MLE, post-selection intervals, and coefficient of varation for each lambda
        estimates_dict[weight] = estimate
        coef_var_dict[weight] = np.sqrt(np.diag(observed_info_mean)) / np.abs(estimate)
        intervals_dict[weight] = intervals
        active_dict[weight] = active_signs

        #Caculate error on validation data for given lambda
        if (active_signs != 0).sum() > 0:
            error = 0
            idx = 0
            for j in range(ntask):
                idx_new = np.sum(active_signs[:, j] != 0)
                if idx_new == 0:
                    error += np.sqrt(np.sum(np.square(response_validate[j])) / sample_sizes_validate)
                else:
                    #If there are no active predictors for task j
                    error += np.sqrt(np.sum(
                        np.square((response_validate[j] - predictor_vars_validate[:, (active_signs[:, j] != 0)].dot(
                            estimate[idx:idx + idx_new])))) / sample_sizes_validate)
                idx = idx + idx_new

        else:
            #If there are no active predictors for any task
            error = 0
            for j in range(ntask):
                error += np.qrt((np.linalg.norm(response_validate[j], 2) ** 2) / sample_sizes_validate)

        error_list.append(error/ntask)

    min_error = np.argmin(error_list)
    final_estimates = estimates_dict[weight_list[min_error]]
    final_intervals = intervals_dict[weight_list[min_error]]
    final_coefs_var = coef_var_dict[weight_list[min_error]]

    #Caculate final testing error and predictive r on test set
    if (active_dict[weight_list[min_error]] != 0).sum() > 0:
        final_error = 0
        predictive_r = []
        idx = 0
        for j in range(ntask):
            idx_new = np.sum(active_dict[weight_list[min_error]][:, j] != 0)
            if idx_new == 0:
                final_error += np.sqrt(np.sum(np.square(response_test[j])) / sample_sizes_test)
            else:
                final_error += np.sqrt(np.sum(
                    np.square((response_test[j] - predictor_vars_test[:, (active_dict[weight_list[min_error]][:, j] != 0)].dot(
                        estimates_dict[weight_list[min_error]][idx:idx + idx_new])))) / sample_sizes_test)
                predictive_r.append(np.corrcoef(response_test[j],predictor_vars_test[:, (active_dict[weight_list[min_error]][:, j] != 0)].dot(
                        estimates_dict[weight_list[min_error]][idx:idx + idx_new]))[0,1])
            idx = idx + idx_new

    else:
        final_error = 0
        predictive_r = []
        for j in range(ntask):
            final_error += np.sqrt(np.linalg.norm(response_test[j], 2) ** 2 / sample_sizes_test)

    #Average final testing error by task
    final_avg_error = final_error/ntask

    #Identify intervals that do not cover zero
    all_variables = {}
    significant = [final_intervals[j, 0] > 0 or final_intervals[j, 1] < 0 for j in range(np.shape(final_intervals)[0])]
    significant_variables = {}
    placeholder = 0
    for i in range(ntask):
        #Identify variables (by task) corresponding to the significant intervals
        active_ = active_dict[weight_list[min_error]][:, i] != 0
        new_placeholder = np.sum(active_)
        all_variables[i] = np.nonzero(active_)[0]
        significant_variables[i] = np.nonzero(active_)[0][significant[placeholder:placeholder+new_placeholder]]
        placeholder = placeholder + new_placeholder

    selective_interval_lengths = np.asarray(final_intervals[:,1]-final_intervals[:,0])

    return(final_estimates, final_intervals,selective_interval_lengths,all_variables,significant_variables,
           final_avg_error,predictive_r,final_coefs_var)

def rand_single_task_selection_inference(predictor_vars_train,predictor_vars_validate,predictor_vars_test,response_train,
                                        response_validate,response_test,weight_list,noise,rand_scale=0.7):

    sample_sizes = predictor_vars_train.shape[0]
    sample_sizes_validate = predictor_vars_validate.shape[0]
    sample_sizes_test = predictor_vars_test.shape[0]
    nfeatures = predictor_vars_train.shape[1]
    estimates_dict = {}
    coef_var_dict = {}
    intervals_dict = {}
    active_dict = {}
    error_list = []

    #Setup for post-selection inference
    noise_levels = []
    for i in range(ntask):
        noise_levels.append(np.sqrt(np.sum(np.array(response_train[i] - predictor_vars_train.dot(
            np.linalg.pinv(predictor_vars_train).dot(response_train[i]))) ** 2) / (sample_sizes - nfeatures)))
    randomizer_scales = rand_scale * np.asarray([noise_levels[i] for i in range(ntask)])

    #Perform inference for given tuning parameter
    for weight in weight_list:
        estimate = np.asarray([])
        intervals = np.asarray([[0, 0]])
        coef_vars = np.asarray([])
        active_signs = np.zerso((nfeatures,ntask))

        for i in range(ntask):

            W = np.ones(nfeatures) * weight
            single_task_lasso = lasso.gaussian(predictor_vars_train[i],
                                               response_train[i],
                                               W,
                                               sigma=noise_levels[i],
                                               ridge_term=0.,
                                               randomizer_scale=randomizer_scales[i])

            initial_omega = np.array(randomizer_scales[i] * noise_levels[i] * noise[i*nfeatures:(i+1)*nfeatures]).T
            signs = single_task_lasso.fit(perturb=initial_omega)
            nonzero = signs != 0

            (observed_target, cov_target, cov_target_score, alternatives) = \
                selected_targets(single_task_lasso.loglike, single_task_lasso._W, nonzero, dispersion=noise_levels[i] ** 2)

            try:
                MLE_result, observed_fi = single_task_lasso.selective_MLE(
                    observed_target,
                    cov_target,
                    cov_target_score,
                    level=0.90)[0:2]

                estimate.extend(MLE_result['MLE'])
                task_intervals = np.asarray(MLE_result[['lower_confidence', 'upper_confidence']])
                intervals = np.vstack([intervals, task_intervals])
                coef_vars.extend(np.sqrt(np.diag(observed_fi)) / np.abs(estimate))

            except:
                pass

            active_signs[i,:] = signs

        #Track the active variables, selective MLE, post-selection intervals, and coefficient of varation for each lambda
        estimates_dict[weight] = estimate
        coef_var_dict[weight] = coef_vars
        intervals_dict[weight] = intervals
        active_dict[weight] = active_signs

        #Caculate error on validation data for given lambda
        if (active_signs != 0).sum() > 0:
            error = 0
            idx = 0
            for j in range(ntask):
                idx_new = np.sum(active_signs[:, j] != 0)
                if idx_new == 0:
                    error += np.sqrt(np.sum(np.square(response_validate[j])) / sample_sizes_validate)
                else:
                    #If there are no active predictors for task j
                    error += np.sqrt(np.sum(
                        np.square((response_validate[j] - predictor_vars_validate[:, (active_signs[:, j] != 0)].dot(
                            estimate[idx:idx + idx_new])))) / sample_sizes_validate)
                idx = idx + idx_new

        else:
            #If there are no active predictors for any task
            error = 0
            for j in range(ntask):
                error += np.qrt((np.linalg.norm(response_validate[j], 2) ** 2) / sample_sizes_validate)

        error_list.append(error/ntask)

    min_error = np.argmin(error_list)
    final_estimates = estimates_dict[weight_list[min_error]]
    final_intervals = intervals_dict[weight_list[min_error]]
    final_coefs_var = coef_var_dict[weight_list[min_error]]

    #Caculate final testing error and predictive r on test set
    if (active_dict[weight_list[min_error]] != 0).sum() > 0:
        final_error = 0
        predictive_r = []
        idx = 0
        for j in range(ntask):
            idx_new = np.sum(active_dict[weight_list[min_error]][:, j] != 0)
            if idx_new == 0:
                final_error += np.sqrt(np.sum(np.square(response_test[j])) / sample_sizes_test)
            else:
                final_error += np.sqrt(np.sum(
                    np.square((response_test[j] - predictor_vars_test[:, (active_dict[weight_list[min_error]][:, j] != 0)].dot(
                        estimates_dict[weight_list[min_error]][idx:idx + idx_new])))) / sample_sizes_test)
                predictive_r.append(np.corrcoef(response_test[j],predictor_vars_test[:, (active_dict[weight_list[min_error]][:, j] != 0)].dot(
                        estimates_dict[weight_list[min_error]][idx:idx + idx_new]))[0,1])
            idx = idx + idx_new

    else:
        final_error = 0
        predictive_r = []
        for j in range(ntask):
            final_error += np.sqrt(np.linalg.norm(response_test[j], 2) ** 2 / sample_sizes_test)

    #Average final testing error by task
    final_avg_error = final_error/ntask

    #Identify intervals that do not cover zero
    all_variables = {}
    significant = [final_intervals[j, 0] > 0 or final_intervals[j, 1] < 0 for j in range(np.shape(final_intervals)[0])]
    significant_variables = {}
    placeholder = 0
    for i in range(ntask):
        #Identify variables (by task) corresponding to the significant intervals
        active_ = active_dict[weight_list[min_error]][:, i] != 0
        new_placeholder = np.sum(active_)
        all_variables[i] = np.nonzero(active_)[0]
        significant_variables[i] = np.nonzero(active_)[0][significant[placeholder:placeholder+new_placeholder]]
        placeholder = placeholder + new_placeholder

    selective_interval_lengths = np.asarray(final_intervals[:,1]-final_intervals[:,0])

    return(final_estimates, final_intervals,selective_interval_lengths,all_variables,significant_variables,
           final_avg_error,predictive_r,final_coefs_var)

#Similar function for data splitting
def ds_multi_task_selection_inference(predictor_vars_selection,predictor_vars_inference,predictor_vars_validate,
                                      predictor_vars_test,response_selection,response_inference,
                                        response_validate,response_test,weight_list):

    sample_sizes_inference = predictor_vars_inference.shape[0]
    sample_sizes_validate = predictor_vars_validate.shape[0]
    sample_sizes_test = predictor_vars_test.shape[0]
    ridge_terms = np.zeros(ntask)
    nfeatures = predictor_vars_selection.shape[1]
    noise_levels = []
    for i in range(ntask):
       noise_levels.append(np.sqrt(np.sum(np.asarray(response_inference[i] - predictor_vars_inference.dot(np.linalg.pinv(predictor_vars_inference).dot(response_inference[i])))**2)/(sample_sizes_inference-nfeatures)))
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
                    response_validate[i] - predictor_vars_validate[:, (active_signs[:, i] != 0)].dot(
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

    min_error = np.argmin(error_list)

    final_estimates = estimates_dict[weight_list[min_error]]
    final_intervals = intervals_dict[weight_list[min_error]][1:, ]
    final_coefs_var = coef_var_dict[weight_list[min_error]]

    #Caculate final error and predictive r on test set
    if (active_dict[weight_list[min_error]] != 0).sum() > 0:
        final_error = 0
        predictive_r = []
        idx = 0
        for j in range(ntask):
            idx_new = np.sum(active_dict[weight_list[min_error]][:, j] != 0)
            if idx_new == 0:
                final_error += np.sqrt(np.sum(np.square(response_test[j])) / sample_sizes_test)
            else:
                final_error += np.sqrt(np.sum(
                    np.square((response_test[j] - predictor_vars_test[:, (active_dict[weight_list[min_error]][:, j] != 0)].dot(
                        estimates_dict[weight_list[min_error]][idx:idx + idx_new])))) / sample_sizes_test)
                predictive_r.append(np.corrcoef(response_test[j], predictor_vars_test[:, (active_dict[weight_list[min_error]][:, j] != 0)].dot(
                        estimates_dict[weight_list[min_error]][idx:idx + idx_new]))[0,1])
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
        active_ = active_dict[weight_list[min_error]][:, i] != 0
        new_placeholder = np.sum(active_)
        all_variables_ds[i] = np.nonzero(active_)[0]
        significant_variables[i] = np.nonzero(active_)[0][significant[placeholder:placeholder+new_placeholder]]
        placeholder = placeholder + new_placeholder

    ds_interval_lengths = np.asarray(final_intervals[:,1]-final_intervals[:,0])

    return (final_estimates, final_intervals, ds_interval_lengths, all_variables_ds, significant_variables,
            final_avg_error, predictive_r, final_coefs_var)

#Compare selective inference with 50/50 data split
#Learn weights for g from 11 task scores
#task_scores = np.genfromtxt('train.csv', delimiter=',')[1:,-11:]
#y = np.asarray(g_train)
#weights = np.linalg.pinv(task_scores).dot(y)

#X = predictors_train
#y = g_train


#-----------------------------------------------------------------
#Compare selective inference to data splitting 67/33

final_estimates_rand07, final_intervals_rand07, selective07_intervals, all_variables_rand07, significant_variables_rand07, final_err_rand07, pred_r_rand07, coefs_var_rand07 = \
    rand_multi_task_selection_inference(predictors_train,predictors_validate,predictors_test, responses_train,
                                        responses_validate, responses_test,np.arange(3.75,4.0,0.25),noise,rand_scale=0.7)

# 3.25 model with all 4
# 2.4 M and L
# 3.0 RC and PV

# 2.4 M and L
# 3-3.25 0.25 RC PV
# 2.5 - 4 0.25
# 1.75 - 3- 0.25
# 3 - 5
jacard_matrix = np.zeros((ntask,ntask))

for i in range(ntask):
    for j in range(ntask):
        jacard_matrix[i,j] = round(len(np.intersect1d(significant_variables_rand07[i],significant_variables_rand07[j]))/len(np.union1d(significant_variables_rand07[i],significant_variables_rand07[j])),2)

print(jacard_matrix)

j_list = []
for i in range(ntask):
    for j in range(ntask):
        if j>i:
            j_list.append(jacard_matrix[i,j])

print(np.mean(j_list))

mat = sns.heatmap(jacard_matrix,vmin=0,vmax=1,cmap="viridis_r")
#mat.set_xticklabels(['PV','FT','LS','CS','PC','PS','RC','Ravlt-Sd','Ravlt-Ld','Matrix','LMT'],rotation=90)
#mat.set_yticklabels(['PV','FT','LS','CS','PC','PS','RC','Ravlt-Sd','Ravlt-Ld','Matrix','LMT'],rotation=0)
fig = mat.get_figure()
fig.tight_layout()
fig.savefig("jaccard_MTL.png")


print(final_err_rand07, "Average testing error per task, rand scale 0.7")
print(pred_r_rand07, "Predictive r, rand scale 0.7")
print(np.mean(selective07_intervals),"Mean interval length, rand scale 0.7")
print(np.std(selective07_intervals), "Sd interval length, rand scale 0.7")
print(np.sum([len(all_variables_rand07[i]) for i in range(len(all_variables_rand07))]),"Sum of selected in total")
print(np.sum([len(significant_variables_rand07[i]) for i in range(len(significant_variables_rand07))]),"Sum of significant in total")
print(significant_variables_rand07,"Significant PCs by task")

#Estimate coefficients in original feature space
running_counter = 0
original_coef_approx = np.zeros((np.shape(V)[0],ntask))
for i in range(ntask):
    singular_values = sv[all_variables_rand07[i]]
    original_coef_approx[:,i] = V[:,all_variables_rand07[i]].dot(np.divide(final_estimates_rand07[running_counter:running_counter+len(all_variables_rand07[i])],singular_values))
    running_counter += len(all_variables_rand07[i])
np.savetxt("original_approx07.csv",original_coef_approx,delimiter=",")


match_length_indx = {}
start = 0
for i in range(ntask):
    match_length_indx[i] = selective07_intervals[start:start+len(all_variables_rand07[i])]
    start += len(all_variables_rand07[i])

#Predict g on testing data using 11 estimated task scores
test_task_scores = []
start = 0
for i in range(ntask):
    test_task_scores.append(predictors_test[:, all_variables_rand07[i]].dot(final_estimates_rand07[start:start + len(all_variables_rand07[i])]))
    start += len(all_variables_rand07[i])
test_task_scores = np.asarray(test_task_scores)

#pred_g = (test_task_scores.T).dot(weights)
#pred_r_general = np.corrcoef(g_test,pred_g)
#print("pred r for g using estimated task scores rand scale 0.7",pred_r_general)

#Model g on testing data with just significant PCs from training
all_significant_predictors = np.asarray([])
for i in range(ntask):
    all_significant_predictors = np.union1d(all_significant_predictors,significant_variables_rand07[i])
print(all_significant_predictors)
all_significant_predictors = np.asarray([int(all_significant_predictors[i]) for i in range(len(all_significant_predictors))])

#X = predictors_train
#y = g_train
#observed_target = np.linalg.pinv(X[:, all_significant_predictors]).dot(y)

singular_values = sv[all_significant_predictors]
#original_coef_approx_g = V[:,all_significant_predictors].dot(np.divide(observed_target,singular_values))
#np.savetxt("original_coef_approx_g07.csv",original_coef_approx_g,delimiter=",")

#Predict g on testing data with just significant PCs from training

#pred_g = predictors_test[:, all_significant_predictors].dot(observed_target)
#pred_r_general = np.corrcoef(g_test,pred_g)
#print("pred r for g using only significant PCs, rand scale 0.7",pred_r_general)

#Data splitting 67/33
sample_sizes = predictors_train.shape[0]
samples = np.arange(int(sample_sizes))
selection = np.random.choice(samples, size=int(0.67 * sample_sizes), replace=False)
inference = np.setdiff1d(samples, selection)
responses_selection = {j: responses_train[j][selection] for j in range(ntask)}
predictors_selection = predictors_train[selection,:]
responses_inference = {j: responses_train[j][inference] for j in range(ntask)}
predictors_inference = predictors_train[inference,:]

final_estimates_ds67, final_intervals_ds67, ds67_intervals, all_variables_ds67, significant_variables_ds67, final_err_ds67, pred_r_ds67, coefs_var_ds67 = \
    ds_multi_task_selection_inference(predictors_selection,predictors_inference,predictors_validate,predictors_test, responses_selection, responses_inference,
                                        responses_validate, responses_test,weight_list = np.arange(0.5, 3.5, 0.25))

#24-38
print(final_err_ds67, "Average testing error per task, data split 67/33")
print(pred_r_ds67, "Predictive r, data split 67/33")
print(np.mean(ds67_intervals),"Mean interval length, data split 67/33")
print(np.std(ds67_intervals), "Sd interval length, data split 67/33")
print(np.sum([len(all_variables_ds67[i]) for i in range(len(all_variables_ds67))]),"Sum of selected across tasks")
print(np.sum([len(significant_variables_ds67[i]) for i in range(len(significant_variables_ds67))]),"Sum of significant across tasks")

#Predict g
test_task_scores = []
start = 0
for i in range(ntask):
    test_task_scores.append(predictors_test[:, all_variables_ds67[i]].dot(final_estimates_ds67[start:start + len(all_variables_ds67[i])]))
    start += len(all_variables_ds67[i])
test_task_scores = np.asarray(test_task_scores)

pred_g = (test_task_scores.T).dot(weights)
pred_r_general = np.corrcoef(g_test,pred_g)
print("general pred r, data split 67/33",pred_r_general)

#Data splitting 50/50
sample_sizes = predictors_train.shape[0]
samples = np.arange(int(sample_sizes))
selection = np.random.choice(samples, size=int(0.5 * sample_sizes), replace=False)
inference = np.setdiff1d(samples, selection)
responses_selection = {j: responses_train[j][selection] for j in range(ntask)}
predictors_selection = predictors_train[selection,:]
responses_inference = {j: responses_train[j][inference] for j in range(ntask)}
predictors_inference = predictors_train[inference,:]


final_estimates_ds50, final_intervals_ds50, ds50_intervals, all_variables_ds50, significant_variables_ds50, final_err_ds50, pred_r_ds50, coefs_var_ds50 = \
    ds_multi_task_selection_inference(predictors_selection,predictors_inference,predictors_validate,predictors_test, responses_selection, responses_inference,
                                        responses_validate, responses_test, weight_list = np.arange(0.5, 3.5, 0.25))


print(final_err_ds50, "Average testing error per task, data split 50/50")
print(pred_r_ds50, "Predictive r, data split 50/50")
print(np.mean(ds50_intervals),"Mean interval length, data split 50/50")
print(np.std(ds50_intervals), "Sd interval length, data split 50/50")
print(len(ds50_intervals),"Number selected in total")
print(np.sum([len(significant_variables_ds50[i]) for i in range(len(significant_variables_ds50))]),"Sum of significant PCs in total")

match_length_indx2 = {}
start2 = 0
for i in range(ntask):
    match_length_indx2[i] = ds50_intervals[start2:start2+len(all_variables_ds50[i])]
    start2 += len(all_variables_ds50[i])

#Predict g on testing data using 11 estimated task scores
test_task_scores = []
start = 0
for i in range(ntask):
    test_task_scores.append(predictors_test[:, all_variables_ds50[i]].dot(final_estimates_ds50[start:start + len(all_variables_ds50[i])]))
    start += len(all_variables_ds50[i])
test_task_scores = np.asarray(test_task_scores)

pred_g = (test_task_scores.T).dot(weights)
pred_r_general = np.corrcoef(g_test,pred_g)
print("pred r for j based on estimated task scores, data split 50/50",pred_r_general)

common = {i:np.intersect1d(all_variables_rand07[i],all_variables_ds50[i]) for i in range(ntask)}
print("common",common)
common_significant = {i:np.intersect1d(significant_variables_rand07[i],significant_variables_ds50[i]) for i in range(ntask)}
print("common significant",common_significant)
common_lengths = []
#Compute length ratio for shared parameters
for i in range(ntask):
    for predictor in common[i]:
        ratio_length = match_length_indx2[i][np.argwhere(all_variables_ds50[i]==predictor)[0][0]]/match_length_indx[i][np.argwhere(all_variables_rand07[i]==predictor)[0][0]]
        common_lengths.append(ratio_length)

match_length_indx2 = {}
start2 = 0
for i in range(ntask):
    match_length_indx2[i] = ds67_intervals[start2:start2+len(all_variables_ds67[i])]
    start2 += len(all_variables_ds67[i])

common = {i:np.intersect1d(all_variables_rand07[i],all_variables_ds67[i]) for i in range(ntask)}
print("common",common)
common_significant = {i:np.intersect1d(significant_variables_rand07[i],significant_variables_ds67[i]) for i in range(ntask)}
print("common significant",common_significant)
common_lengths_67 = []
for i in range(ntask):
    for predictor in common[i]:
        ratio_length = match_length_indx2[i][np.argwhere(all_variables_ds67[i]==predictor)[0][0]]/match_length_indx[i][np.argwhere(all_variables_rand07[i]==predictor)[0][0]]
        common_lengths_67.append(ratio_length)

def set_boxplot_style(bp, color, linestyle):
    plt.setp(bp['boxes'], color=color, linestyle=linestyle, linewidth=3.5)
    plt.setp(bp['whiskers'], color=color, linestyle=linestyle, linewidth=3.5)
    plt.setp(bp['caps'], color=color, linewidth=2.5)
    plt.setp(bp['medians'], color=color, linewidth=2.5)

def common_format(ax):
    ax.grid(True, which='both', color='#f0f0f0')
    ax.set_xlabel('', fontsize=20)
    return ax


fig = plt.figure(figsize=(24 ,8))
ax1 = fig.add_subplot(133)
plt.sca(ax1)
second = plt.boxplot([common_lengths_67], positions=np.asarray([1.6]), sym='', widths=0.3)
first = plt.boxplot([common_lengths], positions=np.asarray([1]), sym='', widths=0.3)
set_boxplot_style(second, '#984ea3', 'solid')  # colors are from http://colorbrewer2.org/
set_boxplot_style(first, '#984ea3', '--')
plt.xlim(0.7, 1.9)
plt.tight_layout()
plt.plot([], c='#984ea3', label='DS (0.5): MTL (0.7) + SI', linestyle='--', linewidth=2.5)
plt.plot([], c='#984ea3', label='DS (0.67): MTL (0.7) + SI', linewidth=2.5)
plt.legend()
plt.ylabel('Ratio of Lengths for Common Parameters', fontsize=20)
plt.yticks(fontsize=18)

#ax1.set_title("Ratio of Interval Lengths", y=1.01 ,fontsize=24)
ax1.legend(loc='lower left', bbox_to_anchor=(0.0, -.295), fontsize=28)
ax1.set_xticklabels([])
ax1.set_xticks([])

common_format(ax1)
ax1.axhline(y=1.0, color='k', linestyle='--', linewidth=2.5)

ax2 = fig.add_subplot(131)
plt.sca(ax2)
first = plt.boxplot([selective07_intervals], positions=np.asarray([1]), sym='', widths=0.4)
second = plt.boxplot([ds50_intervals], positions=np.asarray([1.5]), sym='', widths=0.4)
third = plt.boxplot([ds67_intervals], positions=np.asarray([2.0]), sym='', widths=0.4)
set_boxplot_style(first, '#377eb8', 'solid')  # colors are from http://colorbrewer2.org/
set_boxplot_style(second, '#4daf4a', '--')
set_boxplot_style(third, '#4daf4a', 'solid')
plt.xlim(0.7, 2.4)
plt.tight_layout()
plt.plot([], c='#377eb8', label='MTL (0.7) + SI', linewidth=2.5)
plt.plot([], c='#4daf4a', label='DS (0.5)', linestyle='--', linewidth=2.5)
plt.plot([], c='#4daf4a', label='DS (0.67)', linewidth=2.5)
plt.legend()
plt.ylabel('Interval Length', fontsize=20)
plt.yticks(fontsize=18)
#ax2.set_title("Distribution of Interval Lengths", y=1.01 ,fontsize=24)
ax2.set_xticklabels([])
ax2.set_xticks([])
common_format(ax2)

ax3 = fig.add_subplot(132)
plt.sca(ax3)
first = plt.boxplot([coefs_var_rand07], positions=np.asarray([1]), sym='', widths=0.4)
second = plt.boxplot([coefs_var_ds50], positions=np.asarray([1.5]), sym='', widths=0.4)
third = plt.boxplot([coefs_var_ds67], positions=np.asarray([2.0]), sym='', widths=0.4)
set_boxplot_style(first, '#377eb8', 'solid')  # colors are from http://colorbrewer2.org/
set_boxplot_style(second, '#4daf4a', '--')
set_boxplot_style(third, '#4daf4a', 'solid')
plt.xlim(0.7, 2.4)
plt.tight_layout()
plt.ylabel('Coefficient of Variation for Estimated Effects', fontsize=18)
plt.yticks(fontsize=18)

#ax3.set_title("Distribution of Coefficient of Variation", y=1.01 ,fontsize=24)
ax3.set_xticklabels([])
ax3.set_xticks([])
common_format(ax3)
plt.tight_layout(pad=0.4, w_pad=0.7, h_pad=1.0)
plt.subplots_adjust(wspace=0.2)
ax2.legend(loc='lower left', bbox_to_anchor=(0.25, -0.25), fontsize=28,ncol=3)
plt.savefig('real_data_lengths_cv.png', bbox_inches='tight')