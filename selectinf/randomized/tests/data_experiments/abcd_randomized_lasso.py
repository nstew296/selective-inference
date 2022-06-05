import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t as tdist
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

for i in range(ntask):
    responses_train[i] = np.genfromtxt('train.csv', delimiter=',')[1:,-ntask+i]
    scale = np.std(responses_train[i])
    responses_train[i] /= scale
    responses_validate[i] = np.genfromtxt('validate.csv', delimiter=',')[1:,-ntask+i]
    responses_validate[i] /=  scale
    responses_test[i] = np.genfromtxt('test.csv', delimiter=',')[1:,-ntask+i]
    responses_test[i] /= scale

#PC loadings and singular values
V = np.genfromtxt('V.csv', delimiter=',')[1:,:]
sv = np.genfromtxt('lambda.csv', delimiter=',')[1:]

#g factor
g_train = np.genfromtxt('train.csv', delimiter=',')[1:,-12]
g_test = np.genfromtxt('test.csv', delimiter=',')[1:,-12]

print("HI")

def _noise(n, df=np.inf):
    if df == np.inf:
        return np.random.standard_normal(n)
    else:
        sd_t = np.std(tdist.rvs(df, size=50000))
    return tdist.rvs(df, size=n) / sd_t

#Generate randomization variable
noise = _noise(predictors_train.shape[1]*ntask)

def rand_single_task_selection_inference(predictor_vars_train,predictor_vars_validate,predictor_vars_test,response_train,
                                        response_validate,response_test,weight_list,noise,rand_scale=0.7):

    sample_sizes = predictor_vars_train.shape[0]
    sample_sizes_validate = predictor_vars_validate.shape[0]
    sample_sizes_test = predictor_vars_test.shape[0]
    nfeatures = predictor_vars_train.shape[1]

    predictions = np.zeros((sample_sizes_test,ntask))
    original_coef_approx = np.zeros((np.shape(V)[0],11))
    significant = {}
    #Setup for post-selection inference
    noise_levels = []
    for i in range(ntask):
        noise_levels.append(np.sqrt(np.sum(np.array(response_train[i] - predictor_vars_train.dot(
            np.linalg.pinv(predictor_vars_train).dot(response_train[i]))) ** 2) / (sample_sizes - nfeatures)))
    final_error_list = []
    final_pred_r_list = []
    for i in range(ntask):
        error_list = []
        #Perform inference for given tuning parameter
        for weight in weight_list:

            W = np.ones(nfeatures) * weight
            single_task_lasso = lasso.gaussian(predictor_vars_train,
                                               response_train[i],
                                               W,
                                               sigma=noise_levels[i],
                                               ridge_term=0.,
                                               randomizer_scale=rand_scale* noise_levels[i])

            initial_omega = np.array(rand_scale * noise_levels[i] * noise[i*nfeatures:(i+1)*nfeatures]).T
            active_signs = single_task_lasso.fit(perturb=initial_omega)
            nonzero = active_signs != 0

            (observed_target, cov_target, cov_target_score, alternatives) = \
                selected_targets(single_task_lasso.loglike, single_task_lasso._W, nonzero, dispersion=noise_levels[i] ** 2)

            if np.sum(nonzero)>0:

                MLE_result, observed_fi = single_task_lasso.selective_MLE(
                    observed_target,
                    cov_target,
                    cov_target_score,
                    level=0.90)[0:2]

                estimate = MLE_result['MLE']

            if (active_signs != 0).sum() > 0:
                error = np.sqrt(np.sum(
                    np.square((response_validate[i] - predictor_vars_validate[:, (active_signs != 0)].dot(
                            estimate)))) / sample_sizes_validate)
            else:
                # If there are no active predictors for any task
                error = np.sqrt((np.linalg.norm(response_validate[i], 2) ** 2) / sample_sizes_validate)

            error_list.append(error)

        min_error = np.argmin(error_list)

        W = np.ones(nfeatures) * weight_list[min_error]
        single_task_lasso = lasso.gaussian(predictor_vars_train,
                                           response_train[i],
                                           W,
                                           sigma=noise_levels[i],
                                           ridge_term=0.,
                                           randomizer_scale=rand_scale)

        initial_omega = np.array(rand_scale * noise_levels[i] * noise[i * nfeatures:(i + 1) * nfeatures]).T
        active_signs = single_task_lasso.fit(perturb=initial_omega)
        nonzero = active_signs != 0

        (observed_target, cov_target, cov_target_score, alternatives) = \
            selected_targets(single_task_lasso.loglike, single_task_lasso._W, nonzero, dispersion=noise_levels[i] ** 2)

        if np.sum(nonzero) > 0:
            MLE_result, observed_fi = single_task_lasso.selective_MLE(
                observed_target,
                cov_target,
                cov_target_score,
                level=0.90)[0:2]

            estimate = MLE_result['MLE']
            intervals = np.asarray(MLE_result[['lower_confidence', 'upper_confidence']])
            significant_variables = [intervals[j, 0] > 0 or intervals[j, 1] < 0 for j in range(np.shape(intervals)[0])]
            significant[i] = np.nonzero(active_signs)[0][significant_variables]
            singular_values = sv[np.nonzero(active_signs)[0]]
            original_coef_approx[:, i] = V[:, np.nonzero(active_signs)[0]].dot(np.divide(estimate, singular_values))

        #Caculate final testing error and predictive r on test set
        if (active_signs != 0).sum() > 0:

            final_error = np.sqrt(np.sum(
                np.square((response_test[i] - predictor_vars_test[:, (active_signs != 0)].dot(
                    estimate)))) / sample_sizes_test)
            predictive_r = (np.corrcoef(response_test[i],predictor_vars_test[:, (active_signs != 0)].dot(
                    estimate))[0,1])
        else:
            final_error = np.sqrt(np.linalg.norm(response_test[i], 2) ** 2 / sample_sizes_test)
            predictive_r = 0

        predictions[:, i] = predictor_vars_test[:, (active_signs != 0)].dot(estimate)

        print(predictive_r)
        final_error_list.append(final_error)
        final_pred_r_list.append(predictive_r)

    return(predictions,final_pred_r_list,significant,original_coef_approx)



#Learn weights for g from 11 task scores
task_scores = np.genfromtxt('train.csv', delimiter=',')[1:,-11:]
y = np.asarray(g_train)
weights = np.linalg.pinv(task_scores).dot(y)


predictions07, final_pred_r07, significant07, original_STL_07 = rand_single_task_selection_inference(predictors_train,predictors_validate,predictors_test, responses_train,
                                        responses_validate, responses_test,np.arange(0,10,.25),noise,rand_scale=0.7)

print(final_pred_r07)
pred_g = predictions07.dot(weights)
pred_r_general = np.corrcoef(g_test,pred_g)
print("pred r for g using task predictions, rand scale 0.7",pred_r_general)

jacard_matrix = np.zeros((11,11))
j_list = []
for i in range(11):
    for j in range(11):
        jacard_matrix[i,j] = round(len(np.intersect1d(significant07[i],significant07[j]))/len(np.union1d(significant07[i],significant07[j])),2)
        if j > i:
            j_list.append(jacard_matrix[i, j])

print(np.mean(j_list),"Mean Jaccard index with STL")

mat = sns.heatmap(jacard_matrix,vmin=0,vmax=1,cmap="viridis_r")
mat.set_xticklabels(['PV','FT','LS','CS','PC','PS','RC','Ravlt-Sd','Ravlt-Ld','Matrix','LMT'],rotation=90)
mat.set_yticklabels(['PV','FT','LS','CS','PC','PS','RC','Ravlt-Sd','Ravlt-Ld','Matrix','LMT'],rotation=0)
fig = mat.get_figure()
fig.tight_layout()
fig.savefig("jaccard_matrix_STL.png")
plt.clf()

np.savetxt("original_approx_STL_07.csv",original_STL_07,delimiter=",")

mat = sns.heatmap(np.around(np.corrcoef(original_STL_07.T),2),vmin=0,vmax=1,cmap="OrRd")
mat.set_xticklabels(['PV','FT','LS','CS','PC','PS','RC','Ravlt-Sd','Ravlt-Ld','Matrix','LMT'],rotation=90)
mat.set_yticklabels(['PV','FT','LS','CS','PC','PS','RC','Ravlt-Sd','Ravlt-Ld','Matrix','LMT'],rotation=0)
fig = mat.get_figure()
fig.tight_layout()
fig.savefig("correlation_STL.png")

rho_list = []
for i in range(11):
    for j in range(11):
        if j>i:
            rho_list.append(np.around(np.corrcoef(original_STL_07.T),2)[i,j])

print(np.mean(rho_list),"Mean correlation between coefficients with STL")