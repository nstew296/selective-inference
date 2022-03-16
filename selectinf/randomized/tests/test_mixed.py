import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import norm as ndist
from scipy.stats import t as tdist

from selectinf.randomized.multitask_lasso import multi_task_lasso
from selectinf.tests.instance import gaussian_multitask_instance
from selectinf.randomized.lasso import lasso, selected_targets


def test_multitask_lasso_selective_inference(predictor_vars_train,
                                response_vars_train,
                                predictor_vars_test,
                                response_vars_test,
                                beta,
                                gaussian_noise,
                                sigma,
                                weight=1.0,
                                randomizer_scale=0.7):
    ntask = len(predictor_vars_train.keys())
    nsamples_test = np.asarray([np.shape(predictor_vars_test[i])[0] for i in range(ntask)])
    p = np.shape(beta)[0]

    feature_weight = weight * np.ones(p)
    randomizer_scales = randomizer_scale * np.array([sigma[i] for i in range(ntask)])
    initial_omega = np.array(
        [randomizer_scales[i] * gaussian_noise[p*i:p*(i+1)] for i in range(ntask)]).T


    try:
        multi_lasso = multi_task_lasso.gaussian(predictor_vars_train,
                                                response_vars_train,
                                                feature_weight,
                                                ridge_term=None,
                                                randomizer_scales=randomizer_scales)

        active_signs = multi_lasso.fit(perturbations=initial_omega)
        dispersions = sigma ** 2
        estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(
            dispersions=dispersions)

    except:
        active_signs = np.asarray([])


    coverage = []
    pivot = []

    if (active_signs != 0).sum() > 0:

        beta_target = []

        for i in range(ntask):
            X, y = multi_lasso.loglikes[i].data
            beta_target.extend(np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(X.dot(beta[:, i])))

        beta_target = np.asarray(beta_target)
        pivot_ = ndist.cdf((estimate - beta_target) / np.sqrt(np.diag(observed_info_mean)))
        pivot = 2 * np.minimum(pivot_, 1. - pivot_)

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        error = 0
        idx = 0
        for j in range(ntask):
            idx_new = np.sum(active_signs[:, j] != 0)
            if idx_new == 0:
                error += (np.sum(np.square(response_vars_test[j]))) / nsamples_test[j]
                continue
            error += (np.sum(
                np.square((response_vars_test[j] - (predictor_vars_test[j])[:, (active_signs[:, j] != 0)].dot(
                    estimate[idx:idx + idx_new]))))) / nsamples_test[j]
            idx = idx + idx_new

    else:
        error = 0
        for j in range(ntask):
            error += (np.linalg.norm(response_vars_test[j], 2) ** 2) / nsamples_test[j]
        intervals = np.asarray([[np.nan, np.nan]])

    # Compute snesitivity and specificity after selection
    # true_active = np.transpose(np.nonzero(beta))
    # selected_active = np.transpose(np.nonzero(active_signs))
    # num_true_positive = np.sum(x in true_active.tolist() for x in selected_active.tolist())
    # num_false_positive = np.sum(x not in true_active.tolist() for x in selected_active.tolist())
    # num_positive = np.shape(true_active)[0]
    # num_negative = np.shape(beta)[0]*np.shape(beta)[1]-num_positive
    # sensitivity = np.float(num_true_positive)/np.float(num_positive)
    # specificity = 1.0 - np.float(num_false_positive)/np.float(num_negative)

    # Compute sensitivity and specificity after inference
    true_active = np.transpose(np.nonzero(np.transpose(beta)))
    num_positive = np.shape(true_active)[0]
    if (active_signs != 0).sum() > 0:
        selected_active = np.transpose(np.nonzero(np.transpose(active_signs)))
        true_positive_selected = [x in true_active.tolist() for x in selected_active.tolist()]
        num_true_positive_inference = np.sum(
            [true_positive_selected[i] * (intervals[i, 1] < 0 or intervals[i, 0] > 0) for i in
             range(len(true_positive_selected))])
        num_false_positive_inference = np.sum([(intervals[i, 1] < 0 or intervals[i, 0] > 0) for i in
                                               range(len(true_positive_selected))]) - num_true_positive_inference
    else:
        num_true_positive_inference = 0
        num_false_positive_inference = 0
    num_negative = np.shape(beta)[0] * np.shape(beta)[1] - num_positive
    sensitivity_inference = float(num_true_positive_inference) / np.maximum(float(num_positive), 1)
    specificity_inference = 1.0 - float(num_false_positive_inference) / np.maximum(float(num_negative), 1)

    return np.asarray(coverage), intervals[:, 1] - intervals[:,
                                                   0], pivot, sensitivity_inference, specificity_inference, error

def test_multitask_lasso_naive(predictor_vars_train,
                                      response_vars_train,
                                      predictor_vars_test,
                                      response_vars_test,
                                      beta,
                                      sigma,
                                      weight = 1.0,):

    ntask = len(predictor_vars_train.keys())
    nsamples_test = np.asarray([np.shape(predictor_vars_test[i])[0] for i in range(ntask)])
    p = np.shape(beta)[0]

    feature_weight = weight * np.ones(p)
    sigmas_ = sigma
    perturbations = np.zeros((p, ntask))

    try:
        multi_lasso = multi_task_lasso.gaussian(predictor_vars_train,
                                            response_vars_train,
                                            feature_weight,
                                            ridge_term=None,
                                            randomizer_scales=1. * sigmas_,
                                            perturbations=perturbations)
        active_signs = multi_lasso.fit()

    except:

        active_signs = np.asarray([])


    dispersions = sigma ** 2

    coverage = []
    pivot = []
    CIs = [[0,0]]

    error = 0
    if (active_signs != 0).sum() > 0:

        for i in range(ntask):
            X, y = multi_lasso.loglikes[i].data
            beta_target = np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(X.dot(beta[:, i]))
            Qfeat = np.linalg.inv(X[:, (active_signs[:, i] != 0)].T.dot(X[:, (active_signs[:, i] != 0)]))
            observed_target = np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(y)
            cov_target = Qfeat * dispersions[i]
            alpha = 1. - 0.90
            quantile = ndist.ppf(1 - alpha / 2.)
            intervals = np.vstack([observed_target - quantile * np.sqrt(np.diag(cov_target)),
                                   observed_target + quantile * np.sqrt(np.diag(cov_target))]).T
            CIs = np.vstack([CIs,intervals])
            coverage.extend((beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1]))
            pivot_ = ndist.cdf((observed_target - beta_target) / np.sqrt(np.diag(cov_target)))
            pivot.extend(2 * np.minimum(pivot_, 1. - pivot_))

            idx_new = np.sum(active_signs[:, i] != 0)
            if idx_new == 0:
                error += (0.5 * np.sum(np.square(response_vars_test[i])))/nsamples_test[i]
                continue
            observed_target = np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(y)
            error += (0.5 * np.sum(np.square(
                response_vars_test[i] - (predictor_vars_test[i])[:, (active_signs[:, i] != 0)].dot(
                    observed_target))))/nsamples_test[i]

    else:
        error=0
        for j in range(ntask):
            error += (0.5 * np.linalg.norm(response_vars_test[j], 2) ** 2)/nsamples_test[j]
        CIs = np.asarray([[0, 0],[np.nan,np.nan]])


    # Compute snesitivity and specificity after inference
    true_active = np.transpose(np.nonzero(np.transpose(beta)))
    num_positive = np.shape(true_active)[0]
    if (active_signs != 0).sum() > 0:
        selected_active = np.transpose(np.nonzero(np.transpose(active_signs)))
        true_positive_selected = [x in true_active.tolist() for x in selected_active.tolist()]
        num_true_positive_inference = np.sum(
            [true_positive_selected[i] * (CIs[i + 1, 1] < 0 or CIs[i + 1, 0] > 0) for i in
             range(len(true_positive_selected))])
        num_false_positive_inference = np.sum([(CIs[i + 1, 1] < 0 or CIs[i + 1, 0] > 0) for i in range(
            len(true_positive_selected))]) - num_true_positive_inference
    else:
        num_true_positive_inference = 0
        num_false_positive_inference = 0
    num_negative = np.shape(beta)[0] * np.shape(beta)[1] - num_positive
    sensitivity_inference = float(num_true_positive_inference) / np.maximum(float(num_positive),1)
    specificity_inference = 1.0 - float(num_false_positive_inference) / np.maximum(float(num_negative),1)


    return np.asarray(coverage), CIs[1:, 1] - CIs[1:, 0], pivot, sensitivity_inference, specificity_inference, error


def test_multitask_lasso_data_splitting(predictor_vars_train,
                                      response_vars_train,
                                      predictor_vars_test,
                                      response_vars_test,
                                      beta,
                                      sigma,
                                      weight = 1.0,
                                      split = 0.5):

    ntask = len(predictor_vars_train.keys())
    nsamples = np.asarray([np.shape(predictor_vars_train[i])[0] for i in range(ntask)])
    nsamples_test = np.asarray([np.shape(predictor_vars_test[i])[0] for i in range(ntask)])
    p = np.shape(beta)[0]

    samples = np.arange(int(nsamples[0]))
    selection = np.random.choice(samples, size=int(split * nsamples[0]), replace=False)
    inference = np.setdiff1d(samples, selection)
    response_vars_selection = {j: response_vars_train[j][selection] for j in range(ntask)}
    predictor_vars_selection = {j: predictor_vars_train[j][selection] for j in range(ntask)}
    response_vars_inference = {j: response_vars_train[j][inference] for j in range(ntask)}
    predictor_vars_inference = {j: predictor_vars_train[j][inference] for j in range(ntask)}

    feature_weight = weight * np.ones(p)
    sigmas_ = sigma
    perturbations = np.zeros((p, ntask))


    try:
        multi_lasso = multi_task_lasso.gaussian(predictor_vars_selection,
                                            response_vars_selection,
                                            feature_weight,
                                            ridge_term=None,
                                            randomizer_scales=1. * sigmas_,
                                            perturbations=perturbations)

        active_signs = multi_lasso.fit()

    except:

        active_signs = np.asarray([])


    dispersions = sigma ** 2
    coverage = []
    pivot = []
    CIs = [[0,0]]

    if (active_signs != 0).sum() > 0:

        error = 0

        for i in range(ntask):
            X = predictor_vars_inference[i]
            y = response_vars_inference[i]
            beta_target = np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(X.dot(beta[:, i]))
            Qfeat = np.linalg.inv(X[:, (active_signs[:, i] != 0)].T.dot(X[:, (active_signs[:, i] != 0)]))
            observed_target = np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(y)
            cov_target = Qfeat * dispersions[i]
            alpha = 1. - 0.90
            quantile = ndist.ppf(1 - alpha / 2.)
            intervals = np.vstack([observed_target - quantile * np.sqrt(np.diag(cov_target)),
                                   observed_target + quantile * np.sqrt(np.diag(cov_target))]).T
            coverage.extend((beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1]))
            pivot_ = ndist.cdf((observed_target - beta_target) / np.sqrt(np.diag(cov_target)))
            pivot.extend(2 * np.minimum(pivot_, 1. - pivot_))
            CIs = np.vstack([CIs,intervals])

            idx_new = np.sum(active_signs[:, i] != 0)
            if idx_new == 0:
                error += (0.5 * np.sum(np.square(response_vars_test[i])))/nsamples_test[i]
                continue
            error += (0.5 * np.sum(np.square(
                response_vars_test[i] - (predictor_vars_test[i])[:, (active_signs[:, i] != 0)].dot(
                    observed_target))))/nsamples_test[i]

    else:
        error=0
        for j in range(ntask):
            error += (0.5 * np.linalg.norm(response_vars_test[j], 2) ** 2)/nsamples_test[j]
        CIs = np.asarray([[0, 0],[np.nan,np.nan]])


    # Compute snesitivity and specificity after inference
    true_active = np.transpose(np.nonzero(np.transpose(beta)))
    num_positive = np.shape(true_active)[0]
    if (active_signs != 0).sum() > 0:
        selected_active = np.transpose(np.nonzero(np.transpose(active_signs)))
        true_positive_selected = [x in true_active.tolist() for x in selected_active.tolist()]
        num_true_positive_inference = np.sum(
          [true_positive_selected[i] * (CIs[i+1, 1] < 0 or CIs[i+1, 0] > 0) for i in
            range(len(true_positive_selected))])
        num_false_positive_inference = np.sum([(CIs[i+1, 1] < 0 or CIs[i+1, 0] > 0) for i in range(
            len(true_positive_selected))]) - num_true_positive_inference
    else:
        num_true_positive_inference = 0
        num_false_positive_inference = 0
    num_negative = np.shape(beta)[0] * np.shape(beta)[1] - num_positive
    sensitivity_inference = float(num_true_positive_inference) / float(num_positive)
    specificity_inference = 1.0 - float(num_false_positive_inference) / float(num_negative)

    # Compute snesitivity and specificity after selection
    # true_active = np.transpose(np.nonzero(beta))
    # selected_active = np.transpose(np.nonzero(active_signs))
    # num_true_positive = np.sum(x in true_active.tolist() for x in selected_active.tolist())
    # num_false_positive = np.sum(x not in true_active.tolist() for x in selected_active.tolist())
    # num_positive = np.shape(true_active)[0]
    # num_negative = np.shape(beta)[0] * np.shape(beta)[1] - num_positive
    # sensitivity = np.float(num_true_positive) / np.float(num_positive)
    # specificity = 1.0 - np.float(num_false_positive) / np.float(num_negative)


    return np.asarray(coverage), CIs[1:, 1] - CIs[1:, 0], pivot, sensitivity_inference, specificity_inference, error


def test_single_task_lasso_selective_inference(predictor_vars_train,
                                       response_vars_train,
                                       predictor_vars_test,
                                       response_vars_test,
                                       beta,
                                       gaussian_noise,
                                       sigma,
                                       weight,
                                       randomizer_scale=1.0):
    ntask = len(predictor_vars_train.keys())
    nsamples_test = np.asarray([np.shape(predictor_vars_test[i])[0] for i in range(ntask)])
    p = np.shape(beta)[0]

    coverage = []
    pivot = []
    CIs = [[0, 0]]
    error = 0
    selected_active = []

    for i in range(ntask):

        W = np.ones(p) * weight
        single_task_lasso = lasso.gaussian(predictor_vars_train[i],
                                           response_vars_train[i],
                                           W,
                                           sigma=sigma[i],
                                           ridge_term=0.,
                                           randomizer_scale=randomizer_scale)

        initial_omega = np.array(randomizer_scale * sigma[i] * gaussian_noise[p*i:p*(i+1)]).T
        signs = single_task_lasso.fit(perturb=initial_omega)
        nonzero = signs != 0

        (observed_target, cov_target, cov_target_score, alternatives) = \
            selected_targets(single_task_lasso.loglike, single_task_lasso._W, nonzero, dispersion=sigma[i] ** 2)

        try:
            MLE_result, observed_info_mean = single_task_lasso.selective_MLE(
                observed_target,
                cov_target,
                cov_target_score,
                level=0.90)[0:2]

            final_estimator = MLE_result['MLE']
            intervals = np.asarray(MLE_result[['lower_confidence', 'upper_confidence']])
            beta_target = np.linalg.pinv(predictor_vars_train[i][:, nonzero]).dot(
                predictor_vars_train[i].dot(beta[:, i]))

            coverage.extend(np.asarray(beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1]))
            pivot_ = ndist.cdf((final_estimator - beta_target) / np.sqrt(np.diag(observed_info_mean)))
            pivot.extend(2 * np.minimum(pivot_, 1. - pivot_))
            CIs = np.vstack([CIs, intervals])
            selected_active.extend([[i, j] for j in np.nonzero(signs)[0]])

        except:
            pass

        idx_new = np.sum(signs != 0)
        if idx_new == 0:
            error += (np.sum(np.square(response_vars_test[i]))) / nsamples_test[i]
            continue
        error += (np.sum(np.square(
            response_vars_test[i] - (predictor_vars_test[i])[:, nonzero].dot(
                final_estimator)))) / nsamples_test[i]

    true_active = np.transpose(np.nonzero(np.transpose(beta)))
    num_positive = np.shape(true_active)[0]
    if selected_active != []:
        true_positive_selected = [x in true_active.tolist() for x in selected_active]
        num_true_positive_inference = np.sum(
            [true_positive_selected[i] * (CIs[i + 1, 1] < 0 or CIs[i + 1, 0] > 0) for i in
             range(len(true_positive_selected))])
        num_false_positive_inference = np.sum([(CIs[i + 1, 1] < 0 or CIs[i + 1, 0] > 0) for i in range(
            len(true_positive_selected))]) - num_true_positive_inference
    else:
        CIs = np.asarray([[0, 0], [np.nan, np.nan]])
        num_true_positive_inference = 0
        num_false_positive_inference = 0
    num_negative = np.shape(beta)[0] * np.shape(beta)[1] - num_positive
    sensitivity_inference = float(num_true_positive_inference) / float(num_positive)
    specificity_inference = 1.0 - float(num_false_positive_inference) / float(num_negative)

    return np.asarray(coverage), CIs[1:, 1] - CIs[1:, 0], np.asarray(
        pivot), sensitivity_inference, specificity_inference, error

np.random.seed(5)

ntask = 5
nsamples = 500 * np.ones(ntask)
nsamples_test = 500 * np.ones(ntask)
p = 100
global_sparsity = 0.9
task_sparsity = 0.2
sigma = 1. * np.ones(ntask)
signal = [1.0, 3.0]
signal_fac = np.array(signal)
rhos = 0.3 * np.ones(ntask)
nsamples = nsamples.astype(int)
nsamples_test = nsamples_test.astype(int)
signal = np.sqrt(signal_fac * 2 * np.log(p))

response_vars_train, predictor_vars_train, response_vars_test, predictor_vars_test, beta, gaussian_noise = gaussian_multitask_instance(
        ntask,
        nsamples,
        nsamples_test,
        p,
        global_sparsity,
        task_sparsity,
        sigma,
        signal,
        rhos,
        random_signs=True,
        equicorrelated=True)[:6]

# Print SNR, PVE
SIG = np.full((p, p), 0.3)
np.fill_diagonal(SIG, 1.0)
SNR = beta.T.dot(SIG.dot(beta)) / 1000
SNR = np.diag(SNR)
print(SNR, "SNR")
print(SNR / (1 + SNR), "PVE")

length_path = 8
nsim = 100
lambdamin = 0.5
lambdamax = 3.5
feature_weight_list = np.arange(lambdamin, lambdamax, (lambdamax - lambdamin) / (length_path))
print(feature_weight_list)

#Track results across lambda path with nested list
selective_lengths = []
selective_lengths2 = []
naive_lengths = []
ds_lengths = []
ds_lengths2 = []
single_selective_lengths = []
single_selective_lengths2 = []

selective_coverage = []
selective_coverage2 = []
naive_coverage = []
ds_coverage = []
ds_coverage2 = []
single_selective_coverage = []
single_selective_coverage2 = []

selective_sensitivity = []
selective_sensitivity2 = []
naive_sensitivity = []
ds_sensitivity = []
ds_sensitivity2 = []
single_task_sensitivity = []
single_task_sensitivity2 = []

selective_specificity = []
selective_specificity2 = []
naive_specificity = []
ds_specificity = []
ds_specificity2 = []
single_task_specificity = []
single_task_specificity2 = []

selective_error = []
selective_error2 = []
naive_error = []
ds_error = []
ds_error2 = []
single_selective_error = []
single_selective_error2 = []

for i in range(len(feature_weight_list)):
    weight = feature_weight_list[i]
    #For each weight, run n=nsim simulations
    #Lists to record coverage, length, pivots, sensitivity, specificity, and hold-out error
    # across repetitions of same simulation
    cov = []
    len1 = []
    pivots = []
    sensitivity_list = []
    specificity_list = []
    test_error_list = []

    cov2 = []
    len2 = []
    pivots2 = []
    sensitivity_list2 = []
    specificity_list2 = []
    test_error_list2 = []

    cov_naive = []
    len_naive = []
    pivots_naive = []
    sensitivity_list_naive = []
    specificity_list_naive = []
    naive_test_error_list = []

    cov_data_splitting = []
    len_data_splitting = []
    pivots_data_splitting = []
    sensitivity_list_ds = []
    specificity_list_ds = []
    data_splitting_test_error_list = []

    cov_data_splitting2 = []
    len_data_splitting2 = []
    pivots_data_splitting2 = []
    sensitivity_list_ds2 = []
    specificity_list_ds2 = []
    data_splitting_test_error_list2 = []

    cov_single_task_selective = []
    len_single_task_selective = []
    sensitivity_list_single_task_selective = []
    specificity_list_single_task_selective = []
    single_task_selective_test_error_list = []

    cov_single_task_selective2 = []
    len_single_task_selective2 = []
    sensitivity_list_single_task_selective2 = []
    specificity_list_single_task_selective2 = []
    single_task_selective_test_error_list2 = []

    for n in range(nsim):

        # For each iteration, generate independent training errors, testing errors, and randomization variables
        # Compute new responses from the new errors
        if n >= 1:

            def _noise(n, df=np.inf):
                if df == np.inf:
                    return np.random.standard_normal(n)
                else:
                    sd_t = np.std(tdist.rvs(df, size=50000))
                return tdist.rvs(df, size=n) / sd_t

            gaussian_noise = _noise(nsamples.sum() + nsamples_test.sum() + p * ntask)
            response_vars_train = {}
            response_vars_test = {}
            nsamples_train_cumsum = np.cumsum([nsamples[i] for i in range(ntask)])
            nsamples_test_cumsum = np.cumsum([nsamples_test[i] for i in range(ntask)])

            for i in range(ntask):
                if i == 0:
                    response_vars_train[i] = (predictor_vars_train[i].dot(beta[:, i] / sigma[i])
                                              + gaussian_noise[:nsamples_train_cumsum[i]]) * sigma[i]
                    response_vars_test[i] = (predictor_vars_test[i].dot(beta[:, i] / sigma[i])
                                             + gaussian_noise[nsamples.sum():nsamples.sum()
                                                                             + nsamples_test_cumsum[i]]) * sigma[i]
                else:
                    response_vars_train[i] = (predictor_vars_train[i].dot(beta[:, i] / sigma[i])
                                              + gaussian_noise[
                                                nsamples_train_cumsum[i - 1]: nsamples_train_cumsum[i]]) * \
                                             sigma[i]
                    response_vars_test[i] = (predictor_vars_test[i].dot(beta[:, i] / sigma[i]) +
                                             gaussian_noise[
                                             nsamples.sum() + nsamples_test_cumsum[i - 1]: nsamples.sum() +
                                                                                           nsamples_test_cumsum[
                                                                                               i]]) * sigma[i]
            gaussian_noise = gaussian_noise[nsamples.sum() + nsamples_test.sum():]

        print(n, "n sim")

        coverage, length, pivot, sns, spc, err = test_multitask_lasso_selective_inference(predictor_vars_train,
                                                                         response_vars_train,
                                                                         predictor_vars_test,
                                                                         response_vars_test,
                                                                         beta,
                                                                         gaussian_noise,
                                                                         sigma,
                                                                         weight=weight,
                                                                         randomizer_scale= 0.7)

        if list(coverage):
            cov.append(np.mean(np.asarray(coverage)))
            len1.extend(length)
            pivots.extend(pivot)
        sensitivity_list.append(sns)
        specificity_list.append(spc)
        test_error_list.append(err)

        coverage2, length2, pivot2, sns2, spc2, err2 = test_multitask_lasso_selective_inference(predictor_vars_train,
                                                                             response_vars_train,
                                                                             predictor_vars_test,
                                                                             response_vars_test,
                                                                             beta,
                                                                             gaussian_noise,
                                                                             sigma,
                                                                             weight=weight,
                                                                             randomizer_scale=1.0)

        if list(coverage2):
            cov2.append(np.mean(np.asarray(coverage2)))
            len2.extend(length2)
            pivots2.extend(pivot2)
        sensitivity_list2.append(sns2)
        specificity_list2.append(spc2)
        test_error_list2.append(err2)


        coverage_naive, length_naive, pivot_naive, sensitivity_naive, specificity_naive, naive_err = test_multitask_lasso_naive(predictor_vars_train,
                                                                             response_vars_train,
                                                                             predictor_vars_test,
                                                                             response_vars_test,
                                                                             beta,
                                                                             sigma,
                                                                             weight)

        if list(coverage_naive):
            cov_naive.append(np.mean(np.asarray(coverage_naive)))
            len_naive.extend(length_naive)
            pivots_naive.extend(pivot_naive)
        sensitivity_list_naive.append(sensitivity_naive)
        specificity_list_naive.append(specificity_naive)
        naive_test_error_list.append(naive_err)


        coverage_data_splitting, length_data_splitting, pivot_data_splitting, sns_ds, spc_ds, error_ds = test_multitask_lasso_data_splitting(predictor_vars_train,
                                                                             response_vars_train,
                                                                             predictor_vars_test,
                                                                             response_vars_test,
                                                                             beta,
                                                                             sigma,
                                                                             weight,
                                                                             split = 0.67)

        if list(coverage_data_splitting):
            cov_data_splitting.append(np.mean(np.asarray(coverage_data_splitting)))
            len_data_splitting.extend(length_data_splitting)
            pivots_data_splitting.extend(pivot_data_splitting)
        sensitivity_list_ds.append(sns_ds)
        specificity_list_ds.append(spc_ds)
        data_splitting_test_error_list.append(error_ds)

        coverage_data_splitting2, length_data_splitting2, pivot_data_splitting2, sns_ds2, spc_ds2, error_ds2 = test_multitask_lasso_data_splitting(
                                                                            predictor_vars_train,
                                                                            response_vars_train,
                                                                            predictor_vars_test,
                                                                            response_vars_test,
                                                                            beta,
                                                                            sigma,
                                                                            weight,
                                                                            split=0.5)

        if list(coverage_data_splitting2):
            cov_data_splitting2.append(np.mean(np.asarray(coverage_data_splitting2)))
            len_data_splitting2.extend(length_data_splitting2)
            pivots_data_splitting2.extend(pivot_data_splitting2)
        sensitivity_list_ds2.append(sns_ds2)
        specificity_list_ds2.append(spc_ds2)
        data_splitting_test_error_list2.append(error_ds2)

        coverage_single_task_selective, length_single_task_selective, pivot_single_task_selective, sns_single_task, spc_single_task, err_single_selective = test_single_task_lasso_selective_inference(predictor_vars_train,
                                      response_vars_train,
                                      predictor_vars_test,
                                      response_vars_test,
                                      beta,
                                      gaussian_noise,
                                      sigma,
                                      weight,
                                      randomizer_scale = 0.7)


        if list(coverage_single_task_selective):
            cov_single_task_selective.append(np.mean(np.asarray(coverage_single_task_selective)))
            len_single_task_selective.extend(length_single_task_selective)
        sensitivity_list_single_task_selective.append(sns_single_task)
        specificity_list_single_task_selective.append(spc_single_task)
        single_task_selective_test_error_list.append(err_single_selective)

        coverage_single_task_selective2, length_single_task_selective2, pivot_single_task_selective2, sns_single_task2, spc_single_task2, err_single_selective2 = test_single_task_lasso_selective_inference(
            predictor_vars_train,
            response_vars_train,
            predictor_vars_test,
            response_vars_test,
            beta,
            gaussian_noise,
            sigma,
            weight,
            randomizer_scale=1.0)

        if list(coverage_single_task_selective2):
            cov_single_task_selective2.append(np.mean(np.asarray(coverage_single_task_selective2)))
            len_single_task_selective2.extend(length_single_task_selective2)
        sensitivity_list_single_task_selective2.append(sns_single_task2)
        specificity_list_single_task_selective2.append(spc_single_task2)
        single_task_selective_test_error_list2.append(err_single_selective2)

        print("iteration completed ", n)

    selective_coverage.append(cov)
    selective_coverage2.append(cov2)
    naive_coverage.append(cov_naive)
    ds_coverage.append(cov_data_splitting)
    ds_coverage2.append(cov_data_splitting2)
    single_selective_coverage.append(cov_single_task_selective)
    single_selective_coverage2.append(cov_single_task_selective2)

    selective_lengths.append(len1)
    selective_lengths2.append(len2)
    naive_lengths.append(len_naive)
    ds_lengths.append(len_data_splitting)
    ds_lengths2.append(len_data_splitting2)
    single_selective_lengths.append(len_single_task_selective)
    single_selective_lengths2.append(len_single_task_selective2)

    selective_sensitivity.append(sensitivity_list)
    selective_sensitivity2.append(sensitivity_list2)
    naive_sensitivity.append(sensitivity_list_naive)
    ds_sensitivity.append(sensitivity_list_ds)
    ds_sensitivity2.append(sensitivity_list_ds2)
    single_task_sensitivity.append(sensitivity_list_single_task_selective)
    single_task_sensitivity2.append(sensitivity_list_single_task_selective2)

    selective_specificity.append(specificity_list)
    selective_specificity2.append(specificity_list2)
    naive_specificity.append(specificity_list_naive)
    ds_specificity.append(specificity_list_ds)
    ds_specificity2.append(specificity_list2)
    single_task_specificity.append(specificity_list_single_task_selective)
    single_task_specificity2.append(specificity_list_single_task_selective2)

    selective_error.append(np.mean(np.asarray(test_error_list)))
    selective_error2.append(np.mean(np.asarray(test_error_list2)))
    naive_error.append(np.mean(np.asarray(naive_test_error_list)))
    ds_error.append(np.mean(np.asarray(data_splitting_test_error_list)))
    ds_error2.append(np.mean(np.asarray(data_splitting_test_error_list2)))
    single_selective_error.append(np.mean(np.asarray(single_task_selective_test_error_list)))
    single_selective_error2.append(np.mean(np.asarray(single_task_selective_test_error_list2)))

#Converyt sensitivity and specificity to F1
selective_f1 = []
selective2_f1 = []
ds_f1 = []
ds2_f1 = []
single_selective_f1 = []
single_selective2_f1 = []
positive = (1. - task_sparsity) * (1. - global_sparsity) * ntask * p
negative = 5 * 100 - positive
for i in range(length_path):
    selective_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(selective_specificity)[i, :][n] for n in range(nsim)]), np.asarray(selective_sensitivity)[i, :]]).T


    selective_f1.append(np.asarray([2.0 * selective_tp_fp_mat[n, 1] * positive / (2.0 * selective_tp_fp_mat[n, 1] * positive +
                                                       selective_tp_fp_mat[n, 0] * negative + (1.0 -
                                                       selective_tp_fp_mat[n, 1]) * positive) for n in range(nsim)]))

    selective2_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(selective_specificity2)[i, :][n] for n in range(nsim)]),
         np.asarray(selective_sensitivity2)[i, :]]).T

    selective2_f1.append(
        np.asarray([2.0 * selective2_tp_fp_mat[n, 1] * positive / (2.0 * selective2_tp_fp_mat[n, 1] * positive +
                                                                  selective2_tp_fp_mat[n, 0] * negative + (1.0 -
                                                                                                          selective2_tp_fp_mat[
                                                                                                              n, 1]) * positive)
                    for n in range(nsim)]))

    ds_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(ds_specificity)[i, :][n] for n in range(nsim)]),
         np.asarray(ds_sensitivity)[i, :]]).T

    ds_f1.append(
        np.asarray([2.0 * ds_tp_fp_mat[n, 1] * positive / (2.0 * ds_tp_fp_mat[n, 1] * positive +
                                                                  ds_tp_fp_mat[n, 0] * negative + (1.0 -
                                                                                                          ds_tp_fp_mat[
                                                                                                              n, 1]) * positive)
                    for n in range(nsim)]))

    ds2_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(ds_specificity2)[i, :][n] for n in range(nsim)]),
         np.asarray(ds_sensitivity2)[i, :]]).T

    ds2_f1.append(
        np.asarray([2.0 * ds2_tp_fp_mat[n, 1] * positive / (2.0 * ds2_tp_fp_mat[n, 1] * positive +
                                                           ds2_tp_fp_mat[n, 0] * negative + (1.0 -
                                                                                            ds2_tp_fp_mat[
                                                                                                n, 1]) * positive)
                    for n in range(nsim)]))

    single_selective_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(single_task_specificity)[i, :][n] for n in range(nsim)]),
         np.asarray(single_task_sensitivity)[i, :]]).T

    single_selective_f1.append(
        np.asarray([2.0 * single_selective_tp_fp_mat[n, 1] * positive / (2.0 * single_selective_tp_fp_mat[n, 1] * positive +
                                                           single_selective_tp_fp_mat[n, 0] * negative + (1.0 -
                                                                                            single_selective_tp_fp_mat[
                                                                                                n, 1]) * positive)
                    for n in range(nsim)]))

    single_selective2_tp_fp_mat = np.asarray(
        [np.asarray([1.0 - np.asarray(single_task_specificity2)[i, :][n] for n in range(nsim)]),
         np.asarray(single_task_sensitivity2)[i, :]]).T

    single_selective2_f1.append(
        np.asarray([2.0 * single_selective2_tp_fp_mat[n, 1] * positive / (
                    2.0 * single_selective2_tp_fp_mat[n, 1] * positive +
                    single_selective2_tp_fp_mat[n, 0] * negative + (1.0 -
                                                                   single_selective2_tp_fp_mat[
                                                                       n, 1]) * positive)
                    for n in range(nsim)]))

def set_box_color(bp, color, linestyle):
    plt.setp(bp['boxes'], color=color, linestyle=linestyle,linewidth=2)
    plt.setp(bp['whiskers'], color=color, linestyle=linestyle,linewidth=2)
    plt.setp(bp['caps'], color=color,linewidth=2)
    plt.setp(bp['medians'], color=color,linewidth=2)

length = len(feature_weight_list)

fig = plt.figure(figsize=(17, 21))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

plt.sca(ax1)
first = plt.boxplot(naive_coverage, positions=np.array(range(length)) * 3, sym='', widths=0.3)
second = plt.boxplot(selective_coverage,positions=np.array(range(length)) * 3 + 0.3, sym='', widths=0.3)
third = plt.boxplot(selective_coverage2,positions=np.array(range(length)) * 3 + .6, sym='', widths=0.3)
fourth = plt.boxplot(ds_coverage, positions=np.array(range(length)) * 3 + .9, sym='', widths=0.3)
fifth = plt.boxplot(ds_coverage2, positions=np.array(range(length)) * 3 + 1.2, sym='', widths=0.3)
sixth = plt.boxplot(single_selective_coverage, positions=np.array(range(length)) * 3 + 1.5, sym='', widths=0.3)
seventh = plt.boxplot(single_selective_coverage2, positions=np.array(range(length)) * 3 + 1.8, sym='', widths=0.3)
set_box_color(first, '#D7191C', 'solid')
set_box_color(second, '#2b8cbe', 'solid')  # colors are from http://colorbrewer2.org/
set_box_color(third, '#6baed6', '--')
set_box_color(fourth, '#238443', 'solid')
set_box_color(fifth, '#31a354', '--')
set_box_color(sixth, '#fd8d3c', 'solid')
set_box_color(seventh, '#feb24c', '--')
plt.xticks(range(1, (length) * 3 + 1, 3), [round(num, 2) for num in feature_weight_list],fontsize=14)
plt.xlim(-1, (length - 1) * 3 + 3)
plt.plot(np.argmin(selective_error) * 3 +.3, 1.01, 'o', c='#2b8cbe')
plt.plot(np.argmin(selective_error2) * 3 + .6, 1.01, 'o', c='#6baed6')
#plt.plot(np.argmin(naive_error) * 3, 1.01, 'o', c='#D7191C')
plt.plot(np.argmin(ds_error) * 3 + .9, 1.01, 'o', c='#238443')
plt.plot(np.argmin(ds_error2) * 3 + 1.2, 1.01, 'o', c='#31a354')
plt.plot(np.argmin(single_selective_error) * 3 + 1.5, 1.01, 'o', c='#fd8d3c')
plt.plot(np.argmin(single_selective_error2) * 3 + 1.8, 1.01, 'o', c='#feb24c')
plt.tight_layout()
plt.ylabel('Mean Coverage per Simulation', fontsize=16)
plt.yticks(fontsize=14)

plt.sca(ax2)
second = plt.boxplot(selective_lengths, positions=np.array(range(length)) * 3, sym='', widths=0.3)
third = plt.boxplot(selective_lengths2, positions=np.array(range(length)) * 3 + .3, sym='', widths=0.3)
fourth = plt.boxplot(ds_lengths, positions=np.array(range(length)) * 3 + .6, sym='', widths=0.3)
fifth = plt.boxplot(ds_lengths2, positions=np.array(range(length)) * 3 + .9, sym='', widths=0.3)
sixth = plt.boxplot(single_selective_lengths, positions=np.array(range(length)) * 3 + 1.2, sym='', widths=0.3)
seventh = plt.boxplot(single_selective_lengths2, positions=np.array(range(length)) * 3 + 1.5, sym='', widths=0.3)
set_box_color(second, '#2b8cbe', 'solid')  # colors are from http://colorbrewer2.org/
set_box_color(third, '#6baed6', '--')
set_box_color(fourth, '#238443', 'solid')
set_box_color(fifth, '#31a354', '--')
set_box_color(sixth, '#fd8d3c', 'solid')
set_box_color(seventh, '#feb24c', '--')
plt.xticks(range(1, (length) * 3 + 1, 3), [round(num, 2) for num in feature_weight_list],fontsize=14)
plt.xlim(-1, (length - 1) * 3 + 3)
plt.tight_layout()
plt.ylabel('Interval Length', fontsize=16)
plt.yticks(fontsize=14)

plt.sca(ax3)
second = plt.boxplot(selective_f1, positions=np.array(range(length)) * 3 , sym='', widths=0.3)
third = plt.boxplot(selective2_f1, positions=np.array(range(length)) * 3 + .3, sym='', widths=0.3)
fourth = plt.boxplot(ds_f1, positions=np.array(range(length)) * 3 + .6, sym='', widths=0.3)
fifth = plt.boxplot(ds2_f1, positions=np.array(range(length)) * 3 + .9, sym='', widths=0.3)
sixth = plt.boxplot(single_selective_f1, positions=np.array(range(length)) * 3 + 1.2, sym='', widths=0.3)
seventh = plt.boxplot(single_selective2_f1, positions=np.array(range(length)) * 3 + 1.5, sym='', widths=0.3)
set_box_color(second, '#2b8cbe', 'solid')  # colors are from http://colorbrewer2.org/
set_box_color(third, '#6baed6', '--')
set_box_color(fourth, '#238443', 'solid')
set_box_color(fifth, '#31a354', '--')
set_box_color(sixth, '#fd8d3c', 'solid')
set_box_color(seventh, '#feb24c', '--')
plt.xticks(range(1, (length) * 3 + 1, 3), [round(num, 2) for num in feature_weight_list],fontsize=14)
plt.xlim(-1, (length - 1) * 3 + 3)
plt.plot([], c='#D7191C', label='Naive', linewidth=2.5)
plt.plot([], c='#2b8cbe', label='MTL (0.7) + SI', linewidth=2.5)
plt.plot([], c='#6baed6', label='MTL (1.0) + SI', linestyle='--', linewidth=2.5)
plt.plot([], c='#238443', label='DS (0.67)', linewidth=2.5)
plt.plot([], c='#31a354', label='DS (0.5)', linestyle='--', linewidth=2.5)
plt.plot([], c='#fd8d3c', label='LASSO (0.7) + SI', linewidth=2.5)
plt.plot([], c='#feb24c', label='LASSO (1.0) + SI', linestyle='--', linewidth=2.5)
plt.legend()
plt.tight_layout()
plt.ylabel('F1 Score', fontsize=16)
plt.yticks(fontsize=14)

ax1.set_title("Coverage", y=1.01,fontsize=20)
ax2.set_title("Length", y=1.01,fontsize=20)
ax3.set_title("Accuracy", y=1.01,fontsize=20)

ax3.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.6), fontsize=16)

def common_format(ax):
    ax.grid(True, which='both', color='#f0f0f0')
    ax.set_xlabel('Lambda Value', fontsize=16)
    return ax

common_format(ax1)
common_format(ax2)
common_format(ax3)

ax1.axhline(y=0.9, color='k', linestyle='--', linewidth=2)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.6)
plt.savefig('cov_len_f1_by_lambda2.png', bbox_inches='tight')
