import numpy as np
import matplotlib

matplotlib.use('agg')
from scipy.stats import norm as ndist
from scipy.stats import t as tdist

from selectinf.randomized.multitask_lasso import multi_task_lasso
from selectinf.tests.instance import gaussian_multitask_instance
from selectinf.randomized.lasso import lasso, selected_targets


#Compute intervals, pivots, coverage, sensitivity, specificity, and testing error for post-selection inference
def test_multitask_lasso_selective_inference(predictor_vars_train,
                                response_vars_train,
                                predictor_vars_test,
                                response_vars_test,
                                beta,
                                gaussian_noise,
                                sigma,
                                link="identity",
                                weight=1.0,
                                randomizer_scale=0.7):
    ntask = len(predictor_vars_train.keys())
    nsamples_test = np.asarray([np.shape(predictor_vars_test[i])[0] for i in range(ntask)])
    p = np.shape(beta)[0]

    feature_weight = weight * np.ones(p)
    randomizer_scales = randomizer_scale * np.array([sigma[i] for i in range(ntask)])
    initial_omega = np.array(
        [randomizer_scales[i] * gaussian_noise[p*i:p*(i+1)] for i in range(ntask)]).T

    if link == "identity":
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

    if link == "logit":
        try:
            multi_lasso = multi_task_lasso.logistic(predictor_vars_train,
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

    if link == "log":
        try:
            multi_lasso = multi_task_lasso.poisson(predictor_vars_train,
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
    sensitivity_inference = np.float(num_true_positive_inference) / np.maximum(np.float(num_positive), 1)
    specificity_inference = 1.0 - np.float(num_false_positive_inference) / np.maximum(np.float(num_negative), 1)

    return np.asarray(coverage), intervals[:, 1] - intervals[:,
                                                   0], pivot, sensitivity_inference, specificity_inference, error

#Compute intervals, pivots, coverage, sensitivity, specificity, and testing error for naive inference
def test_multitask_lasso_naive(predictor_vars_train,
                                      response_vars_train,
                                      predictor_vars_test,
                                      response_vars_test,
                                      beta,
                                      sigma,
                                      weight=1.0,
                                      link="identity"):
    ntask = len(predictor_vars_train.keys())
    nsamples_test = np.asarray([np.shape(predictor_vars_test[i])[0] for i in range(ntask)])
    p = np.shape(beta)[0]

    feature_weight = weight * np.ones(p)
    sigmas_ = sigma
    perturbations = np.zeros((p, ntask))

    if link == "identity":
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

    if link == "logit":
        try:
            multi_lasso = multi_task_lasso.logistic(predictor_vars_train,
                                                    response_vars_train,
                                                    feature_weight,
                                                    ridge_term=None,
                                                    randomizer_scales=1. * sigmas_,
                                                    perturbations=perturbations)
            active_signs = multi_lasso.fit()

        except:

            active_signs = np.asarray([])

    if link == "log":
        try:
            multi_lasso = multi_task_lasso.poisson(predictor_vars_train,
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
    CIs = [[0, 0]]

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
            CIs = np.vstack([CIs, intervals])
            coverage.extend((beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1]))
            pivot_ = ndist.cdf((observed_target - beta_target) / np.sqrt(np.diag(cov_target)))
            pivot.extend(2 * np.minimum(pivot_, 1. - pivot_))

            idx_new = np.sum(active_signs[:, i] != 0)
            if idx_new == 0:
                error += (np.sum(np.square(response_vars_test[i]))) / nsamples_test[i]
                continue
            observed_target = np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(y)
            error += (np.sum(np.square(
                response_vars_test[i] - (predictor_vars_test[i])[:, (active_signs[:, i] != 0)].dot(
                    observed_target)))) / nsamples_test[i]

    else:
        error = 0
        for j in range(ntask):
            error += (np.linalg.norm(response_vars_test[j], 2) ** 2) / nsamples_test[j]
        CIs = np.asarray([[0, 0], [np.nan, np.nan]])

    # Compute sensitivity and specificity after inference
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
    sensitivity_inference = np.float(num_true_positive_inference) / np.maximum(np.float(num_positive), 1)
    specificity_inference = 1.0 - np.float(num_false_positive_inference) / np.maximum(np.float(num_negative), 1)

    return np.asarray(coverage), CIs[1:, 1] - CIs[1:, 0], pivot, sensitivity_inference, specificity_inference, error


#Compute intervals, pivots, coverage, sensitivity, specificity, and testing error for data splitting
def test_multitask_lasso_data_splitting(predictor_vars_train,
                                        response_vars_train,
                                        predictor_vars_test,
                                        response_vars_test,
                                        beta,
                                        sigma,
                                        weight=1.0,
                                        split=0.5,
                                        link="identity"):
    ntask = len(predictor_vars_train.keys())
    nsamples = np.asarray([np.shape(predictor_vars_train[i])[0] for i in range(ntask)])
    nsamples_test = np.asarray([np.shape(predictor_vars_test[i])[0] for i in range(ntask)])
    p = np.shape(beta)[0]

    samples = np.arange(np.int(nsamples[0]))
    selection = np.random.choice(samples, size=np.int(split * nsamples[0]), replace=False)
    inference = np.setdiff1d(samples, selection)
    response_vars_selection = {j: response_vars_train[j][selection] for j in range(ntask)}
    predictor_vars_selection = {j: predictor_vars_train[j][selection] for j in range(ntask)}
    response_vars_inference = {j: response_vars_train[j][inference] for j in range(ntask)}
    predictor_vars_inference = {j: predictor_vars_train[j][inference] for j in range(ntask)}

    feature_weight = weight * np.ones(p)
    sigmas_ = sigma
    perturbations = np.zeros((p, ntask))

    if link == "identity":
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

    if link == "logit":
        try:
            multi_lasso = multi_task_lasso.logistic(predictor_vars_selection,
                                                    response_vars_selection,
                                                    feature_weight,
                                                    ridge_term=None,
                                                    randomizer_scales=1. * sigmas_,
                                                    perturbations=perturbations)

            active_signs = multi_lasso.fit()

        except:

            active_signs = np.asarray([])

    if link == "log":
        try:
            multi_lasso = multi_task_lasso.poisson(predictor_vars_selection,
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
    CIs = [[0, 0]]

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
            coverage.extend(np.asarray(beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1]))
            pivot_ = ndist.cdf((observed_target - beta_target) / np.sqrt(np.diag(cov_target)))
            pivot.extend(2 * np.minimum(pivot_, 1. - pivot_))
            CIs = np.vstack([CIs, intervals])

            idx_new = np.sum(active_signs[:, i] != 0)
            if idx_new == 0:
                error += (np.sum(np.square(response_vars_test[i]))) / nsamples_test[i]
                continue
            error += (np.sum(np.square(
                response_vars_test[i] - (predictor_vars_test[i])[:, (active_signs[:, i] != 0)].dot(
                    observed_target)))) / nsamples_test[i]

    else:
        error = 0
        for j in range(ntask):
            error += (np.linalg.norm(response_vars_test[j], 2) ** 2) / nsamples_test[j]
        CIs = np.asarray([[0, 0], [np.nan, np.nan]])

    # Compute sensitivity and specificity after inference
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
    sensitivity_inference = np.float(num_true_positive_inference) / np.float(num_positive)
    specificity_inference = 1.0 - np.float(num_false_positive_inference) / np.float(num_negative)

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


#Compute intervals, pivots, coverage, sensitivity, specificity, and testing error for single-task post-selection inference
def test_single_task_lasso_selective_inference(predictor_vars_train,
                                       response_vars_train,
                                       predictor_vars_test,
                                       response_vars_test,
                                       beta,
                                       gaussian_noise,
                                       sigma,
                                       weight,
                                       randomizer_scale=1.0,
                                       link="identity"):
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
    sensitivity_inference = np.float(num_true_positive_inference) / np.float(num_positive)
    specificity_inference = 1.0 - np.float(num_false_positive_inference) / np.float(num_negative)

    return np.asarray(coverage), CIs[1:, 1] - CIs[1:, 0], np.asarray(
        pivot), sensitivity_inference, specificity_inference, error


def _noise(n, df=np.inf):
    if df == np.inf:
        return np.random.standard_normal(n)
    else:
        sd_t = np.std(tdist.rvs(df, size=50000))
    return tdist.rvs(df, size=n) / sd_t

#################
#Function to generate simulation data
#Tests four methods (selective inference, naive inference, data splitting, and single-task selective inference)
#Takes as input a list of lambda values for each method, a signal parameter, the regression dimension p, the task sparsity and global sparsity rates, and the number of simulations
#Returns the coverage, interval length, sensitivity, and specificity for each method and simulation
#Returns average testing error across simulations for each method at the specified tuning parameters
#################


def test_inference(weight, signal, p, ts, gs, nsim=100):
    np.random.seed(5)
    #Track intervals, pivots, sensitivity, specificity, and testing error for selective inference v1
    cov, len1, pivots, sensitivity_list, specificity_list, test_error_list = ([] for _ in range(6))

    # Track intervals, pivots, sensitivity, specificity, and testing error for selective inference v2
    cov2, len2, pivots2, sensitivity_list2, specificity_list2, test_error_list2 = ([] for _ in range(6))

    # Track intervals, pivots, sensitivity, specificity, and testing error for naive inference
    cov_naive, len_naive, pivots_naive, sensitivity_list_naive, specificity_list_naive, naive_test_error_list \
        = ([] for _ in range(6))

    # Track intervals, pivots, sensitivity, specificity, and testing error for data splitting v1
    cov_data_splitting, len_data_splitting, pivots_data_splitting, sensitivity_list_ds, specificity_list_ds, \
        data_splitting_test_error_list = ([] for _ in range(6))

    # Track intervals, pivots, sensitivity, specificity, and testing error for data splitting v2
    cov_data_splitting2, len_data_splitting2, pivots_data_splitting2, sensitivity_list_ds2, \
        specificity_list_ds2, data_splitting_test_error_list2 = ([] for _ in range(6))

    # Track intervals, sensitivity, specificity, and testing error for single-task selective inference v1
    cov_single_task_selective, len_single_task_selective, sensitivity_list_single_task_selective, \
        specificity_list_single_task_selective, single_task_selective_test_error_list = ([] for _ in range(5))

    #Track intervals, sensitivity, specificity, and testing error for single-task selective inference v2
    cov_single_task_selective2, len_single_task_selective2, sensitivity_list_single_task_selective2, \
        specificity_list_single_task_selective2, single_task_selective_test_error_list2 = ([] for _ in range(5))

    #Generate training and testing data
    ntask = 5
    nsamples = 500 * np.ones(ntask)
    nsamples_test = 500 * np.ones(ntask)
    p = p
    global_sparsity = gs
    task_sparsity = ts
    sigma = 1. * np.ones(ntask)
    signal_fac = np.array(signal)
    rhos = 0.3 * np.ones(ntask)
    nsamples = nsamples.astype(int)
    nsamples_test = nsamples_test.astype(int)
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    response_vars_train, predictor_vars_train, response_vars_test, predictor_vars_test, beta, gaussian_noise = \
        gaussian_multitask_instance(ntask,
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

    #Print SNR, PVE
    SIG = np.full((p, p), 0.3)
    np.fill_diagonal(SIG, 1.0)
    SNR = beta.T.dot(SIG.dot(beta)) / 1000
    SNR = np.diag(SNR)
    print(SNR, "SNR")
    print(SNR / (1 + SNR), "PVE")

    for n in range(nsim):

        #For each iteration, generate independent training errors, testing errors, and randomization variables
        #Compute new responses from the new errors
        if n >= 1:

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
                    response_vars_train[i] = (predictor_vars_train[i].dot(beta[:, i]/ sigma[i])
                                              + gaussian_noise[nsamples_train_cumsum[i-1]: nsamples_train_cumsum[i]]) *\
                                             sigma[i]
                    response_vars_test[i] = (predictor_vars_test[i].dot(beta[:, i]/ sigma[i]) +
                                             gaussian_noise[nsamples.sum() + nsamples_test_cumsum[i-1]: nsamples.sum() +
                                             nsamples_test_cumsum[i]]) * sigma[i]

            gaussian_noise = gaussian_noise[nsamples.sum() + nsamples_test.sum():]

        print(n, "n sim")
        print(weight, "weight")

        #Record results for multi-task selective inference v1
        coverage, length, pivot, sns, spc, err = test_multitask_lasso_selective_inference(predictor_vars_train,
                                                                                          response_vars_train,
                                                                                          predictor_vars_test,
                                                                                          response_vars_test,
                                                                                          beta,
                                                                                          gaussian_noise,
                                                                                          sigma,
                                                                                          link="identity",
                                                                                          weight=weight[0],
                                                                                          randomizer_scale=0.7)

        if list(coverage):
            cov.append(np.mean(np.asarray(coverage)))
            len1.extend(length)
            pivots.extend(pivot)
        sensitivity_list.append(sns)
        specificity_list.append(spc)
        test_error_list.append(err)

        # Record results for multi-task selective inference v2
        coverage2, length2, pivot2, sns2, spc2, err2 = test_multitask_lasso_selective_inference(predictor_vars_train,
                                                                                                response_vars_train,
                                                                                                predictor_vars_test,
                                                                                                response_vars_test,
                                                                                                beta,
                                                                                                gaussian_noise,
                                                                                                sigma,
                                                                                                link="identity",
                                                                                                weight=weight[1],
                                                                                                randomizer_scale=1.0)

        if list(coverage2):
            cov2.append(np.mean(np.asarray(coverage2)))
            len2.extend(length2)
            pivots2.extend(pivot2)
        sensitivity_list2.append(sns2)
        specificity_list2.append(spc2)
        test_error_list2.append(err2)

        # Record results for naive inference
        coverage_naive, length_naive, pivot_naive, naive_sensitivity, naive_specificity, naive_err = \
            test_multitask_lasso_naive(predictor_vars_train,
                                       response_vars_train,
                                       predictor_vars_test,
                                       response_vars_test,
                                       beta,
                                       sigma,
                                       weight[2],
                                       link="identity")

        if list(coverage_naive):
            cov_naive.append(np.mean(np.asarray(coverage_naive)))
            len_naive.extend(length_naive)
            pivots_naive.extend(pivot_naive)
        sensitivity_list_naive.append(naive_sensitivity)
        specificity_list_naive.append(naive_specificity)
        naive_test_error_list.append(naive_err)

        # Record results for data splitting v1
        coverage_data_splitting, length_data_splitting, pivot_data_splitting, sns_ds, spc_ds, ds_error = \
            test_multitask_lasso_data_splitting(predictor_vars_train,
                                                response_vars_train,
                                                predictor_vars_test,
                                                response_vars_test,
                                                beta,
                                                sigma,
                                                weight[3],
                                                split=0.67,
                                                link="identity")

        if list(coverage_data_splitting):
            cov_data_splitting.append(np.mean(np.asarray(coverage_data_splitting)))
            len_data_splitting.extend(length_data_splitting)
            pivots_data_splitting.extend(pivot_data_splitting)
        sensitivity_list_ds.append(sns_ds)
        specificity_list_ds.append(spc_ds)
        data_splitting_test_error_list.append(ds_error)

        # Record results for data splitting v2
        coverage_data_splitting2, length_data_splitting2, pivot_data_splitting2, sns_ds2, spc_ds2, ds_error2 = \
            test_multitask_lasso_data_splitting(predictor_vars_train,
                                                response_vars_train,
                                                predictor_vars_test,
                                                response_vars_test,
                                                beta,
                                                sigma,
                                                weight[4],
                                                split=0.5,
                                                link="identity")

        if list(coverage_data_splitting2):
            cov_data_splitting2.append(np.mean(np.asarray(coverage_data_splitting2)))
            len_data_splitting2.extend(length_data_splitting2)
            pivots_data_splitting2.extend(pivot_data_splitting2)
        sensitivity_list_ds2.append(sns_ds2)
        specificity_list_ds2.append(spc_ds2)
        data_splitting_test_error_list2.append(ds_error2)

        # Record results for single-task selective inference v1
        coverage_single_task_selective, length_single_task_selective, pivot_single_task_selective, sns_single_task, \
            spc_single_task, err_single_selective = test_single_task_lasso_selective_inference(predictor_vars_train,
                                                                                               response_vars_train,
                                                                                               predictor_vars_test,
                                                                                               response_vars_test,
                                                                                               beta,
                                                                                               gaussian_noise,
                                                                                               sigma,
                                                                                               weight[5],
                                                                                               randomizer_scale=0.7,
                                                                                               link="identity")

        if list(coverage_single_task_selective):
            cov_single_task_selective.append(np.mean(np.asarray(coverage_single_task_selective)))
            len_single_task_selective.extend(length_single_task_selective)
        sensitivity_list_single_task_selective.append(sns_single_task)
        specificity_list_single_task_selective.append(spc_single_task)
        single_task_selective_test_error_list.append(err_single_selective)

        # Record results for multi-task selective inference v2
        coverage_single_task_selective2, length_single_task_selective2, pivot_single_task_selective2, sns_single_task2,\
            spc_single_task2, err_single_selective2 = test_single_task_lasso_selective_inference(predictor_vars_train,
                                                                                                 response_vars_train,
                                                                                                 predictor_vars_test,
                                                                                                 response_vars_test,
                                                                                                 beta,
                                                                                                 gaussian_noise,
                                                                                                 sigma,
                                                                                                 weight[6],
                                                                                                 randomizer_scale=1.0,
                                                                                                 link="identity")

        if list(coverage_single_task_selective2):
            cov_single_task_selective2.append(np.mean(np.asarray(coverage_single_task_selective2)))
            len_single_task_selective2.extend(length_single_task_selective2)
        sensitivity_list_single_task_selective2.append(sns_single_task2)
        specificity_list_single_task_selective2.append(spc_single_task2)
        single_task_selective_test_error_list2.append(err_single_selective2)

        print("iteration completed ", n)
        print("posi v1 coverage so far ", np.mean(np.asarray(cov)))
        print("naive coverage so far ", np.mean(np.asarray(cov_naive)))
        print("data splitting v1 coverage so far ", np.mean(np.asarray(cov_data_splitting)))
        print("single-task selective inference v1 coverage so far ", np.mean(np.asarray(cov_single_task_selective)))

    return ({"MTL_SI_07_pivots": pivots,
             "Naive_pivots": pivots_naive,
             "DS_67_pivots": pivots_data_splitting,
             "MTL_SI_07_coverage": np.asarray(cov),
             "MTL_SI_1_coverage": np.asarray(cov2),
             "Naive_coverage": np.asarray(cov_naive),
             "DS_67_coverage": np.asarray(cov_data_splitting),
             "DS_50_coverage": np.asarray(cov_data_splitting2),
             "LASSO_SI_07_coverage": np.asarray(cov_single_task_selective),
             "LASSO_SI_1_coverage": np.asarray(cov_single_task_selective2),
             "MTL_SI_07_length": np.asarray(len1),
             "MTL_SI_1_length": np.asarray(len2),
             "Naive_length": np.asarray(len_naive),
             "DS_67_length": np.asarray(len_data_splitting),
             "DS_50_length": np.asarray(len_data_splitting2),
             "LASSO_SI_07_length": np.asarray(len_single_task_selective),
             "LASSO_SI_1_length": np.asarray(len_single_task_selective2),
             "MTL_SI_07_sensitivity": np.asarray(sensitivity_list),
             "MTL_SI_1_sensitivity": np.asarray(sensitivity_list2),
             "Naive_sensitivity": np.asarray(sensitivity_list_naive),
             "DS_67_sensitivity": np.asarray(sensitivity_list_ds),
             "DS_50_sensitivity": np.asarray(sensitivity_list_ds2),
             "LASSO_SI_07_sensitivity": np.asarray(sensitivity_list_single_task_selective),
             "LASSO_SI_1_sensitivity": np.asarray(sensitivity_list_single_task_selective2),
             "MTL_SI_07_specificity": np.asarray(specificity_list),
             "MTL_SI_1_specificity": np.asarray(specificity_list2),
             "Naive_specificity": np.asarray(specificity_list_naive),
             "DS_67_specificity": np.asarray(specificity_list_ds),
             "DS_50_specificity": np.asarray(specificity_list_ds2),
             "LASSO_SI_07_specificity": np.asarray(specificity_list_single_task_selective),
             "LASSO_SI_1_specificity": np.asarray(specificity_list_single_task_selective2),
             "MTL_SI_07_error": np.mean(np.asarray(test_error_list)),
             "MTL_SI_1_error": np.mean(np.asarray(test_error_list2)),
             "Naive_error": np.mean(np.asarray(naive_test_error_list)),
             "DS_67_error": np.mean(np.asarray(data_splitting_test_error_list)),
             "DS_50_error": np.mean(np.asarray(data_splitting_test_error_list2)),
             "LASSO_SI_07_error": np.mean(np.asarray(single_task_selective_test_error_list)),
             "LASSO_SI_1_error": np.mean(np.asarray(single_task_selective_test_error_list2))})
