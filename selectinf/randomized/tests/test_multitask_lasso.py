import numpy as np
from scipy.stats import norm as ndist
from scipy.stats import t as tdist

from selectinf.randomized.multitask_lasso import multi_task_lasso
from selectinf.tests.instance import gaussian_multitask_instance, gaussian_multitask_instance_joint_pairwise, \
    gaussian_multitask_instance_high_low_ts
from selectinf.randomized.lasso import lasso, selected_targets


# Compute intervals, pivots, coverage, sensitivity, specificity, and testing error for post-selection inference
def test_joint_mtl_si(predictor_vars_train,
                      response_vars_train,
                      predictor_vars_test,
                      response_vars_test,
                      beta,
                      gaussian_noise,
                      sigma,
                      weight=1.0,
                      randomizer_scale=0.7):
    ntask = len(predictor_vars_train.keys())
    nsamples_test = np.asarray([np.shape(predictor_vars_test[j])[0] for j in range(ntask)])
    p = np.shape(beta)[0]

    feature_weight = weight * np.ones(p)
    randomizer_scales = randomizer_scale * np.array([sigma[j] for j in range(ntask)])
    initial_omega = np.array(
        [randomizer_scales[j] * gaussian_noise[p * j:p * (j + 1)] for j in range(ntask)]).T

    true_active = np.transpose(np.nonzero(np.transpose(beta)))
    num_positive = np.shape(true_active)[0]
    num_negative = np.shape(beta)[0] * np.shape(beta)[1] - num_positive

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

        beta_target = []

        for j in range(ntask):
            X, y = multi_lasso.loglikes[j].data
            beta_target.extend(np.linalg.pinv(X[:, (active_signs[:, j] != 0)]).dot(X.dot(beta[:, j])))

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

        selected_active = np.transpose(np.nonzero(np.transpose(active_signs)))
        true_positive_selected = [x in true_active.tolist() for x in selected_active.tolist()]
        num_true_positive_inference = np.sum(
            [true_positive_selected[j] * (intervals[j, 1] < 0 or intervals[j, 0] > 0) for j in
             range(len(true_positive_selected))])
        num_false_positive_inference = np.sum([(intervals[j, 1] < 0 or intervals[j, 0] > 0) for j in
                                               range(len(true_positive_selected))]) - num_true_positive_inference

    except IndexError:
        print("No active predictors for any task.")
        coverage = []
        intervals = np.asarray([[np.nan, np.nan]])
        pivot = []

        error = 0
        for j in range(ntask):
            error += (np.linalg.norm(response_vars_test[j], 2) ** 2) / nsamples_test[j]

        num_true_positive_inference = 0
        num_false_positive_inference = 0

    sensitivity_inference = np.float(num_true_positive_inference) / np.maximum(np.float(num_positive), 1)
    specificity_inference = 1.0 - np.float(num_false_positive_inference) / np.maximum(np.float(num_negative), 1)

    return np.asarray(coverage), intervals[:, 1] - intervals[:, 0], pivot, \
        sensitivity_inference, specificity_inference, error


def test_pairwise_mtl_si(predictor_vars_train,
                         response_vars_train,
                         predictor_vars_test,
                         response_vars_test,
                         beta,
                         gaussian_noise,
                         sigma,
                         weight=1.0,
                         randomizer_scale=0.7):
    """
    Assumes the first two tasks are one group and the second two tasks are another group
    """

    ntask = len(predictor_vars_train.keys())
    nsamples_test = np.asarray([np.shape(predictor_vars_test[j])[0] for j in range(ntask)])
    p = np.shape(beta)[0]

    feature_weight = weight * np.ones(p)
    randomizer_scales = randomizer_scale * np.array([sigma[j] for j in range(ntask)])
    initial_omega = np.array(
        [randomizer_scales[j] * gaussian_noise[p * j:p * (j + 1)] for j in range(ntask)]).T

    try:
        multi_lasso_group1 = multi_task_lasso.gaussian({0: predictor_vars_train[0], 1: predictor_vars_train[1]},
                                                       {0: response_vars_train[0], 1: response_vars_train[1]},
                                                       feature_weight,
                                                       ridge_term=None,
                                                       randomizer_scales=randomizer_scales)

        active_signs_group1 = multi_lasso_group1.fit(perturbations=initial_omega[:, [0, 1]])
        dispersions = sigma ** 2
        estimate_group1, observed_info_mean_group1, Z_scores_group1, pvalues_group1, intervals_group1 = \
            multi_lasso_group1.multitask_inference_hetero(dispersions=dispersions)

    except IndexError:
        print("No active predictors - group 1.")
        active_signs_group1 = np.zeros((p, 2))
        estimate_group1 = np.asarray([])
        intervals_group1 = np.empty((0, 2))

    try:
        multi_lasso_group2 = multi_task_lasso.gaussian({0: predictor_vars_train[2], 1: predictor_vars_train[3]},
                                                       {0: response_vars_train[2], 1: response_vars_train[3]},
                                                       feature_weight,
                                                       ridge_term=None,
                                                       randomizer_scales=randomizer_scales)

        active_signs_group2 = multi_lasso_group2.fit(perturbations=initial_omega[:, [2, 3]])
        dispersions = sigma ** 2
        estimate_group2, observed_info_mean_group2, Z_scores_group2, pvalues_group2, intervals_group2 = \
            multi_lasso_group2.multitask_inference_hetero(dispersions=dispersions)

    except IndexError:
        print("No active predictors - group 2.")
        active_signs_group2 = np.zeros((p, 2))
        estimate_group2 = np.asarray([])
        intervals_group2 = np.empty((0, 2))

    active_signs = np.concatenate((active_signs_group1, active_signs_group2), axis=1)

    coverage = []

    if (active_signs != 0).sum() > 0:
        estimate = np.concatenate((estimate_group1, estimate_group2), axis=0)
        intervals = np.concatenate((intervals_group1, intervals_group2), axis=0)

        beta_target = []

        for j in range(ntask):
            X, y = predictor_vars_train[j], response_vars_train[j]
            beta_target.extend(np.linalg.pinv(X[:, (active_signs[:, j] != 0)]).dot(X.dot(beta[:, j])))

        beta_target = np.asarray(beta_target)
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

    # Compute sensitivity and specificity after inference
    true_active = np.transpose(np.nonzero(np.transpose(beta)))
    num_positive = np.shape(true_active)[0]
    num_negative = np.shape(beta)[0] * np.shape(beta)[1] - num_positive

    if (active_signs != 0).sum() > 0:
        selected_active = np.transpose(np.nonzero(np.transpose(active_signs)))
        true_positive_selected = [x in true_active.tolist() for x in selected_active.tolist()]
        num_true_positive_inference = np.sum(
            [true_positive_selected[j] * (intervals[j, 1] < 0 or intervals[j, 0] > 0) for j in
             range(len(true_positive_selected))])
        num_false_positive_inference = np.sum([(intervals[j, 1] < 0 or intervals[j, 0] > 0) for j in
                                               range(len(true_positive_selected))]) - num_true_positive_inference
    else:
        num_true_positive_inference = 0
        num_false_positive_inference = 0

    sensitivity_inference = np.float(num_true_positive_inference) / np.maximum(np.float(num_positive), 1)
    specificity_inference = 1.0 - np.float(num_false_positive_inference) / np.maximum(np.float(num_negative), 1)

    return np.asarray(coverage), intervals[:, 1] - intervals[:, 0], sensitivity_inference, specificity_inference, error


# Compute intervals, pivots, coverage, sensitivity, specificity, and testing error for data splitting
def test_data_splitting(predictor_vars_train,
                        response_vars_train,
                        predictor_vars_test,
                        response_vars_test,
                        beta,
                        sigma,
                        weight=1.0,
                        split=0.5):
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
    perturbations = np.zeros((p, ntask))
    dispersions = sigma ** 2

    coverage = []
    pivot = []
    CIs = np.empty((0, 2))
    error = 0

    true_active = np.transpose(np.nonzero(np.transpose(beta)))
    num_positive = np.shape(true_active)[0]
    num_negative = np.shape(beta)[0] * np.shape(beta)[1] - num_positive

    try:
        multi_lasso = multi_task_lasso.gaussian(predictor_vars_selection,
                                                response_vars_selection,
                                                feature_weight,
                                                ridge_term=None,
                                                randomizer_scales=1. * sigma,
                                                perturbations=perturbations)

        active_signs = multi_lasso.fit()

        for j in range(ntask):
            X = predictor_vars_inference[j]
            y = response_vars_inference[j]
            beta_target = np.linalg.pinv(X[:, (active_signs[:, j] != 0)]).dot(X.dot(beta[:, j]))

            Qfeat = np.linalg.inv(X[:, (active_signs[:, j] != 0)].T.dot(X[:, (active_signs[:, j] != 0)]))
            observed_target = np.linalg.pinv(X[:, (active_signs[:, j] != 0)]).dot(y)
            cov_target = Qfeat * dispersions[j]

            alpha = 1. - 0.90
            quantile = ndist.ppf(1 - alpha / 2.)
            intervals = np.vstack([observed_target - quantile * np.sqrt(np.diag(cov_target)),
                                   observed_target + quantile * np.sqrt(np.diag(cov_target))]).T
            CIs = np.concatenate((CIs, intervals), axis=0)

            coverage.extend(np.asarray(beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1]))
            pivot_ = ndist.cdf((observed_target - beta_target) / np.sqrt(np.diag(cov_target)))
            pivot.extend(2 * np.minimum(pivot_, 1. - pivot_))

            idx_new = np.sum(active_signs[:, j] != 0)
            if idx_new == 0:
                error += (np.sum(np.square(response_vars_test[j]))) / nsamples_test[j]
                continue
            error += (np.sum(np.square(
                response_vars_test[j] - (predictor_vars_test[j])[:, (active_signs[:, j] != 0)].dot(
                    observed_target)))) / nsamples_test[j]

        selected_active = np.transpose(np.nonzero(np.transpose(active_signs)))
        true_positive_selected = [x in true_active.tolist() for x in selected_active.tolist()]
        num_true_positive_inference = np.sum(
            [true_positive_selected[j] * (CIs[j, 1] < 0 or CIs[j, 0] > 0) for j in
             range(len(true_positive_selected))])
        num_false_positive_inference = np.sum([(CIs[j, 1] < 0 or CIs[j, 0] > 0) for j in range(
            len(true_positive_selected))]) - num_true_positive_inference

    except IndexError:
        print("No active predictors for any task.")

        for j in range(ntask):
            error += (np.linalg.norm(response_vars_test[j], 2) ** 2) / nsamples_test[j]
        CIs = np.asarray([[np.nan, np.nan]])

        num_true_positive_inference = 0
        num_false_positive_inference = 0

    # Compute sensitivity and specificity after inference
    sensitivity_inference = np.float(num_true_positive_inference) / np.float(num_positive)
    specificity_inference = 1.0 - np.float(num_false_positive_inference) / np.float(num_negative)

    return np.asarray(coverage), CIs[:, 1] - CIs[:, 0], pivot, sensitivity_inference, specificity_inference, error


# Compute intervals, pivots, coverage, sensitivity, specificity, and testing error for
# single-task post-selection inference
def test_lasso_si(predictor_vars_train,
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
    CIs = np.empty((0, 2))
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

        initial_omega = np.array(randomizer_scale * sigma[i] * gaussian_noise[p * i:p * (i + 1)]).T
        signs = single_task_lasso.fit(perturb=initial_omega)
        nonzero = signs != 0

        (observed_target, cov_target, cov_target_score, alternatives) = \
            selected_targets(single_task_lasso.loglike, single_task_lasso._W, nonzero, dispersion=sigma[i] ** 2)

        MLE_result, observed_info_mean = single_task_lasso.selective_MLE(
            observed_target,
            cov_target,
            cov_target_score,
            level=0.90)[0:2]

        final_estimator = MLE_result['MLE']
        intervals = np.asarray(MLE_result[['lower_confidence', 'upper_confidence']])
        CIs = np.concatenate((CIs, intervals), axis=0)
        selected_active.extend([[i, j] for j in np.nonzero(signs)[0]])

        beta_target = np.linalg.pinv(predictor_vars_train[i][:, nonzero]).dot(
            predictor_vars_train[i].dot(beta[:, i]))
        coverage.extend(np.asarray(beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1]))
        pivot_ = ndist.cdf((final_estimator - beta_target) / np.sqrt(np.diag(observed_info_mean)))
        pivot.extend(2 * np.minimum(pivot_, 1. - pivot_))

        idx_new = np.sum(signs != 0)
        if idx_new == 0:
            error += (np.sum(np.square(response_vars_test[i]))) / nsamples_test[i]
        else:
            error += (np.sum(np.square(
                response_vars_test[i] - (predictor_vars_test[i])[:, nonzero].dot(
                    final_estimator)))) / nsamples_test[i]

    true_active = np.transpose(np.nonzero(np.transpose(beta)))
    num_positive = np.shape(true_active)[0]
    num_negative = np.shape(beta)[0] * np.shape(beta)[1] - num_positive

    if selected_active != []:
        true_positive_selected = [x in true_active.tolist() for x in selected_active]
        num_true_positive_inference = np.sum(
            [true_positive_selected[i] * (CIs[i, 1] < 0 or CIs[i, 0] > 0) for i in
             range(len(true_positive_selected))])
        num_false_positive_inference = np.sum([(CIs[i, 1] < 0 or CIs[i, 0] > 0) for i in range(
            len(true_positive_selected))]) - num_true_positive_inference
    else:
        CIs = np.asarray([[np.nan, np.nan]])
        num_true_positive_inference = 0
        num_false_positive_inference = 0

    sensitivity_inference = np.float(num_true_positive_inference) / np.float(num_positive)
    specificity_inference = 1.0 - np.float(num_false_positive_inference) / np.float(num_negative)

    return np.asarray(coverage), CIs[:, 1] - CIs[:, 0], np.asarray(pivot), \
        sensitivity_inference, specificity_inference, error


# Compute intervals, pivots, coverage, sensitivity, specificity, and testing error for naive inference
def test_multitask_lasso_naive(predictor_vars_train,
                               response_vars_train,
                               predictor_vars_test,
                               response_vars_test,
                               beta,
                               sigma,
                               weight=1.0):
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

    except Exception as e:
        print('multi-task lasso fitting error', e)
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


def _noise(n, df=np.inf):
    if df == np.inf:
        return np.random.standard_normal(n)
    else:
        sd_t = np.std(tdist.rvs(df, size=50000))
    return tdist.rvs(df, size=n) / sd_t


def _laplace_response_noise(n):
    return np.random.laplace(loc=0, scale=1., size=n) / np.sqrt(2.)


def _exponential_response_noise(n):
    return np.random.exponential(scale=1.0, size=n) - 1.0


#################
# Function to generate simulation data
# Tests four methods (selective inference, naive inference, data splitting, and single-task selective inference)
# Takes as input a list of lambda values for each method, a signal parameter, the regression dimension p, the task
# sparsity and global sparsity rates, and the number of simulations
# Returns the coverage, interval length, sensitivity, and specificity for each method and simulation
# Returns average testing error across simulations for each method at the specified tuning parameters
#################


def test_inference(weight, signal, p, ts, gs, nsim=100, seed=5):
    np.random.seed(seed)
    # Track intervals, pivots, sensitivity, specificity, and testing error for selective inference v1
    cov, len1, pivots, sensitivity_list, specificity_list, test_error_list = ([] for _ in range(6))

    # Track intervals, pivots, sensitivity, specificity, and testing error for data splitting v1
    cov_data_splitting, len_data_splitting, pivots_data_splitting, sensitivity_list_ds, specificity_list_ds, \
        data_splitting_test_error_list = ([] for _ in range(6))

    # Track intervals, pivots, sensitivity, specificity, and testing error for data splitting v2
    cov_data_splitting2, len_data_splitting2, pivots_data_splitting2, sensitivity_list_ds2, \
        specificity_list_ds2, data_splitting_test_error_list2 = ([] for _ in range(6))

    # Track intervals, sensitivity, specificity, and testing error for single-task selective inference v1
    cov_single_task_selective, len_single_task_selective, sensitivity_list_single_task_selective, \
        specificity_list_single_task_selective, single_task_selective_test_error_list = ([] for _ in range(5))

    # Generate training and testing data
    ntask = 4
    nsamples = 5000 * np.ones(ntask)
    nsamples_test = 1000 * np.ones(ntask)
    p = p
    global_sparsity = gs
    task_sparsity = ts
    sigma = 1. * np.ones(ntask)
    signal_fac = np.array(signal)
    rhos = 0.0 * np.ones(ntask)
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

    for n in range(nsim):

        # For each iteration, generate independent training errors, testing errors, and randomization variables
        # Compute new responses from the new errors
        if n >= 1:

            gaussian_noise = _noise(nsamples.sum() + nsamples_test.sum() + p * ntask)
            response_vars_train = {}
            response_vars_test = {}
            nsamples_train_cumsum = np.cumsum([nsamples[i] for i in range(ntask)])
            nsamples_test_cumsum = np.cumsum([nsamples_test[i] for i in range(ntask)])

            for i in range(ntask):
                if i == 0:
                    train_noise = gaussian_noise[:nsamples_train_cumsum[i]] * sigma[i]
                    response_vars_train[i] = predictor_vars_train[i].dot(beta[:, i]) + train_noise

                    test_noise = gaussian_noise[nsamples.sum():nsamples.sum() + nsamples_test_cumsum[i]] * sigma[i]
                    response_vars_test[i] = predictor_vars_test[i].dot(beta[:, i]) + test_noise
                else:
                    train_noise = gaussian_noise[nsamples_train_cumsum[i - 1]: nsamples_train_cumsum[i]] * sigma[i]
                    response_vars_train[i] = predictor_vars_train[i].dot(beta[:, i]) + train_noise

                    test_noise = gaussian_noise[nsamples.sum() + nsamples_test_cumsum[i - 1]:
                                                nsamples.sum() + nsamples_test_cumsum[i]] * sigma[i]
                    response_vars_test[i] = predictor_vars_test[i].dot(beta[:, i]) + test_noise

            gaussian_noise = gaussian_noise[nsamples.sum() + nsamples_test.sum():]

        print(n, "n sim")
        print(weight, "weight")

        # Record results for multi-task selective inference v1
        coverage, length, pivot, sns, spc, err = test_joint_mtl_si(predictor_vars_train,
                                                                   response_vars_train,
                                                                   predictor_vars_test,
                                                                   response_vars_test,
                                                                   beta,
                                                                   gaussian_noise,
                                                                   sigma,
                                                                   weight=weight[0],
                                                                   randomizer_scale=0.7)
        print(np.mean(coverage))
        if list(coverage):
            cov.append(np.mean(np.asarray(coverage)))
            len1.extend(length)
            pivots.extend(pivot)
        sensitivity_list.append(sns)
        specificity_list.append(spc)
        test_error_list.append(err)

        # Record results for data splitting v1
        coverage_data_splitting, length_data_splitting, pivot_data_splitting, sns_ds, spc_ds, ds_error = \
            test_data_splitting(predictor_vars_train,
                                response_vars_train,
                                predictor_vars_test,
                                response_vars_test,
                                beta,
                                sigma,
                                weight[1],
                                split=0.67)

        if list(coverage_data_splitting):
            cov_data_splitting.append(np.mean(np.asarray(coverage_data_splitting)))
            len_data_splitting.extend(length_data_splitting)
            pivots_data_splitting.extend(pivot_data_splitting)
        sensitivity_list_ds.append(sns_ds)
        specificity_list_ds.append(spc_ds)
        data_splitting_test_error_list.append(ds_error)

        # Record results for data splitting v2
        coverage_data_splitting2, length_data_splitting2, pivot_data_splitting2, sns_ds2, spc_ds2, ds_error2 = \
            test_data_splitting(predictor_vars_train,
                                response_vars_train,
                                predictor_vars_test,
                                response_vars_test,
                                beta,
                                sigma,
                                weight[2],
                                split=0.5)

        if list(coverage_data_splitting2):
            cov_data_splitting2.append(np.mean(np.asarray(coverage_data_splitting2)))
            len_data_splitting2.extend(length_data_splitting2)
            pivots_data_splitting2.extend(pivot_data_splitting2)
        sensitivity_list_ds2.append(sns_ds2)
        specificity_list_ds2.append(spc_ds2)
        data_splitting_test_error_list2.append(ds_error2)

        # Record results for single-task selective inference v1
        coverage_single_task_selective, length_single_task_selective, pivot_single_task_selective, sns_single_task, \
            spc_single_task, err_single_selective = test_lasso_si(predictor_vars_train,
                                                                  response_vars_train,
                                                                  predictor_vars_test,
                                                                  response_vars_test,
                                                                  beta,
                                                                  gaussian_noise,
                                                                  sigma,
                                                                  weight[3],
                                                                  randomizer_scale=0.7)

        if list(coverage_single_task_selective):
            cov_single_task_selective.append(np.mean(np.asarray(coverage_single_task_selective)))
            len_single_task_selective.extend(length_single_task_selective)
        sensitivity_list_single_task_selective.append(sns_single_task)
        specificity_list_single_task_selective.append(spc_single_task)
        single_task_selective_test_error_list.append(err_single_selective)

    return ({"MTL_SI_07_pivots": pivots,
             "DS_67_pivots": pivots_data_splitting,
             "MTL_SI_07_coverage": np.asarray(cov),
             "DS_67_coverage": np.asarray(cov_data_splitting),
             "DS_50_coverage": np.asarray(cov_data_splitting2),
             "LASSO_SI_07_coverage": np.asarray(cov_single_task_selective),
             "MTL_SI_07_length": np.asarray(len1),
             "DS_67_length": np.asarray(len_data_splitting),
             "DS_50_length": np.asarray(len_data_splitting2),
             "LASSO_SI_07_length": np.asarray(len_single_task_selective),
             "MTL_SI_07_sensitivity": np.asarray(sensitivity_list),
             "DS_67_sensitivity": np.asarray(sensitivity_list_ds),
             "DS_50_sensitivity": np.asarray(sensitivity_list_ds2),
             "LASSO_SI_07_sensitivity": np.asarray(sensitivity_list_single_task_selective),
             "MTL_SI_07_specificity": np.asarray(specificity_list),
             "DS_67_specificity": np.asarray(specificity_list_ds),
             "DS_50_specificity": np.asarray(specificity_list_ds2),
             "LASSO_SI_07_specificity": np.asarray(specificity_list_single_task_selective),
             "MTL_SI_07_error": np.mean(np.asarray(test_error_list)),
             "DS_67_error": np.mean(np.asarray(data_splitting_test_error_list)),
             "DS_50_error": np.mean(np.asarray(data_splitting_test_error_list2)),
             "LASSO_SI_07_error": np.mean(np.asarray(single_task_selective_test_error_list))})


def test_inference_mixed_ts(weight, signal, p, ts, gs, nsim=100, seed=5):
    np.random.seed(seed)
    # Track intervals, pivots, sensitivity, specificity, and testing error for selective inference
    cov, len1, pivots, sensitivity_list, specificity_list, test_error_list = ([] for _ in range(6))

    # Track intervals, pivots, sensitivity, specificity, and testing error for data splitting v1
    cov_data_splitting, len_data_splitting, pivots_data_splitting, sensitivity_list_ds, specificity_list_ds, \
        data_splitting_test_error_list = ([] for _ in range(6))

    # Track intervals, pivots, sensitivity, specificity, and testing error for data splitting v2
    cov_data_splitting2, len_data_splitting2, pivots_data_splitting2, sensitivity_list_ds2, \
        specificity_list_ds2, data_splitting_test_error_list2 = ([] for _ in range(6))

    # Track intervals, sensitivity, specificity, and testing error for single-task selective inference
    cov_single_task_selective, len_single_task_selective, sensitivity_list_single_task_selective, \
        specificity_list_single_task_selective, single_task_selective_test_error_list = ([] for _ in range(5))

    # Generate training and testing data
    ntask = 4
    nsamples = 5000 * np.ones(ntask)
    nsamples_test = 1000 * np.ones(ntask)
    p = p
    global_sparsity = gs
    task_sparsity_low = ts[0]
    task_sparsity_high = ts[1]
    sigma = 1. * np.ones(ntask)
    signal_fac = np.array(signal)
    rhos = 0.0 * np.ones(ntask)
    nsamples = nsamples.astype(int)
    nsamples_test = nsamples_test.astype(int)
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    response_vars_train, predictor_vars_train, response_vars_test, predictor_vars_test, beta, gaussian_noise = \
        gaussian_multitask_instance_high_low_ts(ntask,
                                                nsamples,
                                                nsamples_test,
                                                p,
                                                global_sparsity,
                                                task_sparsity_low,
                                                task_sparsity_high,
                                                sigma,
                                                signal,
                                                rhos,
                                                random_signs=True,
                                                equicorrelated=True)[:6]

    for n in range(nsim):

        # For each iteration, generate independent training errors, testing errors, and randomization variables
        # Compute new responses from the new errors
        # Note: Beta ia already multiplied by sigma
        if n >= 1:

            gaussian_noise = _noise(nsamples.sum() + nsamples_test.sum() + p * ntask)
            response_vars_train = {}
            response_vars_test = {}
            nsamples_train_cumsum = np.cumsum(nsamples)
            nsamples_test_cumsum = np.cumsum(nsamples_test)

            for i in range(ntask):
                if i == 0:
                    train_noise = gaussian_noise[:nsamples_train_cumsum[i]] * sigma[i]
                    response_vars_train[i] = predictor_vars_train[i].dot(beta[:, i]) + train_noise

                    test_noise = gaussian_noise[nsamples.sum():nsamples.sum() + nsamples_test_cumsum[i]] * sigma[i]
                    response_vars_test[i] = predictor_vars_test[i].dot(beta[:, i]) + test_noise

                else:
                    train_noise = gaussian_noise[nsamples_train_cumsum[i - 1]:nsamples_train_cumsum[i]] * sigma[i]
                    response_vars_train[i] = predictor_vars_train[i].dot(beta[:, i]) + train_noise

                    test_noise = gaussian_noise[nsamples.sum() + nsamples_test_cumsum[i - 1]:
                                                nsamples.sum() + nsamples_test_cumsum[i]] * sigma[i]
                    response_vars_test[i] = predictor_vars_test[i].dot(beta[:, i]) + test_noise

            gaussian_noise = gaussian_noise[nsamples.sum() + nsamples_test.sum():]

        print(n, "n sim")
        print(weight, "weight")

        # Record results for multi-task selective inference
        coverage, length, pivot, sns, spc, err = test_joint_mtl_si(predictor_vars_train,
                                                                   response_vars_train,
                                                                   predictor_vars_test,
                                                                   response_vars_test,
                                                                   beta,
                                                                   gaussian_noise,
                                                                   sigma,
                                                                   weight=weight[0],
                                                                   randomizer_scale=0.7)
        print(np.mean(coverage))
        if list(coverage):
            cov.append(np.mean(np.asarray(coverage)))
            len1.extend(length)
            pivots.extend(pivot)
        sensitivity_list.append(sns)
        specificity_list.append(spc)
        test_error_list.append(err)

        # Record results for data splitting v1
        coverage_data_splitting, length_data_splitting, pivot_data_splitting, sns_ds, spc_ds, ds_error = \
            test_data_splitting(predictor_vars_train,
                                response_vars_train,
                                predictor_vars_test,
                                response_vars_test,
                                beta,
                                sigma,
                                weight[1],
                                split=0.67)

        if list(coverage_data_splitting):
            cov_data_splitting.append(np.mean(np.asarray(coverage_data_splitting)))
            len_data_splitting.extend(length_data_splitting)
            pivots_data_splitting.extend(pivot_data_splitting)
        sensitivity_list_ds.append(sns_ds)
        specificity_list_ds.append(spc_ds)
        data_splitting_test_error_list.append(ds_error)

        # Record results for data splitting v2
        coverage_data_splitting2, length_data_splitting2, pivot_data_splitting2, sns_ds2, spc_ds2, ds_error2 = \
            test_data_splitting(predictor_vars_train,
                                response_vars_train,
                                predictor_vars_test,
                                response_vars_test,
                                beta,
                                sigma,
                                weight[2],
                                split=0.5)

        if list(coverage_data_splitting2):
            cov_data_splitting2.append(np.mean(np.asarray(coverage_data_splitting2)))
            len_data_splitting2.extend(length_data_splitting2)
            pivots_data_splitting2.extend(pivot_data_splitting2)
        sensitivity_list_ds2.append(sns_ds2)
        specificity_list_ds2.append(spc_ds2)
        data_splitting_test_error_list2.append(ds_error2)

        # Record results for single-task selective inference
        coverage_single_task_selective, length_single_task_selective, pivot_single_task_selective, sns_single_task, \
            spc_single_task, err_single_selective = test_lasso_si(predictor_vars_train,
                                                                  response_vars_train,
                                                                  predictor_vars_test,
                                                                  response_vars_test,
                                                                  beta,
                                                                  gaussian_noise,
                                                                  sigma,
                                                                  weight[3],
                                                                  randomizer_scale=0.7)

        if list(coverage_single_task_selective):
            cov_single_task_selective.append(np.mean(np.asarray(coverage_single_task_selective)))
            len_single_task_selective.extend(length_single_task_selective)
        sensitivity_list_single_task_selective.append(sns_single_task)
        specificity_list_single_task_selective.append(spc_single_task)
        single_task_selective_test_error_list.append(err_single_selective)

    return ({"MTL_SI_07_pivots": pivots,
             "DS_67_pivots": pivots_data_splitting,
             "MTL_SI_07_coverage": np.asarray(cov),
             "DS_67_coverage": np.asarray(cov_data_splitting),
             "DS_50_coverage": np.asarray(cov_data_splitting2),
             "LASSO_SI_07_coverage": np.asarray(cov_single_task_selective),
             "MTL_SI_07_length": np.asarray(len1),
             "DS_67_length": np.asarray(len_data_splitting),
             "DS_50_length": np.asarray(len_data_splitting2),
             "LASSO_SI_07_length": np.asarray(len_single_task_selective),
             "MTL_SI_07_sensitivity": np.asarray(sensitivity_list),
             "DS_67_sensitivity": np.asarray(sensitivity_list_ds),
             "DS_50_sensitivity": np.asarray(sensitivity_list_ds2),
             "LASSO_SI_07_sensitivity": np.asarray(sensitivity_list_single_task_selective),
             "MTL_SI_07_specificity": np.asarray(specificity_list),
             "DS_67_specificity": np.asarray(specificity_list_ds),
             "DS_50_specificity": np.asarray(specificity_list_ds2),
             "LASSO_SI_07_specificity": np.asarray(specificity_list_single_task_selective),
             "MTL_SI_07_error": np.mean(np.asarray(test_error_list)),
             "DS_67_error": np.mean(np.asarray(data_splitting_test_error_list)),
             "DS_50_error": np.mean(np.asarray(data_splitting_test_error_list2)),
             "LASSO_SI_07_error": np.mean(np.asarray(single_task_selective_test_error_list))})


def test_inference_joint_vs_pairwise(weight, signal, p, structure, gs, nsim=100, seed=5):
    np.random.seed(seed)
    # Track intervals, pivots, sensitivity, specificity, and testing error for joint MTL + SI
    cov_joint, len_joint, pivots_joint, sensitivity_list_joint, specificity_list_joint, test_error_list_joint = \
        ([] for _ in range(6))

    # Track intervals, pivots, sensitivity, specificity, and testing error for pairwise MTL + SI
    cov_separate, len_separate, sensitivity_list_separate, specificity_list_separate, test_error_list_separate = \
        ([] for _ in range(5))

    # Generate training and testing data
    ntask = 4
    nsamples = 5000 * np.ones(ntask)
    nsamples_test = 1000 * np.ones(ntask)
    p = p
    global_sparsity = gs
    sigma = 1. * np.ones(ntask)
    signal_fac = np.array(signal)
    rhos = 0.0 * np.ones(ntask)
    nsamples = nsamples.astype(int)
    nsamples_test = nsamples_test.astype(int)
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    response_vars_train, predictor_vars_train, response_vars_test, predictor_vars_test, beta, gaussian_noise = \
        gaussian_multitask_instance_joint_pairwise(ntask,
                                                   nsamples,
                                                   nsamples_test,
                                                   p,
                                                   global_sparsity,
                                                   structure,
                                                   sigma,
                                                   signal,
                                                   rhos,
                                                   random_signs=True,
                                                   equicorrelated=True)[:6]

    for n in range(nsim):

        # For each iteration, generate independent training errors, testing errors, and randomization variables
        # Compute new responses from the new errors
        # Note: Beta is already multiplied by sigma
        if n >= 1:

            gaussian_noise = _noise(nsamples.sum() + nsamples_test.sum() + p * ntask)
            response_vars_train = {}
            response_vars_test = {}
            nsamples_train_cumsum = np.cumsum(nsamples)
            nsamples_test_cumsum = np.cumsum(nsamples_test)

            for i in range(ntask):
                if i == 0:
                    train_noise = gaussian_noise[:nsamples_train_cumsum[i]] * sigma[i]
                    response_vars_train[i] = predictor_vars_train[i].dot(beta[:, i]) + train_noise

                    test_noise = gaussian_noise[nsamples.sum():nsamples.sum() + nsamples_test_cumsum[i]] * sigma[i]
                    response_vars_test[i] = predictor_vars_test[i].dot(beta[:, i]) + test_noise
                else:
                    train_noise = gaussian_noise[nsamples_train_cumsum[i - 1]: nsamples_train_cumsum[i]] * sigma[i]
                    response_vars_train[i] = predictor_vars_train[i].dot(beta[:, i]) + train_noise

                    test_noise = gaussian_noise[nsamples.sum() + nsamples_test_cumsum[i - 1]:
                                                nsamples.sum() + nsamples_test_cumsum[i]] * sigma[i]
                    response_vars_test[i] = predictor_vars_test[i].dot(beta[:, i]) + test_noise

            gaussian_noise = gaussian_noise[nsamples.sum() + nsamples_test.sum():]

        print(n, "n sim")
        print(weight, "weight")

        print("joint")

        # Record results for joint multi-task selective inference
        coverage, length, pivot, sns, spc, err = test_joint_mtl_si(predictor_vars_train,
                                                                   response_vars_train,
                                                                   predictor_vars_test,
                                                                   response_vars_test,
                                                                   beta,
                                                                   gaussian_noise,
                                                                   sigma,
                                                                   weight=weight[0],
                                                                   randomizer_scale=0.7)
        print(np.mean(coverage))
        if list(coverage):
            cov_joint.append(np.mean(np.asarray(coverage)))
            len_joint.extend(length)
            pivots_joint.extend(pivot)
        sensitivity_list_joint.append(sns)
        specificity_list_joint.append(spc)
        test_error_list_joint.append(err)

        print('pairwise')

        # Record results for pairwise (separate) multi-task selective inference
        coverage_separate, length_separate, sns_separate, spc_separate, err_separate = \
            test_pairwise_mtl_si(
                predictor_vars_train,
                response_vars_train,
                predictor_vars_test,
                response_vars_test,
                beta,
                gaussian_noise,
                sigma,
                weight=weight[0],
                randomizer_scale=0.7)

        print(np.mean(coverage_separate))
        if list(coverage_separate):
            cov_separate.append(np.mean(np.asarray(coverage_separate)))
            len_separate.extend(length_separate)
        sensitivity_list_separate.append(sns_separate)
        specificity_list_separate.append(spc_separate)
        test_error_list_separate.append(err_separate)

    return ({"joint_coverage": np.asarray(cov_joint),
             "separate_coverage": np.asarray(cov_separate),
             "joint_length": np.asarray(len_joint),
             "separate_length": np.asarray(len_separate),
             "joint_sensitivity": np.asarray(sensitivity_list_joint),
             "separate_sensitivity": np.asarray(sensitivity_list_separate),
             "joint_specificity": np.asarray(specificity_list_joint),
             "separate_specificity": np.asarray(specificity_list_separate),
             "joint_error": np.mean(np.asarray(test_error_list_joint)),
             "separate_error": np.mean(np.asarray(test_error_list_separate))})


def test_inference_error_comparison(weight, signal, p, ts, gs, nsim=100, seed=5):
    np.random.seed(seed)
    coverage_gaussian = []
    coverage_exponential = []
    coverage_laplace = []

    error_gaussian = []
    error_exponential = []
    error_laplace = []

    # Generate training and testing data
    ntask = 4
    nsamples = 5000 * np.ones(ntask)
    nsamples_test = 1000 * np.ones(ntask)
    p = p
    global_sparsity = gs
    task_sparsity = ts
    sigma = 1. * np.ones(ntask)
    signal_fac = np.array(signal)
    rhos = 0.0 * np.ones(ntask)
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

    for n in range(nsim):

        # For each iteration, generate independent training errors, testing errors, and randomization variables
        # Compute new responses from the new errors
        randomization_noise = _noise(p * ntask)
        gaussian_noise = _noise(nsamples.sum() + nsamples_test.sum())
        laplace_noise = _laplace_response_noise(nsamples.sum() + nsamples_test.sum())
        exponential_noise = _exponential_response_noise(nsamples.sum() + nsamples_test.sum())

        gaussian_response_vars_train = {}
        gaussian_response_vars_test = {}
        exponential_response_vars_train = {}
        exponential_response_vars_test = {}
        laplace_response_vars_train = {}
        laplace_response_vars_test = {}

        nsamples_train_cumsum = np.cumsum(nsamples)
        nsamples_test_cumsum = np.cumsum(nsamples_test)

        for i in range(ntask):
            if i == 0:
                train_noise_gaus = gaussian_noise[:nsamples_train_cumsum[i]] * sigma[i]
                gaussian_response_vars_train[i] = predictor_vars_train[i].dot(beta[:, i]) + train_noise_gaus

                test_noise_gaus = gaussian_noise[nsamples.sum():nsamples.sum() + nsamples_test_cumsum[i]] * sigma[i]
                gaussian_response_vars_test[i] = predictor_vars_test[i].dot(beta[:, i]) + test_noise_gaus

                train_noise_exp = exponential_noise[:nsamples_train_cumsum[i]] * sigma[i]
                exponential_response_vars_train[i] = predictor_vars_train[i].dot(beta[:, i]) + train_noise_exp

                test_noise_exp = exponential_noise[nsamples.sum():nsamples.sum() + nsamples_test_cumsum[i]] * sigma[i]
                exponential_response_vars_test[i] = predictor_vars_test[i].dot(beta[:, i]) + test_noise_exp

                train_noise_laplace = laplace_noise[:nsamples_train_cumsum[i]] * sigma[i]
                laplace_response_vars_train[i] = predictor_vars_train[i].dot(beta[:, i]) + train_noise_laplace

                test_noise_lapalce = laplace_noise[nsamples.sum():nsamples.sum() + nsamples_test_cumsum[i]] * sigma[i]
                laplace_response_vars_test[i] = predictor_vars_test[i].dot(beta[:, i]) + test_noise_lapalce
            else:
                train_noise_gaus = gaussian_noise[nsamples_train_cumsum[i - 1]:nsamples_train_cumsum[i]] * sigma[i]
                gaussian_response_vars_train[i] = predictor_vars_train[i].dot(beta[:, i]) + train_noise_gaus

                test_noise_gaus = gaussian_noise[nsamples.sum() + nsamples_test_cumsum[i - 1]:
                                                 nsamples.sum() + nsamples_test_cumsum[i]] * sigma[i]
                gaussian_response_vars_test[i] = predictor_vars_test[i].dot(beta[:, i]) + test_noise_gaus

                train_noise_exp = exponential_noise[nsamples_train_cumsum[i - 1]: nsamples_train_cumsum[i]] * sigma[i]
                exponential_response_vars_train[i] = predictor_vars_train[i].dot(beta[:, i]) + train_noise_exp

                test_noise_exp = exponential_noise[nsamples.sum() + nsamples_test_cumsum[i - 1]:
                                                   nsamples.sum() + nsamples_test_cumsum[i]] * sigma[i]
                exponential_response_vars_test[i] = predictor_vars_test[i].dot(beta[:, i]) + test_noise_exp

                train_noise_lapalce = laplace_noise[nsamples_train_cumsum[i - 1]:nsamples_train_cumsum[i]] * sigma[i]
                laplace_response_vars_train[i] = predictor_vars_train[i].dot(beta[:, i]) + train_noise_lapalce

                test_noise_laplace = laplace_noise[nsamples.sum() + nsamples_test_cumsum[i - 1]:
                                                   nsamples.sum() + nsamples_test_cumsum[i]] * sigma[i]
                laplace_response_vars_test[i] = predictor_vars_test[i].dot(beta[:, i]) + test_noise_laplace

        print(n, "n sim")
        print(weight, "weight")

        # Record results for multi-task selective inference v1
        coverage, length, pivot, sns, spc, err = test_joint_mtl_si(predictor_vars_train,
                                                                   gaussian_response_vars_train,
                                                                   predictor_vars_test,
                                                                   gaussian_response_vars_test,
                                                                   beta,
                                                                   randomization_noise,
                                                                   sigma,
                                                                   weight=weight[0],
                                                                   randomizer_scale=0.7)
        print(np.mean(np.asarray(coverage)))
        if list(coverage):
            coverage_gaussian.append(np.mean(np.asarray(coverage)))
        print(err, "error")
        error_gaussian.append(err)

        # Record results for data splitting v1
        coverage, length, pivot, sns, spc, err = test_joint_mtl_si(predictor_vars_train,
                                                                   exponential_response_vars_train,
                                                                   predictor_vars_test,
                                                                   exponential_response_vars_test,
                                                                   beta,
                                                                   randomization_noise,
                                                                   sigma,
                                                                   weight=weight[0],
                                                                   randomizer_scale=0.7)
        print(np.mean(np.asarray(coverage)))
        if list(coverage):
            coverage_exponential.append(np.mean(np.asarray(coverage)))
        error_exponential.append(err)

        # Record results for data splitting v2
        coverage, length, pivot, sns, spc, err = test_joint_mtl_si(predictor_vars_train,
                                                                   laplace_response_vars_train,
                                                                   predictor_vars_test,
                                                                   laplace_response_vars_test,
                                                                   beta,
                                                                   randomization_noise,
                                                                   sigma,
                                                                   weight=weight[0],
                                                                   randomizer_scale=0.7)
        print(np.mean(np.asarray(coverage)))
        if list(coverage):
            coverage_laplace.append(np.mean(np.asarray(coverage)))
        error_laplace.append(err)

    return ({"Gaussian_coverage": np.asarray(coverage_gaussian),
             "Exponential_coverage": np.asarray(coverage_exponential),
             "Laplace_coverage": np.asarray(coverage_laplace),
             "Gaussian_error": np.mean(np.asarray(error_gaussian)),
             "Exponential_error": np.mean(np.asarray(error_exponential)),
             "Laplace_error": np.mean(np.asarray(error_laplace))})
