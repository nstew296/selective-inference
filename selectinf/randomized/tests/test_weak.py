import numpy as np
#from Tkinter import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import norm as ndist
from scipy.stats import t as tdist
import random

from selectinf.randomized.multitask_lasso import multi_task_lasso
from selectinf.tests.instance import gaussian_multitask_instance
from selectinf.tests.instance import logistic_multitask_instance
from selectinf.tests.instance import poisson_multitask_instance
from selectinf.randomized.lasso import lasso, selected_targets


def test_multitask_lasso_hetero(predictor_vars_train,
                                response_vars_train,
                                predictor_vars_test,
                                response_vars_test,
                                beta,
                                gaussian_noise,
                                sigma,
                                link = "identity",
                                weight = 1.0,
                                randomizer_scale = 1.0):

    ntask = len(predictor_vars_train.keys())
    nsamples_test = np.asarray([np.shape(predictor_vars_test[i])[0] for i in range(ntask)])
    p = np.shape(beta)[0]

    feature_weight = weight * np.ones(p)
    sigmas_ = sigma
    randomizer_scales = randomizer_scale * np.array([sigmas_[i] for i in range(ntask)])
    _initial_omega = np.array(
        [randomizer_scales[i] * gaussian_noise[(i * p):((i + 1) * p)] for i in range(ntask)]).T

    if link == "identity":
        try:
            multi_lasso = multi_task_lasso.gaussian(predictor_vars_train,
                                                response_vars_train,
                                                feature_weight,
                                                ridge_term=None,
                                                randomizer_scales=randomizer_scales)

            active_signs = multi_lasso.fit(perturbations=_initial_omega)

        except:
            active_signs = np.asarray([])

    if link == "logit":
        try:
            multi_lasso = multi_task_lasso.logistic(predictor_vars_train,
                                                response_vars_train,
                                                feature_weight,
                                                ridge_term=None,
                                                randomizer_scales=randomizer_scales)

            active_signs = multi_lasso.fit(perturbations=_initial_omega)

        except:
            active_signs=np.asarray([])

    if link == "log":
        try:
            multi_lasso = multi_task_lasso.poisson(predictor_vars_train,
                                               response_vars_train,
                                               feature_weight,
                                               ridge_term=None,
                                               randomizer_scales=randomizer_scales)
            active_signs = multi_lasso.fit(perturbations=_initial_omega)

        except:
            active_signs= np.asarray([])

    coverage = []
    pivot = []
    intervals = np.asarray([[np.nan,np.nan]])

    if (active_signs != 0).sum() > 0:

        dispersions = sigma ** 2

        estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(
            dispersions=dispersions)

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
                error += (0.5 * np.sum(np.square(response_vars_test[j])))/nsamples_test[j]
                continue
            error += 0.5 * (np.sum(
                np.square((response_vars_test[j] - (predictor_vars_test[j])[:, (active_signs[:, j] != 0)].dot(
                    estimate[idx:idx + idx_new])))))/nsamples_test[j]
            idx = idx + idx_new

    else:
        error=0
        for j in range(ntask):
            error += (0.5 * np.linalg.norm(response_vars_test[j], 2) ** 2)/nsamples_test[j]


    # Compute snesitivity and specificity after selection
    # true_active = np.transpose(np.nonzero(beta))
    # selected_active = np.transpose(np.nonzero(active_signs))
    # num_true_positive = np.sum(x in true_active.tolist() for x in selected_active.tolist())
    # num_false_positive = np.sum(x not in true_active.tolist() for x in selected_active.tolist())
    # num_positive = np.shape(true_active)[0]
    # num_negative = np.shape(beta)[0]*np.shape(beta)[1]-num_positive
    # sensitivity = np.float(num_true_positive)/np.float(num_positive)
    # specificity = 1.0 - np.float(num_false_positive)/np.float(num_negative)

    # Compute snesitivity and specificity after inference
    true_active = np.transpose(np.nonzero(np.transpose(beta)))
    num_positive = np.shape(true_active)[0]
    if (active_signs != 0).sum() > 0:
        selected_active = np.transpose(np.nonzero(np.transpose(active_signs)))
        true_positive_selected = [x in true_active.tolist() for x in selected_active.tolist()]
        num_true_positive_inference = np.sum([true_positive_selected[i]*(intervals[i,1]<0 or intervals[i,0]>0) for i in range(len(true_positive_selected))])
        num_false_positive_inference = np.sum([(intervals[i,1]<0 or intervals[i,0]>0) for i in range(len(true_positive_selected))]) - num_true_positive_inference
    else:
        num_true_positive_inference = 0
        num_false_positive_inference = 0
    num_negative = np.shape(beta)[0]*np.shape(beta)[1]-num_positive
    sensitivity_inference = np.float(num_true_positive_inference)/np.maximum(np.float(num_positive),1)
    specificity_inference = 1.0 - np.float(num_false_positive_inference)/np.maximum(np.float(num_negative),1)

    return np.asarray(coverage), intervals[:,1]-intervals[:,0], pivot, sensitivity_inference, specificity_inference, error


def test_multitask_lasso_naive_hetero(predictor_vars_train,
                                      response_vars_train,
                                      predictor_vars_test,
                                      response_vars_test,
                                      beta,
                                      sigma,
                                      weight = 1.0,
                                      link = "identity"):

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
    sensitivity_inference = np.float(num_true_positive_inference) / np.maximum(np.float(num_positive),1)
    specificity_inference = 1.0 - np.float(num_false_positive_inference) / np.maximum(np.float(num_negative),1)


    return np.asarray(coverage), CIs[1:, 1] - CIs[1:, 0], pivot, sensitivity_inference, specificity_inference, error


def test_multitask_lasso_data_splitting(predictor_vars_train,
                                      response_vars_train,
                                      predictor_vars_test,
                                      response_vars_test,
                                      beta,
                                      sigma,
                                      weight = 1.0,
                                      link = "identity"):


    ntask = len(predictor_vars_train.keys())
    nsamples = np.asarray([np.shape(predictor_vars_train[i])[0] for i in range(ntask)])
    nsamples_test = np.asarray([np.shape(predictor_vars_test[i])[0] for i in range(ntask)])
    p = np.shape(beta)[0]

    samples = np.arange(np.int(nsamples[0]))
    selection = np.random.choice(samples, size=np.int(0.5 * nsamples[0]), replace=False)
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


def test_single_task_lasso_posi_hetero(predictor_vars_train,
                                      response_vars_train,
                                      predictor_vars_test,
                                      response_vars_test,
                                      beta,
                                      sigma,
                                      weight,
                                      link = "identity"):

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
        #W[0] = 0
        single_task_lasso = lasso.gaussian(predictor_vars_train[i],
                     response_vars_train[i],
                     W,
                     sigma=sigma[i],
                     randomizer_scale=1.0)

        signs = single_task_lasso.fit()
        nonzero = signs != 0

        (observed_target, cov_target, cov_target_score, alternatives) = \
            selected_targets(single_task_lasso.loglike, single_task_lasso._W, nonzero)


        try:
            MLE_result, observed_info_mean, _ = single_task_lasso.selective_MLE(
                observed_target,
                cov_target,
                cov_target_score)

            final_estimator = np.asarray(MLE_result['MLE'])

            alpha = 1. - 0.90
            quantile = ndist.ppf(1 - alpha / 2.)

            intervals = np.vstack([final_estimator - quantile * np.sqrt(np.diag(observed_info_mean)),
                               final_estimator + quantile * np.sqrt(np.diag(observed_info_mean))]).T

            beta_target = np.linalg.pinv(predictor_vars_train[i][:, nonzero]).dot(predictor_vars_train[i].dot(beta[:, i]))

            coverage.extend((beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1]))

            pivot_ = ndist.cdf((final_estimator - beta_target) / np.sqrt(np.diag(observed_info_mean)))
            pivot.extend(2 * np.minimum(pivot_, 1. - pivot_))
            CIs = np.vstack([CIs, intervals])

            selected_active.extend([[i, j] for j in np.nonzero(signs)[0]])

        except:
            pass

        idx_new = np.sum(signs != 0)
        if idx_new == 0:
            error += (0.5 * np.sum(np.square(response_vars_test[i]))) / nsamples_test[i]
            continue
        error += (0.5 * np.sum(np.square(
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

    return np.asarray(coverage), CIs[1:, 1] - CIs[1:, 0], np.asarray(pivot), sensitivity_inference, specificity_inference, error


def test_one_lasso_posi(predictor_vars_train,
                                      response_vars_train,
                                      predictor_vars_test,
                                      response_vars_test,
                                      beta,
                                      sigma,
                                      weight,
                                      link = "identity",
                                      randomizer_scale=1.0):

    ntask = len(predictor_vars_train.keys())
    nsamples_test = np.asarray([np.shape(predictor_vars_test[i])[0] for i in range(ntask)])
    p = np.shape(beta)[0]

    coverage = []
    pivot = []
    CIs = [[0, 0]]
    error = 0
    selected_active = []

    predictors_train = predictor_vars_train[0]
    responses_train = response_vars_train[0]

    for i in range(ntask-1):

        predictors_train = np.concatenate((predictors_train,predictor_vars_train[i+1]),axis=0)
        responses_train = np.concatenate((responses_train,response_vars_train[i+1]))

    try:
        W = np.ones(p) * weight
        single_task_lasso = lasso.gaussian(predictors_train,
                                           responses_train,
                                           W,
                                           sigma=np.std(responses_train),
                                           randomizer_scale=1.0)

        signs = single_task_lasso.fit()
        nonzero = signs != 0

        (observed_target, cov_target, cov_target_score, alternatives) = \
            selected_targets(single_task_lasso.loglike, single_task_lasso._W, nonzero)

        MLE_result, observed_info_mean, _ = single_task_lasso.selective_MLE(
            observed_target,
            cov_target,
            cov_target_score)

        final_estimator = np.asarray(MLE_result['MLE'])

        alpha = 1. - 0.90
        quantile = ndist.ppf(1 - alpha / 2.)

        intervals = np.vstack([final_estimator - quantile * np.sqrt(np.diag(observed_info_mean)),
                            final_estimator + quantile * np.sqrt(np.diag(observed_info_mean))]).T


        if np.sum(signs != 0) == 0:
            for i in range(ntask):
                error += (0.5 * np.sum(np.square(response_vars_test[i]))) / nsamples_test[i]

        else:

            for i in range(ntask):
                beta_target = np.linalg.pinv(predictor_vars_train[i][:, nonzero]).dot(predictor_vars_train[i].dot(beta[:, i]))
                coverage.extend((beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1]))

                pivot_ = ndist.cdf((final_estimator - beta_target) / np.sqrt(np.diag(observed_info_mean)))
                pivot.extend(2 * np.minimum(pivot_, 1. - pivot_))
                CIs = np.vstack([CIs, intervals])

                selected_active.extend([[i, j] for j in np.nonzero(signs)[0]])

                error += (0.5 * np.sum(np.square(response_vars_test[i] - (predictor_vars_test[i])[:, nonzero].dot(
                    final_estimator)))) / nsamples_test[i]

    except:

        for i in range(ntask):
            error += (0.5 * np.sum(np.square(response_vars_test[i]))) / nsamples_test[i]


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


    return np.asarray(coverage), CIs[1:, 1] - CIs[1:, 0], np.asarray(pivot), sensitivity_inference, specificity_inference, error



def test_coverage(weight,signal,nsim=100):
    np.random.seed(5)
    cov = []
    len = []
    pivots = []
    sensitivity_list = []
    specificity_list = []
    test_error_list = []

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

    cov_single_task_selective = []
    len_single_task_selective = []
    sensitivity_list_single_task_selective = []
    specificity_list_single_task_selective = []
    single_task_selective_test_error_list = []

    cov_one_lasso = []
    len_one_lasso = []
    sensitivity_one_lasso = []
    specificity_one_lasso = []
    one_lasso_test_error_list = []

    ntask = 5
    nsamples= 1000 * np.ones(ntask)
    p=100
    global_sparsity=0.9
    task_sparsity= 0.40
    sigma=1. * np.ones(ntask)
    signal_fac=np.array(signal)
    rhos=0.3 * np.ones(ntask)
    link="identity"

    nsamples = nsamples.astype(int)
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    if link == "identity":
        response_vars, predictor_vars, beta, gaussian_noise = gaussian_multitask_instance(ntask,
                                                                                           nsamples,
                                                                                           p,
                                                                                           global_sparsity,
                                                                                           task_sparsity,
                                                                                           sigma,
                                                                                           signal,
                                                                                           rhos,
                                                                                           random_signs=True,
                                                                                           equicorrelated=True)[:4]

    if link == "logit":
        response_vars, predictor_vars, beta, gaussian_noise = logistic_multitask_instance(ntask,
                                                                                           nsamples,
                                                                                           p,
                                                                                           global_sparsity,
                                                                                           task_sparsity,
                                                                                           sigma,
                                                                                           signal,
                                                                                           rhos,
                                                                                           random_signs=True,
                                                                                           equicorrelated=True)[:4]

    if link == "log":
        response_vars, predictor_vars, beta, gaussian_noise = poisson_multitask_instance(ntask,
                                                                                          nsamples,
                                                                                          p,
                                                                                          global_sparsity,
                                                                                          task_sparsity,
                                                                                          sigma,
                                                                                          signal,
                                                                                          rhos,
                                                                                          random_signs=True,
                                                                                          equicorrelated=True)[:4]


    for n in range(nsim):

        if n>=1:

            def _noise(n, df=np.inf):
                if df == np.inf:
                    return np.random.standard_normal(n)
                else:
                    sd_t = np.std(tdist.rvs(df, size=50000))
                return tdist.rvs(df, size=n) / sd_t

            if link == "identity":
                noise = _noise(nsamples.sum(), np.inf)
                response_vars = {}
                nsamples_cumsum = np.cumsum(nsamples)
                for i in range(ntask):
                    if i == 0:
                        response_vars[i] = (predictor_vars[i].dot(beta[:, i]) + noise[:nsamples_cumsum[i]]) * \
                                           sigma[i]
                    else:
                        response_vars[i] = (predictor_vars[i].dot(beta[:, i]) + noise[
                                                                                nsamples_cumsum[i - 1]:nsamples_cumsum[
                                                                                    i]]) * sigma[i]

            if link == "logit":
                response_vars = {}
                pis = {}
                for i in range(ntask):
                    pis[i] = predictor_vars[i].dot(beta[:, i]) * sigma[i]
                    response_vars[i] = np.random.binomial(1, np.exp(pis[i]) / (1.0 + np.exp(pis[i])))

            if link == "log":
                response_vars = {}
                pis = {}
                for i in range(ntask):
                    pis[i] = predictor_vars[i].dot(beta[:, i]) * sigma[i]
                    response_vars[i] = np.random.poisson(np.exp(pis[i]))


        print(n,"n sim")

        samples = np.arange(np.int(nsamples[0]))
        train = np.random.choice(samples, size=np.int(0.5*nsamples[0]), replace=False)
        test = np.setdiff1d(samples, train)

        response_vars_train = {j: response_vars[j][train] for j in range(ntask)}
        predictor_vars_train = {j: predictor_vars[j][train] for j in range(ntask)}

        response_vars_test = {j: response_vars[j][test] for j in range(ntask)}
        predictor_vars_test = {j: predictor_vars[j][test] for j in range(ntask)}


        coverage, length, pivot, sns, spc, err = test_multitask_lasso_hetero(predictor_vars_train,
                                                                             response_vars_train,
                                                                             predictor_vars_test,
                                                                             response_vars_test,
                                                                             beta,
                                                                             gaussian_noise,
                                                                             sigma,
                                                                             link="identity",
                                                                             weight=weight,
                                                                             randomizer_scale= 0.7)

        if coverage != []:
            cov.append(np.mean(np.asarray(coverage)))
            len.extend(length)
            pivots.extend(pivot)
        sensitivity_list.append(sns)
        specificity_list.append(spc)
        test_error_list.append(err)


        coverage_naive, length_naive, pivot_naive, naive_sensitivity, naive_specificity, naive_err = test_multitask_lasso_naive_hetero(predictor_vars_train,
                                                                             response_vars_train,
                                                                             predictor_vars_test,
                                                                             response_vars_test,
                                                                             beta,
                                                                             sigma,
                                                                             weight,
                                                                             link="identity")

        if coverage_naive != []:
            cov_naive.append(np.mean(np.asarray((coverage_naive))))
            len_naive.extend(length_naive)
            pivots_naive.extend(pivot_naive)
        sensitivity_list_naive.append(naive_sensitivity)
        specificity_list_naive.append(naive_specificity)
        naive_test_error_list.append(naive_err)




        coverage_data_splitting, length_data_splitting, pivot_data_splitting, sns_ds, spc_ds, ds_error = test_multitask_lasso_data_splitting(predictor_vars_train,
                                                                             response_vars_train,
                                                                             predictor_vars_test,
                                                                             response_vars_test,
                                                                             beta,
                                                                             sigma,
                                                                             weight,
                                                                             link="identity")

        if coverage_data_splitting!=[]:
            cov_data_splitting.append(np.mean(np.asarray(coverage_data_splitting)))
            len_data_splitting.extend(length_data_splitting)
            pivots_data_splitting.extend(pivot_data_splitting)
        sensitivity_list_ds.append(sns_ds)
        specificity_list_ds.append(spc_ds)
        data_splitting_test_error_list.append(ds_error)


        coverage_single_task_selective, length_single_task_selective, pivot_single_task_selective, sns_single_task, spc_single_task, err_single_selective = test_single_task_lasso_posi_hetero(predictor_vars_train,
                                      response_vars_train,
                                      predictor_vars_test,
                                      response_vars_test,
                                      beta,
                                      sigma,
                                      weight,
                                      link="identity")


        if coverage_single_task_selective!=[]:
            cov_single_task_selective.append(np.mean(np.asarray(coverage_single_task_selective)))
            len_single_task_selective.extend(length_single_task_selective)
        sensitivity_list_single_task_selective.append(sns_single_task)
        specificity_list_single_task_selective.append(spc_single_task)
        single_task_selective_test_error_list.append(err_single_selective)


        print("iteration completed ", n)
        print("posi coverage so far ", np.mean(np.asarray(cov)))
        print("naive coverage so far ", np.mean(np.asarray(cov_naive)))
        print("data splitting coverage so far ", np.mean(np.asarray(cov_data_splitting)))
        print("single-task selective inference coverage so far ", np.mean(np.asarray(cov_single_task_selective)))

        print("posi length so far ", np.mean(np.asarray(len)))
        print("naive length so far ", np.mean(np.asarray(len_naive)))
        print("data splitting length so far ", np.mean(np.asarray(len_data_splitting)))
        print("single task selective inference length so far ", np.mean(np.asarray(len_single_task_selective)))

        print("median sensitivity posi", np.median(np.asarray(sensitivity_list)))
        print("median specificity posi", np.median(np.asarray(specificity_list)))
        print("median sensitivity data splitting", np.median(np.asarray(sensitivity_list_ds)))
        print("median specificity data splitting", np.median(np.asarray(specificity_list_ds)))
        print("median sensitivity single lasso", np.median(np.asarray(sensitivity_list_single_task_selective)))
        print("median specificity signle lasso", np.median(np.asarray(specificity_list_single_task_selective)))

        print("error selective", np.median(np.asarray(test_error_list)))
        print("error naive", np.median(np.asarray(naive_test_error_list)))
        print("error ds", np.median(np.asarray(data_splitting_test_error_list)))
        print("error single task", np.median(np.asarray(single_task_selective_test_error_list)))

    return([pivots,pivots_naive,pivots_data_splitting,
            np.asarray(cov), np.asarray(cov_naive), np.asarray(cov_data_splitting), np.asarray(cov_single_task_selective),
            np.asarray(len),np.asarray(len_naive),np.asarray(len_data_splitting),np.asarray(len_single_task_selective),
            np.mean(np.asarray(sensitivity_list)),np.mean(np.asarray(sensitivity_list_naive)),np.mean(np.asarray(sensitivity_list_ds)),np.mean(np.asarray(sensitivity_list_single_task_selective)),
            np.mean(np.asarray(specificity_list)),np.mean(np.asarray(specificity_list_naive)),np.mean(np.asarray(specificity_list_ds)),np.mean(np.asarray(specificity_list_single_task_selective)), np.mean(np.asarray(test_error_list)),np.mean(np.asarray(naive_test_error_list)),
            np.mean(np.asarray(data_splitting_test_error_list)),np.mean(np.asarray(single_task_selective_test_error_list))])

def main():

    #random.seed(5)

    #signals = [[0.2,0.5],[1.0,3.0],[3.0,5.0],[5.0,8.0]]
    #tuning = {0: [], 1: [], 2: [], 3: []}
    #pivot = {0:[],1:[],2:[],3:[]}
    #pivot_naive = {0:[], 1:[],2:[],3:[]}

    #KL_divergence = {0: [[],[]], 1: [[],[]], 2: [[],[]], 3: [[],[]]}
    #length = {0: [[], [], []], 1: [[], [], []], 2: [[], [], []], 3: [[], [], []]}
    #coverage = {0: [[], [], []], 1: [[], [], []], 2: [[], [], []], 3: [[], [], []]}

    #sensitivity = {0: [], 1: [], 2: [], 3: []}
    #specificity = {0: [], 1: [], 2: [], 3: []}
    #prediction_error_posi = {0: [], 1: [], 2: [], 3: []}

    #KL Divergence plot
    # #for j in range(50):

        #for i in range(len(signals)):
            #sims = test_coverage(signals[i],50)
            #pivot[i] = sims[0]
            #pivot_naive[i] = sims[1]

        #pivots = pivot[0]
        #pivots_naive = pivot_naive[0]
        #plt.clf()
        #grid = np.linspace(0, 1, 101)
        #points = [np.max(np.searchsorted(np.sort(np.asarray(pivots)), i, side='right')) for i in np.linspace(0, 1, 101)]
        #points.append(np.float(np.shape(pivots)[0]))
        #p = np.diff(points) / np.float(np.shape(pivots)[0])
        #dist_posi = np.sum([p[i] * np.log((p[i] + 0.00001) / 0.01) for i in range(100)])
        #points_naive = [np.max(np.searchsorted(np.sort(np.asarray(pivots_naive)), i, side='right')) for i in
                       # np.linspace(0, 1, 101)]
        #points_naive.append(np.float(np.shape(pivots_naive)[0]))
        #q = np.diff(points_naive) / np.float(np.shape(pivots_naive)[0])
        #dist_naive = np.sum([q[i] * np.log((q[i] + 0.00001) / 0.01) for i in range(100)])
        #KL_divergence[0][0].append(dist_posi)
        #KL_divergence[0][1].append(dist_naive)

        #pivots = pivot[1]
        #pivots_naive = pivot_naive[1]
        #grid = np.linspace(0, 1, 101)
        #points = [np.max(np.searchsorted(np.sort(np.asarray(pivots)), i, side='right')) for i in np.linspace(0, 1, 101)]
        #points.append(np.float(np.shape(pivots)[0]))
        #p = np.diff(points) / np.float(np.shape(pivots)[0])
        #dist_posi = np.sum([p[i] * np.log((p[i] + 0.00001) / 0.01) for i in range(100)])
        #points_naive = [np.max(np.searchsorted(np.sort(np.asarray(pivots_naive)), i, side='right')) for i in
                        #np.linspace(0, 1, 101)]
        #points_naive.append(np.float(np.shape(pivots_naive)[0]))
        #q = np.diff(points_naive) / np.float(np.shape(pivots_naive)[0])
        #dist_naive = np.sum([q[i] * np.log((q[i] + 0.00001) / 0.01) for i in range(100)])
        #KL_divergence[1][0].append(dist_posi)
        #KL_divergence[1][1].append(dist_naive)

        #pivots = pivot[2]
        #pivots_naive = pivot_naive[2]
        #grid = np.linspace(0, 1, 101)
        #points = [np.max(np.searchsorted(np.sort(np.asarray(pivots)), i, side='right')) for i in np.linspace(0, 1, 101)]
        #points.append(np.float(np.shape(pivots)[0]))
        #p = np.diff(points) / np.float(np.shape(pivots)[0])
        #dist_posi = np.sum([p[i] * np.log((p[i] + 0.00001) / 0.01) for i in range(100)])
        #points_naive = [np.max(np.searchsorted(np.sort(np.asarray(pivots_naive)), i, side='right')) for i in
                   #     np.linspace(0, 1, 101)]
        #points_naive.append(np.float(np.shape(pivots_naive)[0]))
        #q = np.diff(points_naive) / np.float(np.shape(pivots_naive)[0])
        #dist_naive = np.sum([q[i] * np.log((q[i] + 0.00001) / 0.01) for i in range(100)])
        #KL_divergence[2][0].append(dist_posi)
        #KL_divergence[2][1].append(dist_naive)

        #pivots = pivot[3]
        #pivots_naive = pivot_naive[3]
        #grid = np.linspace(0, 1, 101)
        #points = [np.max(np.searchsorted(np.sort(np.asarray(pivots)), i, side='right')) for i in np.linspace(0, 1, 101)]
        #points.append(np.float(np.shape(pivots)[0]))
        #p = np.diff(points) / np.float(np.shape(pivots)[0])
        #dist_posi = np.sum([p[i] * np.log((p[i] + 0.00001) / 0.01) for i in range(100)])
        #points_naive = [np.max(np.searchsorted(np.sort(np.asarray(pivots_naive)), i, side='right')) for i in
            #            np.linspace(0, 1, 101)]
        #points_naive.append(np.float(np.shape(pivots_naive)[0]))
        #q = np.diff(points_naive) / np.float(np.shape(pivots_naive)[0])
        #dist_naive = np.sum([q[i] * np.log((q[i] + 0.00001) / 0.01) for i in range(100)])
        #KL_divergence[3][0].append(dist_posi)
        #KL_divergence[3][1].append(dist_naive)

    # Boxplot of KL divergence
    # print(tuning)
    # print(KL_divergence)
    # fig = plt.figure(figsize=(32, 8))
    # fig.add_subplot(1, 4, 1)
    # plt.boxplot(KL_divergence[0], positions=[1, 2], widths=0.6)
    # plt.xticks([1, 2], ['POSI', 'Naive'])
    # plt.ylabel('KL Divergence')
    # plt.title('SNR 0.2-0.5')
    # fig.add_subplot(1, 4, 2)
    # plt.boxplot(KL_divergence[1], positions=[1, 2], widths=0.6)
    # plt.xticks([1, 2], ['POSI', 'Naive'])
    # plt.ylabel('KL Divergence')
    # plt.title('SNR 0.5-1.0')
    # fig.add_subplot(1, 4, 3)
    # plt.boxplot(KL_divergence[2], positions=[1, 2], widths=0.6)
    # plt.xticks([1, 2], ['POSI', 'Naive'])
    # plt.ylabel('KL Divergence')
    # plt.title('SNR 1.0-3.0')
    # fig.add_subplot(1, 4, 4)
    # plt.boxplot(KL_divergence[3], positions=[1, 2], widths=0.6)
    # plt.xticks([1, 2], ['POSI', 'Naive'])
    # plt.ylabel('KL Divergence')
    # plt.title('SNR 3.0-5.0')
    # plt.savefig("boxplot25.png")

    length_path = 10

    #Coverage, length, and pivot plots
    coverage = {i:[[],[],[],[]] for i in range(length_path)}
    length = {i:[[],[],[],[]] for i in range(length_path)}
    pivot = {i:[[],[],[]] for i in range(length_path)}
    sensitivity = {i:[[],[],[],[]] for i in range(length_path)}
    specificity = {i:[[],[],[],[]] for i in range(length_path)}
    error = {i:[[],[],[],[]] for i in range(length_path)}

    lambdamin = 0.75
    lambdamax = 3.5
    #weights = np.arange(np.log(lambdamin), np.log(lambdamax), (np.log(lambdamax) - np.log(lambdamin)) / (length_path))
    #feature_weight_list = np.exp(weights)
    feature_weight_list = np.arange(lambdamin, lambdamax,(lambdamax - lambdamin) / (length_path))
    print(feature_weight_list)

    for i in range(len(feature_weight_list)):
        sims = test_coverage(feature_weight_list[i],[0.2,3.0],100)
        pivot[i][0].extend(sims[0])
        pivot[i][1].extend(sims[1])
        pivot[i][2].extend(sims[2])
        coverage[i][0].extend(sims[3])
        coverage[i][1].extend(sims[4])
        coverage[i][2].extend(sims[5])
        coverage[i][3].extend(sims[6])
        length[i][0].extend(sims[7])
        length[i][1].extend(sims[8])
        length[i][2].extend(sims[9])
        length[i][3].extend(sims[10])
        sensitivity[i][0].append(sims[11])
        sensitivity[i][1].append(sims[12])
        sensitivity[i][2].append(sims[13])
        sensitivity[i][3].append(sims[14])
        specificity[i][0].append(sims[15])
        specificity[i][1].append(sims[16])
        specificity[i][2].append(sims[17])
        specificity[i][3].append(sims[18])
        error[i][0].append(sims[19])
        error[i][1].append(sims[20])
        error[i][2].append(sims[21])
        error[i][3].append(sims[22])

    selective_lengths = [length[i][0] for i in range(length_path)]
    naive_lengths = [length[i][1] for i in range(length_path)]
    ds_lengths = [length[i][2] for i in range(length_path)]
    single_selective_lengths = [length[i][3] for i in range(length_path)]

    selective_coverage = [coverage[i][0] for i in range(length_path)]
    naive_coverage = [coverage[i][1] for i in range(length_path)]
    ds_coverage = [coverage[i][2] for i in range(length_path)]
    single_selective_coverage = [coverage[i][3] for i in range(length_path)]

    selective_sensitivity = [sensitivity[i][0] for i in range(length_path)]
    naive_sensitivity = [sensitivity[i][1] for i in range(length_path)]
    ds_sensitivity = [sensitivity[i][2] for i in range(length_path)]
    single_task_sensitivity = [sensitivity[i][3] for i in range(length_path)]

    selective_specificity = [specificity[i][0] for i in range(length_path)]
    naive_specificity = [specificity[i][1] for i in range(length_path)]
    ds_specifity = [specificity[i][2] for i in range(length_path)]
    single_task_specifity = [specificity[i][3] for i in range(length_path)]

    selective_error = [error[i][0] for i in range(length_path)]
    naive_error = [error[i][1] for i in range(length_path)]
    ds_error = [error[i][2] for i in range(length_path)]
    single_selective_error = [error[i][3] for i in range(length_path)]

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    fig = plt.figure(figsize=(25, 10))
    first = plt.boxplot(selective_lengths, positions=np.array(xrange(length_path)) * 3, sym='', widths=0.3)
    second = plt.boxplot(naive_lengths, positions=np.array(xrange(length_path)) * 3 + .3, sym='', widths=0.3)
    third = plt.boxplot(ds_lengths, positions=np.array(xrange(length_path)) * 3 + .6, sym='', widths=0.3)
    fourth = plt.boxplot(single_selective_lengths, positions=np.array(xrange(length_path)) * 3 + 0.9, sym='',widths=0.3)
    set_box_color(first, '#2b8cbe')  # colors are from http://colorbrewer2.org/
    set_box_color(second, '#D7191C')
    set_box_color(third, '#31a354')
    set_box_color(fourth, '#feb24c')
    plt.plot([], c='#2b8cbe', label='Randomized Multi-Task Lasso')
    plt.plot([], c='#D7191C', label='Multi-Task Lasso')
    plt.plot([], c='#31a354', label='Data Splitting')
    plt.plot([], c='#feb24c', label='K Randomized Lassos')
    plt.legend()
    plt.xticks(xrange(1, (length_path) * 3 + 1, 3), feature_weight_list)
    plt.xlim(-1, (length_path - 1) * 3 + 3)
    max_length = np.max(np.concatenate((np.concatenate(selective_lengths),np.concatenate(naive_lengths),np.concatenate(ds_lengths),np.concatenate(single_selective_lengths))))
    plt.plot(np.argmin(selective_error)*3, max_length, 'ro',c='#2b8cbe')
    plt.plot(np.argmin(naive_error)* 3+.3, max_length, 'ro', c='#D7191C')
    plt.plot(np.argmin(ds_error) * 3 + .6, max_length, 'ro', c='#31a354')
    plt.plot(np.argmin(single_selective_error) * 3 + 0.9, max_length, 'ro', c='#feb24c')
    plt.tight_layout()
    plt.ylabel('Interval Length')
    plt.title('Interval Length Along Lambda Path')
    plt.savefig('lengthcompare_weak_ts40.png', bbox_inches='tight')

    fig = plt.figure(figsize=(25, 10))
    first = plt.boxplot(selective_coverage, positions=np.array(xrange(length_path)) * 3, sym='', widths=0.3)
    second = plt.boxplot(naive_coverage, positions=np.array(xrange(length_path)) * 3 + .3, sym='', widths=0.3)
    third = plt.boxplot(ds_coverage, positions=np.array(xrange(length_path)) * 3 + 0.6, sym='', widths=0.3)
    fourth = plt.boxplot(single_selective_coverage, positions=np.array(xrange(length_path)) * 3 + 0.9, sym='',widths=0.3)
    set_box_color(first, '#2b8cbe')  # colors are from http://colorbrewer2.org/
    set_box_color(second, '#D7191C')
    set_box_color(third, '#31a354')
    set_box_color(fourth, '#feb24c')
    plt.plot([], c='#2b8cbe', label='Randomized Multi-Task Lasso')
    plt.plot([], c='#D7191C', label='Multi-Task Lasso')
    plt.plot([], c='#31a354', label='Data Splitting')
    plt.plot([], c='#feb24c', label='K Randomized Lassos')
    plt.legend()
    plt.xticks(xrange(1, (length_path) * 3 + 1, 3), feature_weight_list)
    plt.xlim(-1, (length_path - 1) * 3 + 3)
    plt.plot(np.argmin(selective_error) * 3, 1.01, 'ro', c='#2b8cbe')
    plt.plot(np.argmin(naive_error) * 3 + .3, 1.01, 'ro', c='#D7191C')
    plt.plot(np.argmin(ds_error) * 3 + .6, 1.01, 'ro', c='#31a354')
    plt.plot(np.argmin(single_selective_error) * 3 + 0.9, 1.01, 'ro', c='#feb24c')
    plt.tight_layout()
    plt.ylabel('Coverage')
    plt.title('Coverage Along Lambda Path')
    plt.savefig('coveragecompare_weak_ts40.png', bbox_inches='tight')

    fig = plt.figure(figsize=(25, 10))
    fig.tight_layout()
    fig.add_subplot(1, 2, 1)
    plt.plot(feature_weight_list, selective_sensitivity, c='#2b8cbe')
    plt.plot(feature_weight_list, naive_sensitivity, c='#D7191C')
    plt.plot(feature_weight_list, ds_sensitivity, c='#31a354')
    plt.plot(feature_weight_list, single_task_sensitivity, c='#feb24c')
    plt.plot([], c='#2b8cbe', label='Randomized Multi-Task Lasso')
    plt.plot([], c='#D7191C', label='Multi-Task Lasso')
    plt.plot([], c='#31a354', label='Data Splitting')
    plt.plot([], c='#feb24c', label='K Randomized Lassos')
    plt.legend()
    min_errors = [np.argmin(selective_error), np.argmin(naive_error), np.argmin(ds_error),
                  np.argmin(single_selective_error)]
    plt.plot(feature_weight_list[np.argmin(selective_error)]-0.03*np.sum([min_errors[0]==x for x in min_errors[1:]]), 1.0, 'ro', c='#2b8cbe')
    min_errors.pop(0)
    plt.plot(feature_weight_list[np.argmin(naive_error)]-0.03*np.sum([min_errors[0]==x for x in min_errors[1:]]), 1.0, 'ro', c='#D7191C')
    min_errors.pop(0)
    plt.plot(feature_weight_list[np.argmin(ds_error)]-0.03*np.sum([min_errors[0]==x for x in min_errors[1:]]), 1.0, 'ro', c='#31a354')
    plt.plot(feature_weight_list[np.argmin(single_selective_error)], 1.0, 'ro', c='#feb24c')
    plt.tight_layout()
    plt.ylabel('Average Sensitivity')
    plt.xlabel('Lambda Value')
    plt.title('Sensitivity Along Lambda Path')
    fig.add_subplot(1, 2, 2)
    plt.plot(feature_weight_list, selective_specificity, c='#2b8cbe')
    plt.plot(feature_weight_list, naive_specificity, c='#D7191C')
    plt.plot(feature_weight_list, ds_specifity, c='#31a354')
    plt.plot(feature_weight_list, single_task_specifity, c='#feb24c')
    plt.plot([], c='#2b8cbe', label='Randomized Multi-Task Lasso')
    plt.plot([], c='#D7191C', label='Multi-Task Lasso')
    plt.plot([], c='#31a354', label='Data Splitting')
    plt.plot([], c='#feb24c', label='K Randomized Lassos')
    plt.legend()
    min_errors = [np.argmin(selective_error), np.argmin(naive_error), np.argmin(ds_error),
                  np.argmin(single_selective_error)]
    plt.plot(
        feature_weight_list[np.argmin(selective_error)] - 0.03 * np.sum([min_errors[0] == x for x in min_errors[1:]]),
        1.0, 'ro', c='#2b8cbe')
    min_errors.pop(0)
    plt.plot(feature_weight_list[np.argmin(naive_error)] - 0.03 * np.sum([min_errors[0] == x for x in min_errors[1:]]),
             1.0, 'ro', c='#D7191C')
    min_errors.pop(0)
    plt.plot(feature_weight_list[np.argmin(ds_error)] - 0.03 * np.sum([min_errors[0] == x for x in min_errors[1:]]),
             1.0, 'ro', c='#31a354')
    plt.plot(feature_weight_list[np.argmin(single_selective_error)], 1.0, 'ro', c='#feb24c')
    plt.tight_layout()
    plt.ylabel('Average Specificity')
    plt.xlabel('Lambda Value')
    plt.title('Specificity Along Lambda Path')
    plt.savefig('specificitycompare_weak_ts40.png')


    fig = plt.figure(figsize=(8, 10))
    plt.plot(feature_weight_list, selective_error, c='#2b8cbe')
    plt.plot(feature_weight_list, naive_error, c='#D7191C')
    plt.plot(feature_weight_list, ds_error, c='#31a354')
    plt.plot(feature_weight_list, single_selective_error, c='#feb24c')
    plt.plot([], c='#2b8cbe', label='Randomized Multi-Task Lasso')
    plt.plot([], c='#D7191C', label='Multi-Task Lasso')
    plt.plot([], c='#31a354', label='Data Splitting')
    plt.plot([], c='#feb24c', label='K Randomized Lassos')
    plt.legend()
    plt.tight_layout()
    plt.ylabel('Average MSE')
    plt.xlabel('Lambda Value')
    plt.title('Error Along Lambda Path')
    plt.savefig('errcompare_weak_ts40.png',bbox_inches='tight')



    #Plot distribution of pivots
    #pivots = pivot[0]
    #pivots_naive = pivot_naive[0]
    #plt.clf()
    #grid = np.linspace(0, 1, 101)
   # points = [np.max(np.searchsorted(np.sort(np.asarray(pivots)), i, side='right')) / np.float(np.shape(pivots)[0]) for
          #    i in np.linspace(0, 1, 101)]
   # points_naive = [np.max(np.searchsorted(np.sort(np.asarray(pivots_naive)), i, side='right')) / np.float(
     #   np.shape(pivots_naive)[0]) for i in np.linspace(0, 1, 101)]
    #fig = plt.figure(figsize=(32, 8))
    #fig.tight_layout()
    #fig.add_subplot(1, 4, 1)
    #plt.plot(grid, points, c='blue', marker='^')
    #plt.plot(grid, points_naive, c='red', marker='^')
    #plt.plot(grid, grid, 'k--')
   # plt.title('Task Sparsity 25%, SNR 0.2-0.5')

    #pivots = pivot[1]
    #pivots_naive = pivot_naive[1]
    #grid = np.linspace(0, 1, 101)
    #points = [np.searchsorted(np.sort(np.asarray(pivots)), i, side='right') / np.float(np.shape(pivots)[0]) for i in
        #      np.linspace(0, 1, 101)]
    #points_naive = [
     #   np.searchsorted(np.sort(np.asarray(pivots_naive)), i, side='right') / np.float(np.shape(pivots_naive)[0]) for i
     #   in np.linspace(0, 1, 101)]
    #fig.add_subplot(1, 4, 2)
    #plt.plot(grid, points, c='blue', marker='^')
    #plt.plot(grid, points_naive, c='red', marker='^')
    #plt.plot(grid, grid, 'k--')
    #plt.title('Task Sparsity 25%, SNR 0.5-1.0')

    #pivots = pivot[2]
    #pivots_naive = pivot_naive[2]
    #grid = np.linspace(0, 1, 101)
    #points = [np.searchsorted(np.sort(np.asarray(pivots)), i, side='right') / np.float(np.shape(pivots)[0]) for i in
     #         np.linspace(0, 1, 101)]
   # points_naive = [
     #   np.searchsorted(np.sort(np.asarray(pivots_naive)), i, side='right') / np.float(np.shape(pivots_naive)[0]) for i
     #   in np.linspace(0, 1, 101)]
    #fig.add_subplot(1, 4, 3)
    #plt.plot(grid, points, c='blue', marker='^')
    #plt.plot(grid, points_naive, c='red', marker='^')
    #plt.plot(grid, grid, 'k--')
   # plt.title('Task Sparsity 25%, SNR 1.0-3.0')

    #pivots = pivot[3]
    #pivots_naive = pivot_naive[3]
   # grid = np.linspace(0, 1, 101)
    #points = [np.searchsorted(np.sort(np.asarray(pivots)), i, side='right') / np.float(np.shape(pivots)[0]) for i in
    #          np.linspace(0, 1, 101)]
    #points_naive = [
    #    np.searchsorted(np.sort(np.asarray(pivots_naive)), i, side='right') / np.float(np.shape(pivots_naive)[0]) for i
    #    in np.linspace(0, 1, 101)]
    #fig.add_subplot(1, 4, 4)
    #plt.plot(grid, points, c='blue', marker='^')
    #plt.plot(grid, points_naive, c='red', marker='^')
    #plt.plot(grid, grid, 'k--')
   # plt.title('Task Sparsity 25%, SNR 3.0-5.0')

    #plt.savefig("25_p100.png")

if __name__ == "__main__":
    main()