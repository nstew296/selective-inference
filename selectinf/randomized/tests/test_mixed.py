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
                                randomizer_scale = 0.7):

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

            active_signs = multi_lasso.fit(perturbations=_initial_omega)

            dispersions = sigma ** 2

            estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(
                dispersions=dispersions)

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

            dispersions = sigma ** 2

            estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(
                dispersions=dispersions)

        except:
            active_signs= np.asarray([])

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
                                      split = 0.5,
                                      link = "identity"):


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
                                      randomizer_scale = 1.0,
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
                     randomizer_scale=randomizer_scale)

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
                                           randomizer_scale=0.7)

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

    ntask = 5
    nsamples= 1000 * np.ones(ntask)
    p=100
    global_sparsity=0.95
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

        coverage2, length2, pivot2, sns2, spc2, err2 = test_multitask_lasso_hetero(predictor_vars_train,
                                                                             response_vars_train,
                                                                             predictor_vars_test,
                                                                             response_vars_test,
                                                                             beta,
                                                                             gaussian_noise,
                                                                             sigma,
                                                                             link="identity",
                                                                             weight=weight,
                                                                             randomizer_scale=1.0)

        if coverage2 != []:
            cov2.append(np.mean(np.asarray(coverage2)))
            len2.extend(length2)
            pivots2.extend(pivot2)
        sensitivity_list2.append(sns2)
        specificity_list2.append(spc2)
        test_error_list2.append(err2)


        coverage_naive, length_naive, pivot_naive, naive_sensitivity, naive_specificity, naive_err = test_multitask_lasso_naive_hetero(predictor_vars_train,
                                                                             response_vars_train,
                                                                             predictor_vars_test,
                                                                             response_vars_test,
                                                                             beta,
                                                                             sigma,
                                                                             weight,
                                                                             link="identity")

        if coverage_naive != []:
            cov_naive.append(np.mean(np.asarray(coverage_naive)))
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
                                                                             split = 0.67,
                                                                             link="identity")

        if coverage_data_splitting!=[]:
            cov_data_splitting.append(np.mean(np.asarray(coverage_data_splitting)))
            len_data_splitting.extend(length_data_splitting)
            pivots_data_splitting.extend(pivot_data_splitting)
        sensitivity_list_ds.append(sns_ds)
        specificity_list_ds.append(spc_ds)
        data_splitting_test_error_list.append(ds_error)

        coverage_data_splitting2, length_data_splitting2, pivot_data_splitting2, sns_ds2, spc_ds2, ds_error2 = test_multitask_lasso_data_splitting(
                                                                            predictor_vars_train,
                                                                            response_vars_train,
                                                                            predictor_vars_test,
                                                                            response_vars_test,
                                                                            beta,
                                                                            sigma,
                                                                            weight,
                                                                            split=0.5,
                                                                            link="identity")

        if coverage_data_splitting2 != []:
            cov_data_splitting2.append(np.mean(np.asarray(coverage_data_splitting2)))
            len_data_splitting2.extend(length_data_splitting2)
            pivots_data_splitting2.extend(pivot_data_splitting2)
        sensitivity_list_ds2.append(sns_ds2)
        specificity_list_ds2.append(spc_ds2)
        data_splitting_test_error_list2.append(ds_error2)

        coverage_single_task_selective, length_single_task_selective, pivot_single_task_selective, sns_single_task, spc_single_task, err_single_selective = test_single_task_lasso_posi_hetero(predictor_vars_train,
                                      response_vars_train,
                                      predictor_vars_test,
                                      response_vars_test,
                                      beta,
                                      sigma,
                                      weight,
                                      randomizer_scale = 0.7,
                                      link="identity")


        if coverage_single_task_selective!=[]:
            cov_single_task_selective.append(np.mean(np.asarray(coverage_single_task_selective)))
            len_single_task_selective.extend(length_single_task_selective)
        sensitivity_list_single_task_selective.append(sns_single_task)
        specificity_list_single_task_selective.append(spc_single_task)
        single_task_selective_test_error_list.append(err_single_selective)

        coverage_single_task_selective2, length_single_task_selective2, pivot_single_task_selective2, sns_single_task2, spc_single_task2, err_single_selective2 = test_single_task_lasso_posi_hetero(
            predictor_vars_train,
            response_vars_train,
            predictor_vars_test,
            response_vars_test,
            beta,
            sigma,
            weight,
            randomizer_scale=1.0,
            link="identity")

        if coverage_single_task_selective2 != []:
            cov_single_task_selective2.append(np.mean(np.asarray(coverage_single_task_selective2)))
            len_single_task_selective2.extend(length_single_task_selective2)
        sensitivity_list_single_task_selective2.append(sns_single_task2)
        specificity_list_single_task_selective2.append(spc_single_task2)
        single_task_selective_test_error_list2.append(err_single_selective2)

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
        print("median sensitivity posi2", np.median(np.asarray(sensitivity_list2)))
        print("median specificity posi2", np.median(np.asarray(specificity_list2)))
        print("median sensitivity data splitting", np.median(np.asarray(sensitivity_list_ds)))
        print("median sensitivity data splitting2", np.median(np.asarray(sensitivity_list_ds2)))
        print("median specificity data splitting", np.median(np.asarray(specificity_list_ds)))
        print("median sensitivity single lasso", np.median(np.asarray(sensitivity_list_single_task_selective)))
        print("median specificity signle lasso", np.median(np.asarray(specificity_list_single_task_selective)))
        print("median sensitivity single lasso2", np.median(np.asarray(sensitivity_list_single_task_selective2)))

        print("error selective", np.median(np.asarray(test_error_list)))
        print("error naive", np.median(np.asarray(naive_test_error_list)))
        print("error ds", np.median(np.asarray(data_splitting_test_error_list)))
        print("error single task", np.median(np.asarray(single_task_selective_test_error_list)))

    return([pivots,pivots_naive,pivots_data_splitting,
            np.asarray(cov), np.asarray(cov2), np.asarray(cov_naive), np.asarray(cov_data_splitting), np.asarray(cov_data_splitting2), np.asarray(cov_single_task_selective), np.asarray(cov_single_task_selective2),
            np.asarray(len),np.asarray(len2),np.asarray(len_naive),np.asarray(len_data_splitting),np.asarray(len_data_splitting2),np.asarray(len_single_task_selective), np.asarray(len_single_task_selective2),
            np.asarray(sensitivity_list),np.asarray(sensitivity_list2),np.asarray(sensitivity_list_naive),np.asarray(sensitivity_list_ds),np.asarray(sensitivity_list_ds2),np.asarray(sensitivity_list_single_task_selective),
            np.asarray(sensitivity_list_single_task_selective2),np.asarray(specificity_list),np.asarray(specificity_list2),np.asarray(specificity_list_naive),np.asarray(specificity_list_ds),np.asarray(specificity_list_ds2),np.asarray(specificity_list_single_task_selective),
            np.asarray(specificity_list_single_task_selective2), np.mean(np.asarray(test_error_list)),np.mean(np.asarray(test_error_list2)),np.mean(np.asarray(naive_test_error_list)),
            np.mean(np.asarray(data_splitting_test_error_list)),np.mean(np.asarray(data_splitting_test_error_list2)),np.mean(np.asarray(single_task_selective_test_error_list)),np.mean(np.asarray(single_task_selective_test_error_list2))])

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

    lambdamin = 0.5
    lambdamax = 4.0
    #weights = np.arange(np.log(lambdamin), np.log(lambdamax), (np.log(lambdamax) - np.log(lambdamin)) / (length_path))
    #feature_weight_list = np.exp(weights)
    feature_weight_list = np.arange(lambdamin, lambdamax,(lambdamax - lambdamin) / (length_path))
    print(feature_weight_list)

    coverage = {i: [[], [], [], [], [], [], []] for i in range(length_path)}
    length = {i: [[], [], [], [], [], [], []] for i in range(length_path)}
    sensitivity = {i: [[], [], [], [], [], [], []] for i in range(length_path)}
    specificity = {i: [[], [], [], [], [], [], []] for i in range(length_path)}
    error = {i: [[], [], [], [], [], [], []] for i in range(length_path)}

    for i in range(len(feature_weight_list)):
        sims = test_coverage(feature_weight_list[i], [2.0, 5.0], nsim=100)
        coverage[i][0].extend(sims[3])
        coverage[i][1].extend(sims[4])
        coverage[i][2].extend(sims[5])
        coverage[i][3].extend(sims[6])
        coverage[i][4].extend(sims[7])
        coverage[i][5].extend(sims[8])
        coverage[i][6].extend(sims[9])
        length[i][0].extend(sims[10])
        length[i][1].extend(sims[11])
        length[i][2].extend(sims[12])
        length[i][3].extend(sims[13])
        length[i][4].extend(sims[14])
        length[i][5].extend(sims[15])
        length[i][6].extend(sims[16])
        sensitivity[i][0].extend(sims[17])
        sensitivity[i][1].extend(sims[18])
        sensitivity[i][2].extend(sims[19])
        sensitivity[i][3].extend(sims[20])
        sensitivity[i][4].extend(sims[21])
        sensitivity[i][5].extend(sims[22])
        sensitivity[i][6].extend(sims[23])
        specificity[i][0].extend(sims[24])
        specificity[i][1].extend(sims[25])
        specificity[i][2].extend(sims[26])
        specificity[i][3].extend(sims[27])
        specificity[i][4].extend(sims[28])
        specificity[i][5].extend(sims[29])
        specificity[i][6].extend(sims[30])
        error[i][0].append(sims[31])
        error[i][1].append(sims[32])
        error[i][2].append(sims[33])
        error[i][3].append(sims[34])
        error[i][4].append(sims[35])
        error[i][5].append(sims[36])
        error[i][6].append(sims[37])

    selective_lengths = [length[i][0] for i in range(length_path)]
    selective_lengths2 = [length[i][1] for i in range(length_path)]
    naive_lengths = [length[i][2] for i in range(length_path)]
    ds_lengths = [length[i][3] for i in range(length_path)]
    ds_lengths2 = [length[i][4] for i in range(length_path)]
    single_selective_lengths = [length[i][5] for i in range(length_path)]
    single_selective_lengths2 = [length[i][6] for i in range(length_path)]

    selective_coverage = [coverage[i][0] for i in range(length_path)]
    selective_coverage2 = [coverage[i][1] for i in range(length_path)]
    naive_coverage = [coverage[i][2] for i in range(length_path)]
    ds_coverage = [coverage[i][3] for i in range(length_path)]
    ds_coverage2 = [coverage[i][4] for i in range(length_path)]
    single_selective_coverage = [coverage[i][5] for i in range(length_path)]
    single_selective_coverage2 = [coverage[i][6] for i in range(length_path)]

    selective_error = [error[i][0] for i in range(length_path)]
    selective_error2 = [error[i][1] for i in range(length_path)]
    naive_error = [error[i][2] for i in range(length_path)]
    ds_error = [error[i][3] for i in range(length_path)]
    ds_error2 = [error[i][4] for i in range(length_path)]
    single_selective_error = [error[i][5] for i in range(length_path)]
    single_selective_error2 = [error[i][6] for i in range(length_path)]

    def set_box_color(bp, color, linestyle):
        plt.setp(bp['boxes'], color=color, linestyle=linestyle)
        plt.setp(bp['whiskers'], color=color, linestyle=linestyle)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    length = len(feature_weight_list)

    fig = plt.figure(figsize=(17, 14))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    plt.sca(ax1)
    first = plt.boxplot(selective_coverage,positions=np.array(xrange(length)) * 3, sym='', widths=0.3)
    second = plt.boxplot(selective_coverage2,positions=np.array(xrange(length)) * 3 + .3, sym='', widths=0.3)
    third = plt.boxplot(naive_coverage, positions=np.array(xrange(length)) * 3 + .6, sym='', widths=0.3)
    fourth = plt.boxplot(ds_coverage, positions=np.array(xrange(length)) * 3 + .9, sym='', widths=0.3)
    fifth = plt.boxplot(ds_coverage2, positions=np.array(xrange(length)) * 3 + 1.2, sym='', widths=0.3)
    sixth = plt.boxplot(single_selective_coverage, positions=np.array(xrange(length)) * 3 + 1.5, sym='', widths=0.3)
    seventh = plt.boxplot(single_selective_coverage2, positions=np.array(xrange(length)) * 3 + 1.8, sym='', widths=0.3)
    set_box_color(first, '#2b8cbe', 'solid')  # colors are from http://colorbrewer2.org/
    set_box_color(second, '#6baed6', '--')
    set_box_color(third, '#D7191C', 'solid')
    set_box_color(fourth, '#238443', 'solid')
    set_box_color(fifth, '#31a354', '--')
    set_box_color(sixth, '#fd8d3c', 'solid')
    set_box_color(seventh, '#feb24c', '--')
    plt.xticks(xrange(1, (length) * 3 + 1, 3), [round(num, 1) for num in feature_weight_list])
    plt.xlim(-1, (length - 1) * 3 + 3)
    plt.plot(np.argmin(selective_error) * 3, 1.01, 'ro', c='#2b8cbe')
    plt.plot(np.argmin(selective_error2) * 3 + .3, 1.01, 'ro', c='#6baed6')
    plt.plot(np.argmin(naive_error) * 3 + .6, 1.01, 'ro', c='#D7191C')
    plt.plot(np.argmin(ds_error) * 3 + .9, 1.01, 'ro', c='#238443')
    plt.plot(np.argmin(ds_error2) * 3 + 1.2, 1.01, 'ro', c='#31a354')
    plt.plot(np.argmin(single_selective_error) * 3 + 1.5, 1.01, 'ro', c='#fd8d3c')
    plt.plot(np.argmin(single_selective_error2) * 3 + 1.8, 1.01, 'ro', c='#feb24c')
    plt.tight_layout()
    plt.ylabel('Mean Coverage per Simulation', fontsize=12)

    plt.sca(ax2)
    first = plt.boxplot(selective_lengths,positions=np.array(xrange(length)) * 3, sym='', widths=0.3)
    second = plt.boxplot(selective_lengths2,positions=np.array(xrange(length)) * 3 + .3, sym='', widths=0.3)
    third = plt.boxplot(naive_lengths,positions=np.array(xrange(length)) * 3 + .6, sym='', widths=0.3)
    fourth = plt.boxplot(ds_lengths, positions=np.array(xrange(length)) * 3 + .9, sym='', widths=0.3)
    fifth = plt.boxplot(ds_lengths2,positions=np.array(xrange(length)) * 3 + 1.2, sym='', widths=0.3)
    sixth = plt.boxplot(single_selective_lengths,positions=np.array(xrange(length)) * 3 + 1.5, sym='', widths=0.3)
    seventh = plt.boxplot(single_selective_lengths2, positions=np.array(xrange(length)) * 3 + 1.8, sym='', widths=0.3)
    set_box_color(first, '#2b8cbe', 'solid')  # colors are from http://colorbrewer2.org/
    set_box_color(second, '#6baed6', '--')
    set_box_color(third, '#D7191C', 'solid')
    set_box_color(fourth, '#238443', 'solid')
    set_box_color(fifth, '#31a354', '--')
    set_box_color(sixth, '#fd8d3c', 'solid')
    set_box_color(seventh, '#feb24c', '--')
    plt.xticks(xrange(1, (length) * 3 + 1, 3), [round(num, 1) for num in feature_weight_list])
    plt.xlim(-1, (length - 1) * 3 + 3)
    max_length = np.max(np.concatenate(
        (np.concatenate(selective_lengths), np.concatenate(selective_lengths2), np.concatenate(naive_lengths),
         np.concatenate(ds_lengths), np.concatenate(ds_lengths2), np.concatenate(single_selective_lengths),
         np.concatenate(single_selective_lengths2))))
    plt.plot(np.argmin(selective_error) * 3, max_length, 'ro', c='#2b8cbe')
    plt.plot(np.argmin(selective_error2) * 3 + .3, max_length, 'ro', c='#6baed6')
    plt.plot(np.argmin(naive_error) * 3 + .6, max_length, 'ro', c='#D7191C')
    plt.plot(np.argmin(ds_error) * 3 + .9, max_length, 'ro', c='#238443')
    plt.plot(np.argmin(ds_error2) * 3 + 1.2, max_length, 'ro', c='#31a354')
    plt.plot(np.argmin(single_selective_error) * 3 + 1.5, max_length, 'ro', c='#fd8d3c')
    plt.plot(np.argmin(single_selective_error2) * 3 + 1.8, max_length, 'ro', c='#feb24c')
    plt.tight_layout()
    plt.plot([], c='#D7191C', label='Multi-Task Lasso', linewidth=2.5)
    plt.plot([], c='#fd8d3c', label='K Randomized Lassos 0.7', linewidth=2.5)
    plt.plot([], c='#feb24c', label='K Randomized Lassos 1.0', linestyle='--', linewidth=2.5)
    plt.plot([], c='#238443', label='Data Splitting 67/33', linewidth=2.5)
    plt.plot([], c='#31a354', label='Data Splitting 50/50', linestyle='--', linewidth=2.5)
    plt.plot([], c='#2b8cbe', label='Randomized Multi-Task Lasso 0.7', linewidth=2.5)
    plt.plot([], c='#6baed6', label='Randomized Multi-Task Lasso 1.0', linestyle='--', linewidth=2.5)
    plt.legend()
    plt.ylabel('Interval Length', fontsize=12)

    ax1.set_title("Coverage", y=1.01)
    ax2.set_title("Length", y=1.01)

    ax2.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.6), fontsize=14)

    def common_format(ax):
        ax.grid(True, which='both', color='#f0f0f0')
        ax.set_xlabel('Lambda Value', fontsize=12)
        return ax

    common_format(ax1)
    common_format(ax2)

    ax1.axhline(y=0.9, color='k', linestyle='--', linewidth=2)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('cov_len_by_lambda.png', bbox_inches='tight')


if __name__ == "__main__":
    main()