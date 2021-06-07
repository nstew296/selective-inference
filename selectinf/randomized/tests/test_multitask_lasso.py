import numpy as np
#from Tkinter import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.stats import norm as ndist
from scipy.stats import t as tdist

from selectinf.randomized.multitask_lasso import multi_task_lasso
from selectinf.tests.instance import gaussian_multitask_instance
from selectinf.tests.instance import logistic_multitask_instance
from selectinf.tests.instance import poisson_multitask_instance
from selectinf.randomized.lasso import lasso, selected_targets


def cross_validate_posi_hetero(ntask=2,
                   nsamples=500 * np.ones(2),
                   p=100,
                   global_sparsity=.8,
                   task_sparsity=.3,
                   sigma=1. * np.ones(2),
                   signal_fac=np.array([1., 5.]),
                   rhos=0. * np.ones(2),
                   link = "identity",
                   randomizer_scale = .5):

    nsamples = nsamples.astype(int)

    signal = np.sqrt(signal_fac * 2 * np.log(p))

    if link=="identity":

        response_vars, predictor_vars, beta, _gaussian_noise = gaussian_multitask_instance(ntask,
                                                                                       nsamples,
                                                                                       p,
                                                                                       global_sparsity,
                                                                                       task_sparsity,
                                                                                       sigma,
                                                                                       signal,
                                                                                       rhos,
                                                                                       random_signs=True,
                                                                                       equicorrelated=False)[:4]

    if link == "logit":
        response_vars, predictor_vars, beta, _gaussian_noise = logistic_multitask_instance(ntask,
                                                                                           nsamples,
                                                                                           p,
                                                                                           global_sparsity,
                                                                                           task_sparsity,
                                                                                           sigma,
                                                                                           signal,
                                                                                           rhos,
                                                                                           random_signs=True,
                                                                                           equicorrelated=False)[:4]

    if link == "log":
        response_vars, predictor_vars, beta, _gaussian_noise = poisson_multitask_instance(ntask,
                                                                                          nsamples,
                                                                                          p,
                                                                                          global_sparsity,
                                                                                          task_sparsity,
                                                                                          sigma,
                                                                                          signal,
                                                                                          rhos,
                                                                                          random_signs=True,
                                                                                          equicorrelated=False)[:4]

    folds = {i: [] for i in range(5)}
    holdout = np.round(nsamples[0]/2.0)
    samples = np.arange(np.int(holdout))

    for i in range(5):
        folds[i] = np.random.choice(samples, size=np.int(np.round(.2 * holdout)), replace=False)
        samples = np.setdiff1d(samples, folds[i])

    lambdamin = 1.5
    lambdamax = 4.0
    weights = np.arange(np.log(lambdamin), np.log(lambdamax), (np.log(lambdamax) - np.log(lambdamin)) / 10)
    weights = np.exp(weights)

    errors = np.zeros(len(weights))

    for i in range(5):

        train = np.setdiff1d(np.arange(np.int(holdout)), folds[i])
        test = folds[i]

        response_vars_train = {j: response_vars[j][train] for j in range(ntask)}
        predictor_vars_train = {j: predictor_vars[j][train] for j in range(ntask)}

        response_vars_test = {j: response_vars[j][test] for j in range(ntask)}
        predictor_vars_test = {j: predictor_vars[j][test] for j in range(ntask)}

        for w in range(len(weights)):

            feature_weight = weights[w] * np.ones(p)
            sigmas_ = sigma
            randomizer_scales = randomizer_scale * sigmas_

            _initial_omega = np.array(
                [randomizer_scales[j] * _gaussian_noise[(j * p):((j + 1) * p)] for j in range(ntask)]).T


            try:
                if link=="identity":
                    multi_lasso = multi_task_lasso.gaussian(predictor_vars_train,
                                                    response_vars_train,
                                                    feature_weight,
                                                    ridge_term=None,
                                                    randomizer_scales=randomizer_scales)

                if link == "logit":
                    multi_lasso = multi_task_lasso.logistic(predictor_vars_train,
                                                            response_vars_train,
                                                            feature_weight,
                                                            ridge_term=None,
                                                            randomizer_scales=randomizer_scales)

                if link == "log":
                    multi_lasso = multi_task_lasso.poisson(predictor_vars_train,
                                                            response_vars_train,
                                                            feature_weight,
                                                            ridge_term=None,
                                                            randomizer_scales=randomizer_scales)

                multi_lasso.fit(perturbations=_initial_omega)

                active_signs = multi_lasso.fit(perturbations=_initial_omega)

                dispersions = sigma ** 2

                estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(
                dispersions=dispersions)

            except:
                sum = 0
                for j in range(ntask):
                    sum += 0.5*np.linalg.norm(response_vars_test[j], 2)**2
                errors[w] += sum
                continue


            error = 0

            idx = 0

            for j in range(ntask):

                idx_new = np.sum(active_signs[:, j] != 0)
                if idx_new == 0:
                    error += 0.5*np.sum(np.square(response_vars_test[j]))
                    continue
                error += 0.5 * np.sum(np.square((response_vars_test[j] - (predictor_vars_test[j])[:, (active_signs[:, j] != 0)].dot(
                    estimate[idx:idx + idx_new]))))
                idx = idx + idx_new

            errors[w] += error

    print("errors",errors)
    idx_min_error = np.int(np.argmin(errors))
    lam_min = weights[idx_min_error]
    print(lam_min,"tuning param")

    unused_predictor = {}
    for j in range(ntask):
        unused_predictor[j] = predictor_vars[j][np.int(holdout)+1:]

    return (lam_min,unused_predictor,beta)


def cross_validate_naive_hetero(ntask=2,
                   nsamples=500 * np.ones(2),
                   p=100,
                   global_sparsity=.8,
                   task_sparsity=.3,
                   sigma=1. * np.ones(2),
                   signal_fac=np.array([1., 5.]),
                   rhos=0. * np.ones(2),
                   link="identity"):

    nsamples = nsamples.astype(int)

    signal = np.sqrt(signal_fac * 2 * np.log(p))

    if link == "identity":
        response_vars, predictor_vars, beta, _gaussian_noise = gaussian_multitask_instance(ntask,
                                                                                           nsamples,
                                                                                           p,
                                                                                           global_sparsity,
                                                                                           task_sparsity,
                                                                                           sigma,
                                                                                           signal,
                                                                                           rhos,
                                                                                           random_signs=True,
                                                                                           equicorrelated=False)[:4]

    if link == "logit":
        response_vars, predictor_vars, beta, _gaussian_noise = logistic_multitask_instance(ntask,
                                                                                           nsamples,
                                                                                           p,
                                                                                           global_sparsity,
                                                                                           task_sparsity,
                                                                                           sigma,
                                                                                           signal,
                                                                                           rhos,
                                                                                           random_signs=True,
                                                                                           equicorrelated=False)[:4]

    if link == "log":
        response_vars, predictor_vars, beta, _gaussian_noise = poisson_multitask_instance(ntask,
                                                                                          nsamples,
                                                                                          p,
                                                                                          global_sparsity,
                                                                                          task_sparsity,
                                                                                          sigma,
                                                                                          signal,
                                                                                          rhos,
                                                                                          random_signs=True,
                                                                                          equicorrelated=False)[:4]

    folds = {i: [] for i in range(5)}
    holdout = np.round(nsamples[0]/2.0)
    samples = np.arange(np.int(holdout))

    for i in range(5):
        folds[i] = np.random.choice(samples, size=np.int(np.round(.2 * holdout)), replace=False)
        samples = np.setdiff1d(samples, folds[i])

    lambdamin = 1.0
    lambdamax = 4.0
    weights = np.arange(np.log(lambdamin), np.log(lambdamax), (np.log(lambdamax) - np.log(lambdamin)) /10)
    weights = np.exp(weights)

    errors = np.zeros(len(weights))

    for i in range(5):

        train = np.setdiff1d(np.arange(np.int(holdout)), folds[i])
        test = folds[i]

        response_vars_train = {j: response_vars[j][train] for j in range(ntask)}
        predictor_vars_train = {j: predictor_vars[j][train] for j in range(ntask)}

        response_vars_test = {j: response_vars[j][test] for j in range(ntask)}
        predictor_vars_test = {j: predictor_vars[j][test] for j in range(ntask)}

        for w in range(len(weights)):

            feature_weight = weights[w] * np.ones(p)

            sigmas_ = sigma

            perturbations = np.zeros((p, ntask))

            try:
                if link == "identity":
                    multi_lasso = multi_task_lasso.gaussian(predictor_vars_train,
                                                            response_vars_train,
                                                            feature_weight,
                                                            ridge_term=None,
                                                            randomizer_scales=1.*sigmas_,
                                                            perturbations=perturbations)

                if link == "logit":
                    multi_lasso = multi_task_lasso.logistic(predictor_vars_train,
                                                            response_vars_train,
                                                            feature_weight,
                                                            ridge_term=None,
                                                            randomizer_scales=1. * sigmas_,
                                                            perturbations=perturbations)

                if link == "log":
                    multi_lasso = multi_task_lasso.poisson(predictor_vars_train,
                                                           response_vars_train,
                                                           feature_weight,
                                                           ridge_term=None,
                                                           randomizer_scales=1. * sigmas_,
                                                           perturbations=perturbations)
                active_signs = multi_lasso.fit()


            except:
                sum = 0
                for j in range(ntask):
                    sum += np.linalg.norm(response_vars_test[j], 2)**2
                errors[w] += sum
                continue

            error = 0

            for j in range(ntask):

                idx_new = np.sum(active_signs[:, j] != 0)
                if idx_new == 0:
                    error += 0.5 * np.sum(np.square(response_vars_test[j]))
                    continue
                X, y = multi_lasso.loglikes[j].data
                observed_target = np.linalg.pinv(X[:, (active_signs[:, j] != 0)]).dot(y)
                error += 0.5 * np.sum(np.square(response_vars_test[j] - (predictor_vars_test[j])[:, (active_signs[:, j] != 0)].dot(observed_target)))

            errors[w] += error

    print("errors", errors)
    idx_min_error = np.int(np.argmin(errors))
    lam_min = weights[idx_min_error]
    print(lam_min, "tuning param")


    unused_predictor = {}
    for j in range(ntask):
        unused_predictor[j] = predictor_vars[j][np.int(holdout)+1:]

    return (lam_min,unused_predictor,beta)


def test_multitask_lasso_hetero(predictor_vars,
                                beta,
                                sigma=1. * np.ones(2),
                                weight=2.,
                                link = "identity",
                                randomizer_scale=0.5):

    ntask = len(predictor_vars.keys())
    nsamples = np.asarray([np.shape(predictor_vars[i])[0] for i in range(ntask)])
    p = np.shape(beta)[0]

    def _noise(n, df=np.inf):
        if df == np.inf:
            return np.random.standard_normal(n)
        else:
            sd_t = np.std(tdist.rvs(df, size=50000))
        return tdist.rvs(df, size=n) / sd_t

    if link=="identity":
        gaussian_noise = _noise(nsamples.sum() + p*ntask, np.inf)
        response_vars = {}
        nsamples_cumsum = np.cumsum(nsamples)
        for i in range(ntask):
            if i == 0:
                response_vars[i] = (predictor_vars[i].dot(beta[:, i]) + gaussian_noise[:nsamples_cumsum[i]]) * sigma[i]
            else:
                response_vars[i] = (predictor_vars[i].dot(beta[:, i]) + gaussian_noise[nsamples_cumsum[i-1]:nsamples_cumsum[i]]) * sigma[i]

    if link=="logit":
        gaussian_noise = _noise(p * ntask, np.inf)
        response_vars = {}
        pis = {}
        for i in range(ntask):
            pis[i] = predictor_vars[i].dot(beta[:, i]) * sigma[i]
            response_vars[i] = np.random.binomial(1, np.exp(pis[i]) / (1.0 + np.exp(pis[i])))

    if link=="log":
        gaussian_noise = _noise(p * ntask, np.inf)
        response_vars = {}
        pis = {}
        for i in range(ntask):
            pis[i] = predictor_vars[i].dot(beta[:, i]) * sigma[i]
            response_vars[i] = np.random.poisson(np.exp(pis[i]))


    samples = np.arange(np.int(nsamples[0]))
    train = np.random.choice(samples, size=np.int(nsamples[0]), replace=False)
    test = np.setdiff1d(samples, train)

    response_vars_train = {j: response_vars[j][train] for j in range(ntask)}
    predictor_vars_train = {j: predictor_vars[j][train] for j in range(ntask)}

    response_vars_test = {j: response_vars[j][test] for j in range(ntask)}
    predictor_vars_test = {j: predictor_vars[j][test] for j in range(ntask)}

    while True:

        feature_weight = weight * np.ones(p)

        sigmas_ = sigma
        randomizer_scales = randomizer_scale * np.array([sigmas_[i] for i in range(ntask)])

        _initial_omega = np.array(
            [randomizer_scales[i] * gaussian_noise[(i * p):((i + 1) * p)] for i in range(ntask)]).T

        if link == "identity":
            multi_lasso = multi_task_lasso.gaussian(predictor_vars_train,
                                                    response_vars_train,
                                                    feature_weight,
                                                    ridge_term=None,
                                                    randomizer_scales=randomizer_scales)

        if link == "logit":
            multi_lasso = multi_task_lasso.logistic(predictor_vars_train,
                                                    response_vars_train,
                                                    feature_weight,
                                                    ridge_term=None,
                                                    randomizer_scales=randomizer_scales)

        if link == "log":
            multi_lasso = multi_task_lasso.poisson(predictor_vars_train,
                                                   response_vars_train,
                                                   feature_weight,
                                                   ridge_term=None,
                                                   randomizer_scales=randomizer_scales)

        active_signs = multi_lasso.fit(perturbations=_initial_omega)

        #Compute snesitivity and specificity
        #true_active = np.transpose(np.nonzero(beta))
        #selected_active = np.transpose(np.nonzero(active_signs))
        #num_true_positive = np.sum(x in true_active.tolist() for x in selected_active.tolist())
        #num_false_positive = np.sum(x not in true_active.tolist() for x in selected_active.tolist())
        #num_positive = np.shape(true_active)[0]
        #num_negative = np.shape(beta)[0]*np.shape(beta)[1]-num_positive
        #sensitivity = np.float(num_true_positive)/np.float(num_positive)
        #specificity = 1.0 - np.float(num_false_positive)/np.float(num_negative)

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
            if np.shape(test)[0]==0:
                pass
            else:
                for j in range(ntask):
                    idx_new = np.sum(active_signs[:, j] != 0)
                    if idx_new == 0:
                        error += 0.5 * np.sum(np.square(response_vars_test[j]))
                        continue
                    error += 0.5 * np.sum(
                        np.square((response_vars_test[j] - (predictor_vars_test[j])[:, (active_signs[:, j] != 0)].dot(
                            estimate[idx:idx + idx_new]))))
                    idx = idx + idx_new

            # Compute snesitivity and specificity
            true_active = np.transpose(np.nonzero(beta))
            selected_active = np.transpose(np.nonzero(active_signs))
            true_positive_selected = [x in true_active.tolist() for x in selected_active.tolist()]
            num_true_positive_inference = np.sum([true_positive_selected[i]*(intervals[i,1]<0 or intervals[i,0]>0) for i in range(len(true_positive_selected))])
            num_positive = np.shape(true_active)[0]
            num_false_positive_inference = np.sum([(intervals[i,1]<0 or intervals[i,0]>0) for i in range(len(true_positive_selected))]) - num_true_positive_inference
            num_negative = np.shape(beta)[0]*np.shape(beta)[1]-num_positive
            sensitivity_inference = np.float(num_true_positive_inference)/np.float(num_positive)
            specificity_inference = 1.0 - np.float(num_false_positive_inference)/np.float(num_negative)

            return coverage, intervals[:, 1] - intervals[:, 0], pivot, sensitivity_inference, specificity_inference, error


def test_multitask_lasso_naive_hetero(predictor_vars,
                                      beta,
                                      sigma=1. * np.ones(2),
                                      weight=2.,
                                      link = "identity"):

    ntask = len(predictor_vars.keys())
    nsamples = np.asarray([np.shape(predictor_vars[i])[0] for i in range(ntask)])
    p = np.shape(beta)[0]

    def _noise(n, df=np.inf):
        if df == np.inf:
            return np.random.standard_normal(n)
        else:
            sd_t = np.std(tdist.rvs(df, size=50000))
        return tdist.rvs(df, size=n) / sd_t

    if link == "identity":
        gaussian_noise = _noise(nsamples.sum(), np.inf)
        response_vars = {}
        nsamples_cumsum = np.cumsum(nsamples)
        for i in range(ntask):
            if i == 0:
                response_vars[i] = (predictor_vars[i].dot(beta[:, i]) + gaussian_noise[:nsamples_cumsum[i]]) * sigma[i]
            else:
                response_vars[i] = (predictor_vars[i].dot(beta[:, i]) + gaussian_noise[nsamples_cumsum[i - 1]:nsamples_cumsum[i]]) * sigma[i]

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

    samples = np.arange(np.int(nsamples[0]))
    train = np.random.choice(samples, size=np.int(nsamples[0]), replace=False)
    test = np.setdiff1d(samples, train)

    response_vars_train = {j: response_vars[j][train] for j in range(ntask)}
    predictor_vars_train = {j: predictor_vars[j][train] for j in range(ntask)}

    response_vars_test = {j: response_vars[j][test] for j in range(ntask)}
    predictor_vars_test = {j: predictor_vars[j][test] for j in range(ntask)}

    while True:

        feature_weight = weight * np.ones(p)

        sigmas_ = sigma

        perturbations = np.zeros((p, ntask))

        if link == "identity":
            multi_lasso = multi_task_lasso.gaussian(predictor_vars_train,
                                                    response_vars_train,
                                                    feature_weight,
                                                    ridge_term=None,
                                                    randomizer_scales=1. * sigmas_,
                                                    perturbations=perturbations)

        if link == "logit":
            multi_lasso = multi_task_lasso.logistic(predictor_vars_train,
                                                    response_vars_train,
                                                    feature_weight,
                                                    ridge_term=None,
                                                    randomizer_scales=1. * sigmas_,
                                                    perturbations=perturbations)

        if link == "log":
            multi_lasso = multi_task_lasso.poisson(predictor_vars_train,
                                                   response_vars_train,
                                                   feature_weight,
                                                   ridge_term=None,
                                                   randomizer_scales=1. * sigmas_,
                                                   perturbations=perturbations)
        active_signs = multi_lasso.fit()

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

                if np.shape(test)[0]==0:
                    continue
                else:
                    idx_new = np.sum(active_signs[:, i] != 0)
                    if idx_new == 0:
                        error += 0.5 * np.sum(np.square(response_vars_test[i]))
                        continue
                    observed_target = np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(y)
                    error += 0.5 * np.sum(np.square(
                        response_vars_test[i] - (predictor_vars_test[i])[:, (active_signs[:, i] != 0)].dot(
                            observed_target)))

        return np.asarray(coverage), CIs[1:, 1] - CIs[1:, 0], np.asarray(pivot), error



def test_multitask_lasso_data_splitting(predictor_vars,
                                      beta,
                                      sigma=1. * np.ones(2),
                                      weight=2.,
                                      link = "identity"):

    ntask = len(predictor_vars.keys())
    nsamples = np.asarray([np.shape(predictor_vars[i])[0] for i in range(ntask)])
    p = np.shape(beta)[0]


    def _noise(n, df=np.inf):
        if df == np.inf:
            return np.random.standard_normal(n)
        else:
            sd_t = np.std(tdist.rvs(df, size=50000))
        return tdist.rvs(df, size=n) / sd_t

    if link == "identity":
        gaussian_noise = _noise(nsamples.sum(), np.inf)
        response_vars = {}
        nsamples_cumsum = np.cumsum(nsamples)
        for i in range(ntask):
            if i == 0:
                response_vars[i] = (predictor_vars[i].dot(beta[:, i]) + gaussian_noise[:nsamples_cumsum[i]]) * sigma[i]
            else:
                response_vars[i] = (predictor_vars[i].dot(beta[:, i]) + gaussian_noise[nsamples_cumsum[i - 1]:nsamples_cumsum[i]]) * sigma[i]

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

    samples = np.arange(np.int(nsamples[0]))
    train = np.random.choice(samples, size=np.int(0.5 * nsamples[0]), replace=False)
    test = np.setdiff1d(samples, train)

    response_vars_train = {j: response_vars[j][train] for j in range(ntask)}
    predictor_vars_train = {j: predictor_vars[j][train] for j in range(ntask)}

    response_vars_test = {j: response_vars[j][test] for j in range(ntask)}
    predictor_vars_test = {j: predictor_vars[j][test] for j in range(ntask)}


    while True:

        feature_weight = weight * np.ones(p)

        sigmas_ = sigma

        perturbations = np.zeros((p, ntask))

        if link == "identity":
            multi_lasso = multi_task_lasso.gaussian(predictor_vars_train,
                                                    response_vars_train,
                                                    feature_weight,
                                                    ridge_term=None,
                                                    randomizer_scales=1. * sigmas_,
                                                    perturbations=perturbations)

        if link == "logit":
            multi_lasso = multi_task_lasso.logistic(predictor_vars_train,
                                                    response_vars_train,
                                                    feature_weight,
                                                    ridge_term=None,
                                                    randomizer_scales=1. * sigmas_,
                                                    perturbations=perturbations)

        if link == "log":
            multi_lasso = multi_task_lasso.poisson(predictor_vars_train,
                                                   response_vars_train,
                                                   feature_weight,
                                                   ridge_term=None,
                                                   randomizer_scales=1. * sigmas_,
                                                   perturbations=perturbations)
        active_signs = multi_lasso.fit()

        # Compute snesitivity and specificity
        #true_active = np.transpose(np.nonzero(beta))
        #selected_active = np.transpose(np.nonzero(active_signs))
        #num_true_positive = np.sum(x in true_active.tolist() for x in selected_active.tolist())
        #num_false_positive = np.sum(x not in true_active.tolist() for x in selected_active.tolist())
        #num_positive = np.shape(true_active)[0]
        #num_negative = np.shape(beta)[0] * np.shape(beta)[1] - num_positive
        #sensitivity = np.float(num_true_positive) / np.float(num_positive)
        #specificity = 1.0 - np.float(num_false_positive) / np.float(num_negative)


        dispersions = sigma ** 2

        coverage = []
        pivot = []
        CIs = [[0,0]]

        if (active_signs != 0).sum() > 0:

            for i in range(ntask):
                X = predictor_vars_test[i]
                y = response_vars_test[i]
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

            # Compute snesitivity and specificity
            true_active = np.transpose(np.nonzero(beta))
            selected_active = np.transpose(np.nonzero(active_signs))
            true_positive_selected = [x in true_active.tolist() for x in selected_active.tolist()]
            num_true_positive_inference = np.sum(
                [true_positive_selected[i] * (CIs[i+1, 1] < 0 or CIs[i+1, 0] > 0) for i in
                 range(len(true_positive_selected))])
            num_positive = np.shape(true_active)[0]
            num_false_positive_inference = np.sum([(CIs[i+1, 1] < 0 or CIs[i+1, 0] > 0) for i in range(
                len(true_positive_selected))]) - num_true_positive_inference
            num_negative = np.shape(beta)[0] * np.shape(beta)[1] - num_positive
            sensitivity_inference = np.float(num_true_positive_inference) / np.float(num_positive)
            specificity_inference = 1.0 - np.float(num_false_positive_inference) / np.float(num_negative)


        return np.asarray(coverage), CIs[1:, 1] - CIs[1:, 0], np.asarray(pivot), sensitivity_inference, specificity_inference



def test_coverage(signal,nsim=100):
    cov = []
    len = []
    pivots = []
    penalties = []
    sensitivity_list = []
    specificity_list = []
    posi_prediction_error_list = []

    cov_naive = []
    len_naive = []
    pivots_naive = []
    penalties_naive = []
    naive_prediction_error_list = []

    cov_data_splitting = []
    len_data_splitting = []
    sensitivity_list_ds = []
    specificity_list_ds = []

    ntask = 5

    penalty_hetero, predictor, coef = cross_validate_posi_hetero(ntask=ntask,
                                                                 nsamples=2000 * np.ones(ntask),
                                                                 p=100,
                                                                 global_sparsity=0.9,
                                                                 task_sparsity=.25,
                                                                 sigma=1. * np.ones(ntask),
                                                                 signal_fac=np.array(signal),
                                                                 rhos=.7 * np.ones(ntask),
                                                                 link="identity",
                                                                 randomizer_scale=1)

    penalty_hetero_naive, predictor_naive, coef_naive = cross_validate_naive_hetero(ntask=ntask,
                                                                                    nsamples=2000 * np.ones(ntask),
                                                                                    p=100,
                                                                                    global_sparsity=0.9,
                                                                                    task_sparsity=.25,
                                                                                    sigma=1. * np.ones(ntask),
                                                                                    signal_fac=np.array(signal),
                                                                                    rhos=.7 * np.ones(ntask),
                                                                                    link="identity")

    penalties.append(penalty_hetero)
    penalties_naive.append(penalty_hetero_naive)

    for n in range(nsim):

        print(n,"n sim")

        try:

            coverage, length, pivot, sns, spc, err = test_multitask_lasso_hetero(predictor,
                                                                  coef,
                                                                  sigma=1. * np.ones(ntask),
                                                                  weight=np.float(penalty_hetero),
                                                                  link = "identity",
                                                                  randomizer_scale = 1)
            cov.append(np.mean(np.asarray(coverage)))
            len.extend(length)
            pivots.extend(pivot)
            sensitivity_list.append(sns)
            specificity_list.append(spc)
            posi_prediction_error_list.append(err)

        except:
            print("no selection posi")


        try:

             coverage_naive, length_naive, pivot_naive, naive_err = test_multitask_lasso_naive_hetero(predictor_naive,
                                                                        coef_naive,
                                                                        sigma=1. * np.ones(ntask),
                                                                        link = "identity",
                                                                        weight=np.float(penalty_hetero_naive))


             cov_naive.append(np.mean(np.asarray((coverage_naive))))
             len_naive.extend(length_naive)
             pivots_naive.extend(pivot_naive)
             naive_prediction_error_list.append(naive_err)


        except:
            print("no selection naive")

        try:

             coverage_data_splitting, length_data_splitting, pivot_data_splitting, sns_ds, spc_ds = test_multitask_lasso_data_splitting(predictor_naive,
                                                                        coef_naive,
                                                                        sigma=1. * np.ones(ntask),
                                                                        link = "identity",
                                                                        weight=np.float(penalty_hetero_naive))

             cov_data_splitting.append(np.mean(np.asarray(coverage_data_splitting)))
             len_data_splitting.extend(length_data_splitting)
             sensitivity_list_ds.append(sns_ds)
             specificity_list_ds.append(spc_ds)

        except:
            print("no selection data splitting")


        print("iteration completed ", n)
        print("posi coverage so far ", np.mean(np.asarray(cov)))
        print("naive coverage so far ", np.mean(np.asarray(cov_naive)))
        print("data splitting coverage so far ", np.mean(np.asarray(cov_data_splitting)))

        print("posi length so far ", np.mean(np.asarray(len)))
        print("naive length so far ", np.mean(np.asarray(len_naive)))
        print("data splitting length so far ", np.mean(np.asarray(len_data_splitting)))

        print("median sensitivity posi", np.median(np.asarray(sensitivity_list)))
        print("median specificity posi", np.median(np.asarray(specificity_list)))
        #print("mean prediction error posi", np.mean(np.asarray(posi_prediction_error_list)))
        print("median sensitivity data splitting", np.median(np.asarray(sensitivity_list_ds)))
        print("median specificity data splitting", np.median(np.asarray(specificity_list_ds)))

    return([pivots,pivots_naive,
            [np.mean(np.asarray(penalties)),np.mean(np.asarray(penalties_naive))],
            np.median(np.asarray(sensitivity_list)),np.median(np.asarray(specificity_list)),
            np.mean(np.asarray(posi_prediction_error_list)),
            np.asarray(len),np.asarray(len_naive),np.asarray(len_data_splitting),
            np.asarray(cov),np.asarray(cov_naive),np.asarray(cov_data_splitting)])

def main():

    signals = [[0.2,0.5],[0.5,1.0],[1.0,3.0],[3.0,5.0]]
    tuning = {0: [], 1: [], 2: [], 3: []}
    pivot = {0:[],1:[],2:[],3:[]}
    pivot_naive = {0:[], 1:[],2:[],3:[]}

    KL_divergence = {0: [[],[]], 1: [[],[]], 2: [[],[]], 3: [[],[]]}
    length = {0: [[], [], []], 1: [[], [], []], 2: [[], [], []], 3: [[], [], []]}
    coverage = {0: [[], [], []], 1: [[], [], []], 2: [[], [], []], 3: [[], [], []]}

    sensitivity = {0: [], 1: [], 2: [], 3: []}
    specificity = {0: [], 1: [], 2: [], 3: []}
    prediction_error_posi = {0: [], 1: [], 2: [], 3: []}

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

    #Coverage, length, and pivot plots
    for i in range(len(signals)):
        sims = test_coverage(signals[i],50)
        pivot[i] = sims[0]
        pivot_naive[i] = sims[1]
        tuning[i] = sims[2]
        sensitivity[i] = sims[3]
        specificity[i] = sims[4]
        prediction_error_posi[i] = sims[5]
        length[i][0].extend(sims[6])
        length[i][1].extend(sims[7])
        length[i][2].extend(sims[8])
        coverage[i][0].extend(sims[9])
        coverage[i][1].extend(sims[10])
        coverage[i][2].extend(sims[11])

    # Boxplot of length
    fig = plt.figure(figsize=(32, 8))
    fig.add_subplot(1, 4, 1)
    plt.boxplot(length[0], positions=[1, 2, 3], widths=0.4)
    plt.xticks([1, 2, 3], ['POSI', 'Naive', 'Data Splitting'])
    plt.ylabel('Length')
    plt.title('SNR 0.2-0.5')
    fig.add_subplot(1, 4, 2)
    plt.boxplot(length[1], positions=[1, 2, 3], widths=0.4)
    plt.xticks([1, 2, 3], ['POSI', 'Naive', 'Data Splitting'])
    plt.ylabel('Length')
    plt.title('SNR 0.5-1.0')
    fig.add_subplot(1, 4, 3)
    plt.boxplot(length[2], positions=[1, 2, 3], widths=0.4)
    plt.xticks([1, 2, 3], ['POSI', 'Naive', 'Data Splitting'])
    plt.ylabel('Length')
    plt.title('SNR 1.0-3.0')
    fig.add_subplot(1, 4, 4)
    plt.boxplot(length[3], positions=[1, 2, 3], widths=0.4)
    plt.xticks([1, 2, 3], ['POSI', 'Naive', 'Data Splitting'])
    plt.ylabel('Length')
    plt.title('SNR 3.0-5.0')
    plt.savefig("boxplot25length.png")

    # Boxplot of coverage
    fig = plt.figure(figsize=(32, 8))
    fig.add_subplot(1, 4, 1)
    plt.boxplot(coverage[0], positions=[1, 2, 3], widths=0.4)
    plt.xticks([1, 2, 3], ['POSI', 'Naive', 'Data Splitting'])
    plt.ylabel('Coverage')
    plt.title('SNR 0.2-0.5')
    fig.add_subplot(1, 4, 2)
    plt.boxplot(coverage[1], positions=[1, 2, 3], widths=0.4)
    plt.xticks([1, 2, 3], ['POSI', 'Naive', 'Data Splitting'])
    plt.ylabel('Coverage')
    plt.title('SNR 0.5-1.0')
    fig.add_subplot(1, 4, 3)
    plt.boxplot(coverage[2], positions=[1, 2, 3], widths=0.4)
    plt.xticks([1, 2, 3], ['POSI', 'Naive', 'Data Splitting'])
    plt.ylabel('Coverage')
    plt.title('SNR 1.0-3.0')
    fig.add_subplot(1, 4, 4)
    plt.boxplot(coverage[3], positions=[1, 2, 3], widths=0.4)
    plt.xticks([1, 2, 3], ['POSI', 'Naive', 'Data Splitting'])
    plt.ylabel('Coverage')
    plt.title('SNR 3.0-5.0')
    plt.savefig("boxplot25coverage.png")

    #Plot distribution of pivots
    pivots = pivot[0]
    pivots_naive = pivot_naive[0]
    plt.clf()
    grid = np.linspace(0, 1, 101)
    points = [np.max(np.searchsorted(np.sort(np.asarray(pivots)), i, side='right')) / np.float(np.shape(pivots)[0]) for
              i in np.linspace(0, 1, 101)]
    points_naive = [np.max(np.searchsorted(np.sort(np.asarray(pivots_naive)), i, side='right')) / np.float(
        np.shape(pivots_naive)[0]) for i in np.linspace(0, 1, 101)]
    fig = plt.figure(figsize=(32, 8))
    fig.tight_layout()
    fig.add_subplot(1, 4, 1)
    plt.plot(grid, points, c='blue', marker='^')
    plt.plot(grid, points_naive, c='red', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.title('Task Sparsity 0%, SNR 0.2-0.5')

    pivots = pivot[1]
    pivots_naive = pivot_naive[1]
    grid = np.linspace(0, 1, 101)
    points = [np.searchsorted(np.sort(np.asarray(pivots)), i, side='right') / np.float(np.shape(pivots)[0]) for i in
              np.linspace(0, 1, 101)]
    points_naive = [
        np.searchsorted(np.sort(np.asarray(pivots_naive)), i, side='right') / np.float(np.shape(pivots_naive)[0]) for i
        in np.linspace(0, 1, 101)]
    fig.add_subplot(1, 4, 2)
    plt.plot(grid, points, c='blue', marker='^')
    plt.plot(grid, points_naive, c='red', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.title('Task Sparsity 0%, SNR 0.5-1.0')

    pivots = pivot[2]
    pivots_naive = pivot_naive[2]
    grid = np.linspace(0, 1, 101)
    points = [np.searchsorted(np.sort(np.asarray(pivots)), i, side='right') / np.float(np.shape(pivots)[0]) for i in
              np.linspace(0, 1, 101)]
    points_naive = [
        np.searchsorted(np.sort(np.asarray(pivots_naive)), i, side='right') / np.float(np.shape(pivots_naive)[0]) for i
        in np.linspace(0, 1, 101)]
    fig.add_subplot(1, 4, 3)
    plt.plot(grid, points, c='blue', marker='^')
    plt.plot(grid, points_naive, c='red', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.title('Task Sparsity 0%, SNR 1.0-3.0')

    pivots = pivot[3]
    pivots_naive = pivot_naive[3]
    grid = np.linspace(0, 1, 101)
    points = [np.searchsorted(np.sort(np.asarray(pivots)), i, side='right') / np.float(np.shape(pivots)[0]) for i in
              np.linspace(0, 1, 101)]
    points_naive = [
        np.searchsorted(np.sort(np.asarray(pivots_naive)), i, side='right') / np.float(np.shape(pivots_naive)[0]) for i
        in np.linspace(0, 1, 101)]
    fig.add_subplot(1, 4, 4)
    plt.plot(grid, points, c='blue', marker='^')
    plt.plot(grid, points_naive, c='red', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.title('Task Sparsity 0%, SNR 3.0-5.0')

    plt.savefig("0_p100.png")

if __name__ == "__main__":
    main()