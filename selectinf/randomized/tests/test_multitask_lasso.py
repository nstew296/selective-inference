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

    lambdamin = 1.25
    lambdamax = 4.0
    weights = np.arange(np.log(lambdamin), np.log(lambdamax), (np.log(lambdamax) - np.log(lambdamin)) / 25)
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

                estimate, observed_info_mean, Z_scores, pvalues, intervals, diffs, time_diff = multi_lasso.multitask_inference_hetero(
                dispersions=dispersions)

            except:
                sum = 0
                for j in range(ntask):
                    sum += 0.5*np.linalg.norm(response_vars_test[j], 2)
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

    #errors = errors[errors>0]
    print("errors",np.maximum(errors-(np.std(errors)/np.sqrt(len(errors))),0))
    #idx_min_error = np.int(np.argmin(np.maximum(errors-(np.std(errors)/np.sqrt(len(errors))),0)))
    idx_min_error = np.int(np.argmin(np.maximum(errors, 0)))
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

    lambdamin = 0.75
    lambdamax = 3.5
    weights = np.arange(np.log(lambdamin), np.log(lambdamax), (np.log(lambdamax) - np.log(lambdamin)) / 25)
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
                    sum += np.linalg.norm(response_vars_test[j], 2)
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

    errors = errors[errors>0]
    print("errors naive",np.argmin(np.maximum(errors-(np.std(errors)/np.sqrt(len(errors))),0)))
    idx_min_error = np.int(np.argmin(np.maximum(errors-(np.std(errors)/np.sqrt(len(errors))),0)))
    lam_min = weights[idx_min_error]
    print(lam_min,"tuning param naive")


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

    while True:

        feature_weight = weight * np.ones(p)

        sigmas_ = sigma
        randomizer_scales = randomizer_scale * np.array([sigmas_[i] for i in range(ntask)])

        _initial_omega = np.array(
            [randomizer_scales[i] * gaussian_noise[(i * p):((i + 1) * p)] for i in range(ntask)]).T

        if link == "identity":
            multi_lasso = multi_task_lasso.gaussian(predictor_vars,
                                                    response_vars,
                                                    feature_weight,
                                                    ridge_term=None,
                                                    randomizer_scales=randomizer_scales)

        if link == "logit":
            multi_lasso = multi_task_lasso.logistic(predictor_vars,
                                                    response_vars,
                                                    feature_weight,
                                                    ridge_term=None,
                                                    randomizer_scales=randomizer_scales)

        if link == "log":
            multi_lasso = multi_task_lasso.poisson(predictor_vars,
                                                   response_vars,
                                                   feature_weight,
                                                   ridge_term=None,
                                                   randomizer_scales=randomizer_scales)

        active_signs = multi_lasso.fit(perturbations=_initial_omega)

        if (active_signs != 0).sum() > 0:

            dispersions = sigma ** 2

            estimate, observed_info_mean, Z_scores, pvalues, intervals, mean_diff, time_diff = multi_lasso.multitask_inference_hetero(
                dispersions=dispersions)

            beta_target = []

            for i in range(ntask):
                X, y = multi_lasso.loglikes[i].data
                beta_target.extend(np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(X.dot(beta[:, i])))

            beta_target = np.asarray(beta_target)
            pivot_ = ndist.cdf((estimate - beta_target) / np.sqrt(np.diag(observed_info_mean)))
            pivot = 2 * np.minimum(pivot_, 1. - pivot_)

            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

            return coverage, intervals[:, 1] - intervals[:, 0], pivot, mean_diff, time_diff


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

    while True:

        feature_weight = weight * np.ones(p)

        sigmas_ = sigma

        perturbations = np.zeros((p, ntask))

        if link == "identity":
            multi_lasso = multi_task_lasso.gaussian(predictor_vars,
                                                    response_vars,
                                                    feature_weight,
                                                    ridge_term=None,
                                                    randomizer_scales=1. * sigmas_,
                                                    perturbations=perturbations)

        if link == "logit":
            multi_lasso = multi_task_lasso.logistic(predictor_vars,
                                                    response_vars,
                                                    feature_weight,
                                                    ridge_term=None,
                                                    randomizer_scales=1. * sigmas_,
                                                    perturbations=perturbations)

        if link == "log":
            multi_lasso = multi_task_lasso.poisson(predictor_vars,
                                                   response_vars,
                                                   feature_weight,
                                                   ridge_term=None,
                                                   randomizer_scales=1. * sigmas_,
                                                   perturbations=perturbations)
        active_signs = multi_lasso.fit()

        dispersions = sigma ** 2

        coverage = []
        pivot = []

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
                coverage.extend((beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1]))
                pivot_ = ndist.cdf((observed_target - beta_target) / np.sqrt(np.diag(cov_target)))
                pivot.extend(2 * np.minimum(pivot_, 1. - pivot_))

        return np.asarray(coverage), intervals[:, 1] - intervals[:, 0], np.asarray(pivot)





def test_coverage(signal,nsim=100):
    cov = []
    len = []
    pivots = []
    penalties = []
    diff_solns = []
    time_diffs = []

    cov_naive = []
    len_naive = []
    pivots_naive = []
    penalties_naive = []
    time_diffs = []

    ntask = 5

    penalty_hetero, predictor, coef = cross_validate_posi_hetero(ntask=ntask,
                                                                 nsamples=2000 * np.ones(ntask),
                                                                 p=50,
                                                                 global_sparsity=0.9,
                                                                 task_sparsity=.5,
                                                                 sigma=1. * np.ones(ntask),
                                                                 signal_fac=np.array(signal),
                                                                 rhos=.7 * np.ones(ntask),
                                                                 link="identity",
                                                                 randomizer_scale=1)

    penalty_hetero_naive, predictor_naive, coef_naive = cross_validate_naive_hetero(ntask=ntask,
                                                                                    nsamples=2000 * np.ones(ntask),
                                                                                    p=50,
                                                                                    global_sparsity=0.9,
                                                                                    task_sparsity=.5,
                                                                                    sigma=1. * np.ones(ntask),
                                                                                    signal_fac=np.array(signal),
                                                                                    rhos=.7 * np.ones(ntask),
                                                                                    link="identity")

    penalties.append(penalty_hetero)
    penalties_naive.append(penalty_hetero_naive)

    for n in range(nsim):


        print(n,"n sim")

        try:
            #print(pivot,"pivot")

            coverage, length, pivot, diff_soln, time_difference = test_multitask_lasso_hetero(predictor,
                                                                  coef,
                                                                  sigma=1. * np.ones(ntask),
                                                                  weight=np.float(penalty_hetero),
                                                                  link = "identity",
                                                                  randomizer_scale = 1)
            cov.extend(coverage)
            len.extend(length)
            pivots.extend(pivot)
            diff_solns.append(diff_soln)
            time_diffs.append(time_difference)

        except:
            print("no selection posi")


        try:

             coverage_naive, length_naive, pivot_naive = test_multitask_lasso_naive_hetero(predictor_naive,
                                                                        coef_naive,
                                                                        sigma=1. * np.ones(ntask),
                                                                        link = "identity",
                                                                        weight=np.float(penalty_hetero_naive))


             cov_naive.extend(coverage_naive)
             len_naive.extend(length_naive)
             pivots_naive.extend(pivot_naive)


        except:
            print("no selection naive")


        print("iteration completed ", n)
        print("coverage so far ", np.mean(np.asarray(cov)))
        print("naive coverage so far ", np.mean(np.asarray(cov_naive)))
        print("length so far ", np.mean(np.asarray(len)))
        print("length so far ", np.mean(np.asarray(len_naive)))
        print("mean penalty", np.mean(np.asarray(penalties)))
        print("mean penalty naive", np.mean(np.asarray(penalties_naive)))
        print("mean diff so far", np.mean(np.asarray(diff_solns)))
        print("mean time diff so far", np.mean(np.asarray(time_diffs)))

    return([pivots,pivots_naive,[np.mean(np.asarray(penalties)),np.mean(np.asarray(penalties_naive)),np.mean(diff_solns)]])

def main():

    signals = [[0.2,0.5],[0.5,1.0],[1.0,3.0],[3.0,5.0]]
    pivot = {0:[],1:[],2:[],3:[]}
    pivot_naive = {0:[], 1:[],2:[],3:[]}
    tuning = {0: [], 1: [],2:[],3:[]}
    hellinger_dist = {0: [], 1: [], 2: [], 3: []}

    for i in range(len(signals)):
        sims = test_coverage(signals[i],200)
        pivot[i] = sims[0]
        pivot_naive[i] = sims[1]
        tuning[i] = sims[2]

    pivots = pivot[0]
    pivots_naive = pivot_naive[0]
    plt.clf()
    grid = np.linspace(0, 1, 101)
    points = [np.max(np.searchsorted(np.sort(np.asarray(pivots)), i, side='right')) / np.float(np.shape(pivots)[0]) for i in np.linspace(0, 1, 101)]
    dist_posi = np.sum([points[i]*np.log((points[i]+0.00001)/((np.float(i)+1.0)/100)) for i in range(100)])
    points_naive = [np.max(np.searchsorted(np.sort(np.asarray(pivots_naive)), i, side='right')) / np.float(np.shape(pivots_naive)[0]) for i in np.linspace(0, 1, 101)]
    dist_naive = np.sum([points_naive[i]*np.log((points_naive[i]+0.00001)/((np.float(i)+1.0)/100)) for i in range(100)])
    hellinger_dist[0] = [dist_posi,dist_naive]
    fig = plt.figure(figsize=(32, 8))
    fig.tight_layout()
    fig.add_subplot(1, 4, 1)
    plt.plot(grid, points, c='blue', marker='^')
    plt.plot(grid, points_naive, c='red', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.title('Task Sparsity 50%, SNR 0.2-0.5')

    pivots = pivot[1]
    pivots_naive = pivot_naive[1]
    grid = np.linspace(0, 1, 101)
    points = [np.searchsorted(np.sort(np.asarray(pivots)), i, side='right') / np.float(np.shape(pivots)[0]) for i in
              np.linspace(0, 1, 101)]
    dist_posi = np.sum([points[i]*np.log((points[i]+0.00001)/((np.float(i)+1)/100)) for i in range(100)])
    points_naive = [np.searchsorted(np.sort(np.asarray(pivots_naive)), i, side='right') / np.float(np.shape(pivots_naive)[0]) for i in np.linspace(0, 1, 101)]
    dist_naive = np.sum([points_naive[i]*np.log((points_naive[i]+0.00001)/((np.float(i)+1.0)/100)) for i in range(100)])
    hellinger_dist[1] = [dist_posi, dist_naive]
    fig.add_subplot(1, 4, 2)
    plt.plot(grid, points, c='blue', marker='^')
    plt.plot(grid, points_naive, c='red', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.title('Task Sparsity 50%, SNR 0.5-1.0')

    pivots = pivot[2]
    pivots_naive = pivot_naive[2]
    grid = np.linspace(0, 1, 101)
    points = [np.searchsorted(np.sort(np.asarray(pivots)), i, side='right') / np.float(np.shape(pivots)[0]) for i in np.linspace(0, 1, 101)]
    dist_posi = np.sum([points[i]*np.log((points[i]+0.00001)/((np.float(i)+1)/100)) for i in range(100)])
    points_naive = [np.searchsorted(np.sort(np.asarray(pivots_naive)), i, side='right') / np.float(np.shape(pivots_naive)[0]) for i in np.linspace(0, 1, 101)]
    dist_naive = np.sum([points_naive[i]*np.log((points_naive[i]+0.00001)/((np.float(i)+1)/100)) for i in range(100)])
    hellinger_dist[2] = [dist_posi, dist_naive]
    fig.add_subplot(1, 4, 3)
    plt.plot(grid, points, c='blue', marker='^')
    plt.plot(grid, points_naive, c='red', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.title('Task Sparsity 50%, SNR 1.0-3.0')

    pivots = pivot[3]
    pivots_naive = pivot_naive[3]
    grid = np.linspace(0, 1, 101)
    points = [np.searchsorted(np.sort(np.asarray(pivots)), i, side='right') / np.float(np.shape(pivots)[0]) for i in np.linspace(0, 1, 101)]
    dist_posi = np.sum([points[i]*np.log((points[i]+0.00001)/((np.float(i)+1)/100)) for i in range(100)])
    points_naive = [np.searchsorted(np.sort(np.asarray(pivots_naive)), i, side='right') / np.float(np.shape(pivots_naive)[0]) for i in np.linspace(0, 1, 101)]
    dist_naive = np.sum([points_naive[i]*np.log((points_naive[i]+0.00001)/((np.float(i)+1)/100)) for i in range(100)])
    hellinger_dist[3] = [dist_posi, dist_naive]
    fig.add_subplot(1, 4, 4)
    plt.plot(grid, points, c='blue', marker='^')
    plt.plot(grid, points_naive, c='red', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.title('Task Sparsity 50%, SNR 3.0-5.0')

    plt.savefig("50_90_pgd_hess_ub_time2.png")

    print(tuning)
    print(hellinger_dist)

    #scale = [1]
    #coverage = []
    #coverage_er = []
    #lengths = []
    #lengths_er = []

    #for i in range(len(scale)):
        #print(scale[i], 'signal')
        #results = test_coverage(scale[i], nsim=200)
        #coverage = np.append(coverage, results[0])
        #print(coverage,"cov")
        #coverage_er = np.append(coverage_er, 1.64 * results[1] / np.sqrt(100))
        #lengths = np.append(lengths, results[1])
        #lengths_er = np.append(lengths_er, 1.64 * results[3] / np.sqrt(100))

    #fig = plt.figure()
    #plt.errorbar(scale, coverage, yerr=coverage_er)
    #axes = plt.gca()
    #axes.set_ylim([0, 1])
    #plt.ylabel('Mean Coverage')
    #plt.xlabel('Randomizer Scale (sigma_omega)')
    #plt.title('Coverage by Randomizer Scale')
    #plt.savefig("coverage.png")
    #plt.show()

    #fig = plt.figure()
    #plt.errorbar(scale, lengths, yerr=lengths_er)
    #plt.ylabel('Mean Length')
    #plt.xlabel('Randomizer Scale (sigma_omega)')
    #plt.title('Length by Randomizer Scale')
    #plt.savefig("length.png")
    #plt.show()


if __name__ == "__main__":
    main()