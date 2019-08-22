import numpy as np
import random
from scipy.stats import norm
import pandas as pd
from selection.algorithms.lasso import lasso_full
from selection.randomized.lasso import lasso, selected_targets
import matplotlib.pyplot as plt

def split(Y, G, S, X, alpha, alpha0, alpha1, prop_sel=0.5):

    n, p = G.shape
    obs_selection = np.full(n, False, dtype='bool')
    obs_selection[random.sample(range(n), int(np.floor(n * prop_sel)))] = True
    true_mean = X[~obs_selection, :].dot(np.append(alpha0, alpha1))

    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * (np.sqrt(X.shape[0]) / np.sqrt(X.shape[0] - 1.)))
    Y = Y - Y.mean()

    Y1, X1, n1 = Y[obs_selection], X[obs_selection, :], sum(obs_selection)
    Y2, X2, n2 = Y[~obs_selection], X[~obs_selection, :], n - n1

    # nonrandomized lasso
    sigma_ = np.std(Y1)
    lam = sigma_ * 1. * np.mean(np.fabs(np.dot(X1.T, np.random.standard_normal((n1, 2000)))).max(0))
    W = lam * np.ones(X1.shape[1])
    W[-1:] = 0.

    canonical_lasso = lasso_full.gaussian(X1, Y1, W)
    signs = canonical_lasso.fit()
    nonzero = signs != 0

    Xsel = X2[:, nonzero]
    OLS_est = np.linalg.pinv(Xsel).dot(Y2)
    var = np.linalg.inv(Xsel.T.dot(Xsel)) * 5. # use true noise var here, instead of np.var(Y2)
    quantile = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    split_intervals = np.vstack([OLS_est - quantile * np.sqrt(np.diag(var)),
                                 OLS_est + quantile * np.sqrt(np.diag(var))]).T

    # selective target
    X_orig = np.concatenate((G, S), axis=1)
    Xsel_orig = X_orig[~obs_selection, :][:, nonzero]
    selective_target = np.linalg.pinv(Xsel_orig).dot(true_mean)

    # linear combination
    G2 = G[~obs_selection,:][:,nonzero[:p]]
    S2 = S[~obs_selection,:]

    # G2 = X2[:, nonzero][:, :(sum(nonzero) - 1)]
    # S2 = X2[:, -1][:, None]
    rT = np.append(np.linalg.pinv(S2).dot(G2), 0.)
    est = rT.dot(OLS_est)
    var = rT.dot(var).dot(rT.T)
    interval = [est - quantile * np.sqrt(var), est + quantile * np.sqrt(var)]
    target = rT.dot(selective_target)

    print(split_intervals.shape)
    print(selective_target.shape)

    return split_intervals, selective_target, interval, target, 2* quantile * np.sqrt(var), (est-target)**2

def selective_MLE(Y, G, S, X, true_mean, alpha):
    n, p = G.shape
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * (np.sqrt(X.shape[0]) / np.sqrt(X.shape[0] - 1.)))
    Y = Y - Y.mean()

    randomizer_scale = 1.
    sigma_ = np.std(Y)

    lam = sigma_ * 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    W = lam * np.ones(X.shape[1])
    W[-1:] = 0.

    randomized_lasso = lasso.gaussian(X, Y, W, randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)
    signs = randomized_lasso.fit()
    nonzero = signs != 0

    (observed_target,
     cov_target,
     cov_target_score,
     alternatives) = selected_targets(randomized_lasso.loglike,
                                      randomized_lasso._W,
                                      nonzero,
                                      dispersion=None)

    print("check ", nonzero[p])
    MLE_estimate, obs_info, _, _, MLE_intervals, _, _, _ = randomized_lasso.selective_MLE(observed_target=observed_target,
                                                                                    cov_target=cov_target,
                                                                                    cov_target_score=cov_target_score,
                                                                                    level=1. - alpha)

    # selective target
    X_orig = np.concatenate((G, S), axis=1)
    selective_target = np.linalg.pinv(X_orig[:, nonzero]).dot(true_mean)
    #selective_target = np.linalg.pinv(X[:, nonzero]).dot(true_mean)

    # linear combination
    rT = np.append(np.linalg.pinv(S).dot(G[:, nonzero[:p]]), 0.)
    # rT = np.full_like(rT, 0.) # 1st canonical basis vector as sanity check
    # rT = np.zeros(nonzero.sum())
    # rT[:1] = 1.
    #rT = np.random.standard_normal(nonzero.sum())

    est = rT.dot(MLE_estimate)
    naive_est = rT.dot(observed_target)

    var = rT.dot(obs_info).dot(rT.T)
    naive_var = rT.dot(cov_target).dot(rT.T)

    quantile = norm.ppf(1 - alpha / 2, loc=0, scale=1)
    interval = (est - quantile * np.sqrt(var), est + quantile * np.sqrt(var))
    interval_naive = (naive_est - quantile * np.sqrt(naive_var), naive_est + quantile * np.sqrt(naive_var))
    target = rT.dot(selective_target)

    print(MLE_intervals.shape)
    print(selective_target.shape)
    print(interval_naive)

    return MLE_intervals, selective_target, interval, target, 2*quantile * np.sqrt(var), (est-target)**2., est, np.sqrt(var), interval_naive

def simulation(nreps, c, d):
    # global params
    n, p = 100, 500 # 300, 500
    c = c
    d = d
    overlap = 10
    eps1 = 10.
    alpha1 = 0.1
    alpha = 0.1

    COVERAGE = 0. # coverage for linear combination, selective MLE interval
    NAIVE_COVERAGE = 0.
    SEL_TARGET_COVERAGE = 0. # coverage for the selective target, selective MLE interval
    LENGTH = 0.
    RISK = 0.
    split_COVERAGE = 0. # coverage for linear combination, splitting (0.5) interval
    split_SEL_TARGET_COVERAGE = 0. # coverage for selective target, splitting (0.5) interval
    split_LENGTH = 0.
    split_RISK = 0.

    target_ = []
    linear_est_ = []

    # fix gamma across replications
    gamma = np.zeros(p)
    gamma[np.random.choice(p, 15, replace=False)] = np.random.uniform(-1, 1, 15)

    for i in range(nreps):
        S = np.random.standard_normal((n, 1))
        error_cov = np.identity(p)
        error = np.random.multivariate_normal(np.zeros(p), error_cov, n)
        G = c * S.dot(gamma.reshape((1, p))) + error

        gamma_nonzero = np.nonzero(gamma)[0]
        alpha0 = np.zeros(p)
        true_mediators = np.random.choice(gamma_nonzero, overlap, replace=False)
        alpha0[true_mediators] = d*1.
        alpha0[np.random.choice(np.setdiff1d(np.arange(p), gamma_nonzero),
                                15 - overlap,
                                replace=False)] = 1.
        Y = G.dot(alpha0) + S.dot(alpha1).reshape((n,)) + np.random.standard_normal(n) * np.sqrt(eps1)
        X = np.concatenate((G, S), axis=1)
        true_mean = X.dot(np.append(alpha0, alpha1)) # before standardization


        # selective MLE intervals
        MLE_intervals, selective_target, interval, target, length, risk, est, sd, interval_naive = selective_MLE(Y, G, S, X, true_mean, alpha)
        #pop_target = c * np.append(gamma[nonzero_lasso[:p]], 0.).dot(selective_target)
        # print("population target ", np.linalg.pinv(S).dot(G[:, nonzero_lasso[:p]]), c * gamma[nonzero_lasso[:p]])
        # print("population target ", pop_target, target)
        target_.append(target)
        linear_est_.append((est-target)/sd)

        COVERAGE += (target >= interval[0]) * (target <= interval[1])
        LENGTH += length
        SEL_TARGET_COVERAGE += np.mean((selective_target > MLE_intervals[:, 0]) * (selective_target < MLE_intervals[:, 1]))
        RISK += risk

        NAIVE_COVERAGE += np.mean((target > interval_naive[0]) * (target < interval_naive[1]))
        print("coverage (selective target, linear comb)", SEL_TARGET_COVERAGE / (i + 1.), COVERAGE / (i + 1.), NAIVE_COVERAGE/(i+1.))

        split_intervals, selective_target, interval, target, split_length, split_risk = split(Y, G, S, X, alpha, alpha0, alpha1, prop_sel=0.70)
        split_COVERAGE += (target > interval[0]) * (target < interval[1])
        split_SEL_TARGET_COVERAGE += np.mean(
            (selective_target > split_intervals[:, 0]) * (selective_target < split_intervals[:, 1]))
        split_LENGTH += split_length
        split_RISK += split_risk
        print("split coverage (selective target, linear comb)", split_SEL_TARGET_COVERAGE / (i + 1.),
              split_COVERAGE / (i + 1.))

        #print("risk", (i+1), RISK/(i+1.), split_RISK/(i+1.))

        #print("compare lengths ", i+1, LENGTH/(i+1.), split_LENGTH/(i+1.))

    #return SEL_TARGET_COVERAGE / float(nreps), COVERAGE / float(nreps), split_SEL_TARGET_COVERAGE / float(nreps), split_COVERAGE / float(nreps)
    return target_, linear_est_

simulation(nreps=300, c=0.5, d=0.5)