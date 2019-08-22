from __future__ import division, print_function
import numpy as np, os
import pandas as pd

from selection.randomized.lasso import lasso, selected_targets, full_targets, debiased_targets
from selection.multiple_splits.utils import sim_xy

def upper_triangular(A):
    N = A.shape[0]
    return np.array([p for i, p in enumerate(A.flatten()) if i > (i / N) * (1 + N)])

def test_mse_theory(seedn, n=100, p=500, nval=100, alpha=2., rho=0.70, s=10, beta_type=1, snr=0.55,
                    randomizer_scale=1., B=5):

    X, y, _, _, Sigma, beta, sigma, _ = sim_xy(n=n, p=p, nval=nval, seedn=seedn, alpha=alpha, rho=rho, s=s,
                                               beta_type=beta_type,
                                               snr=snr)
    X -= X.mean(0)[None, :]
    y = y - y.mean()
    scaling = X.std(0)[None, :] * np.sqrt(n / (n - 1.))

    X /= scaling
    dispersion = None
    sigma_ = np.std(y)
    # dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
    # sigma_ = np.sqrt(dispersion)

    lam = np.ones(p - 1) * sigma_ * 1. * np.mean(np.fabs(np.dot(X[:, 1:].T, np.random.standard_normal((n, 2000)))).max(0))

    sel_mle = np.zeros(B)
    covar_mle = np.zeros((B,B))
    alpha_target_randomized = np.zeros(B)

    for j in range(B):
        lasso_sol = lasso.gaussian(X,
                                   y,
                                   feature_weights=np.append(0.01, lam),
                                   randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)
        signs = lasso_sol.fit()
        nonzero = signs != 0
        print("selected ", nonzero.sum(), nonzero[0])

        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(lasso_sol.loglike,
                                          lasso_sol._W,
                                          nonzero,
                                          dispersion=dispersion)

        observed_target_uni = (observed_target[0]).reshape((1,))
        cov_target_uni = (np.diag(cov_target)[0]).reshape((1, 1))
        cov_target_score_uni = cov_target_score[0, :].reshape((1, p))

        mle, obs_infoinv, _, _, _, _, _, _ = lasso_sol.selective_MLE(observed_target_uni,
                                                                     cov_target_uni,
                                                                     cov_target_score_uni,
                                                                     alternatives)

        sel_mle[j] = mle
        covar_mle[j,j] = np.asscalar(obs_infoinv)
        alpha_target_randomized[j] = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))[0]

    sample_mean = np.mean(sel_mle)
    sample_cov = np.outer(sel_mle-sample_mean, sel_mle-sample_mean)
    for k in range(B):
        sample_cov[k,k] = covar_mle[k,k]
    #cov_terms = (sample_cov.sum()-np.diag(sample_cov).sum())/(B*(B-1.))
    #print("check covariance matrix ", sel_mle, sample_cov, cov_terms)

    #var_bagged = (np.diag(covar_mle).sum())/float(B**2) + (cov_terms)/float(B**2)
    cov_term = (sample_cov.sum()- np.diag(sample_cov).sum())/(B*(B-1.))
    var_term = np.diag(sample_cov).sum()/float(B)
    var_bagged = n* (sample_cov.sum()/float(B**2))
    #var_bagged = n*(var_term+ ((B-1.)* cov_term))/float(B)
    print("check ", var_bagged, var_term + B*cov_term)
    if var_bagged<0:
        print("NEGATIVE")
    pivot = (np.mean(sel_mle) - np.mean(alpha_target_randomized))/np.sqrt(var_bagged)

    lc_pooled = sample_mean - 1.65 * np.sqrt(var_bagged)
    uc_pooled = sample_mean + 1.65 * np.sqrt(var_bagged)
    coverage = (lc_pooled < np.mean(alpha_target_randomized)) * (np.mean(alpha_target_randomized) < uc_pooled)

    return pivot, coverage

import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm as ndist

def main(nsim = 500):

    _pivot = []
    cov = 0.
    for i in range(nsim):
        pivot, coverage = test_mse_theory(seedn = i+1, n=100, p=500, nval=100, alpha=2., rho=0.35, s=5, beta_type=1, snr=0.60,randomizer_scale=1., B=50)
        _pivot.append(pivot)
        cov += coverage
        print("coverage so far  ", i+1, cov/(i+1.))

    plt.clf()
    ecdf_MLE = ECDF(ndist.cdf(np.asarray(_pivot)))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()

main(nsim = 200)