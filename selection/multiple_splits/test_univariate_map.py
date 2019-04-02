from __future__ import division, print_function
import numpy as np, os

from selection.randomized.lasso import lasso, selected_targets, full_targets, debiased_targets
import seaborn as sns
import pylab
import matplotlib.pyplot as plt
import scipy.stats as stats
from selection.multiple_splits.utils import sim_xy, glmnet_lasso_cv1se, glmnet_lasso_cvmin, glmnet_lasso

from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm as ndist

def test_lasso_estimate(X, y, sigma, beta, randomizer_scale):

    while True:
        dispersion = None
        sigma_ = np.std(y)
        print("sigma ", sigma, sigma_)
        n, p = X.shape

        #lam_theory = sigma_ * np.mean(np.fabs(np.dot(X[:,1:].T, np.random.standard_normal((n, 2000)))).max(0))
        glm_LASSO_1se, lam_1se = glmnet_lasso_cv1se(X, y)
        lasso_sol = lasso.gaussian(X,
                                   y,
                                   feature_weights=np.append(0.0001, np.ones(p - 1) * n * lam_1se),
                                   randomizer_scale= np.sqrt(n)* randomizer_scale * sigma_)

        signs = lasso_sol.fit()
        nonzero = signs != 0
        print("solution", nonzero.sum(), nonzero[0])
        if nonzero.sum()>1:
            beta_target = np.linalg.inv(X[:, nonzero].T.dot(X[:, nonzero])).dot(X[:, nonzero].T).dot(X.dot(beta))
            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(lasso_sol.loglike,
                                              lasso_sol._W,
                                              nonzero,
                                              dispersion=dispersion)

            estimate, info, _, pval, intervals, _ = lasso_sol.selective_MLE(observed_target,
                                                                            cov_target,
                                                                            cov_target_score,
                                                                            alternatives)

            observed_target_uni = (observed_target[0]).reshape((1,))
            cov_target_uni = (np.diag(cov_target)[0]).reshape((1, 1))
            cov_target_score_uni = cov_target_score[0, :].reshape((1, p))

            estimate_uni, info_uni, _, pval_uni, intervals_uni, _ = lasso_sol.selective_MLE(observed_target_uni,
                                                                                            cov_target_uni,
                                                                                            cov_target_score_uni,
                                                                                            alternatives)

            print("check ", beta_target[0], estimate_uni, estimate[0], info_uni, np.diag(info)[0])
            return np.asscalar((estimate[0]-beta_target[0])/np.sqrt(info_uni))


def test_uni(n=200, p=1000, nval=200, alpha= 2., rho=0.70, s=10, beta_type=1, snr=0.20, randomizer_scale=1., nsim=100):

    _pivot = []
    for i in range(nsim):
        X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, alpha=alpha, rho=rho, s=s, beta_type=beta_type,
                                                snr=snr)
        X -= X.mean(0)[None, :]
        scale = (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
        X /= scale
        y = y - y.mean()
        _pivot.append(test_lasso_estimate(X, y, sigma, beta, randomizer_scale=randomizer_scale))

    plt.clf()
    ecdf_MLE = ECDF(ndist.cdf(np.asarray(_pivot)))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()

#test_uni(nsim=200)

def test_lasso_estimate_sum(X, y, sigma, beta, randomizer_scale = 1.):

    while True:
        dispersion = None
        sigma_ = np.std(y)
        print("sigma ", sigma, sigma_)
        n, p = X.shape

        lam_theory = sigma_ * np.mean(np.fabs(np.dot(X[:,1:].T, np.random.standard_normal((n, 2000)))).max(0))
        #glm_LASSO_1se, lam_1se = glmnet_lasso_cv1se(X, y)
        lasso_sol = lasso.gaussian(X,
                                   y,
                                   feature_weights=np.append(0.001, np.ones(p - 1) * lam_theory),
                                   randomizer_scale= np.sqrt(n)* randomizer_scale * sigma_)

        signs = lasso_sol.fit()
        nonzero = signs != 0
        print("solution", nonzero.sum(), nonzero[0])
        if nonzero.sum()>0:
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
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

            estimate_uni, info_uni, _, pval_uni, intervals_uni, _, _, _ = lasso_sol.selective_MLE(observed_target_uni,
                                                                                            cov_target_uni,
                                                                                            cov_target_score_uni,
                                                                                            alternatives)

            return np.asscalar(estimate_uni), np.asscalar(info_uni), np.asscalar(beta_target[0])

def test_sum(n=200, p=1000, nval=200, alpha= 2., rho=0.70, s=10, beta_type=1, snr=0.20, randomizer_scale=1., nsim=100):

    _pivot = []
    for i in range(nsim):
        X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, alpha=alpha, rho=rho, s=s, beta_type=beta_type,
                                                snr=snr)
        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
        y = y - y.mean()

        cc_1, var_1, tar_1 = test_lasso_estimate_sum(X, y, sigma, beta, randomizer_scale)
        cc_2, var_2, tar_2 = test_lasso_estimate_sum(X, y, sigma, beta, randomizer_scale)
        pooled_est = (cc_1 + cc_2)/2.
        pooled_sd = np.sqrt(var_1 + var_2)/2.
        mean = (tar_1 +tar_2)/2.

        _pivot.append((pooled_est-mean)/pooled_sd)

    plt.clf()
    ecdf_MLE = ECDF(ndist.cdf(np.asarray(_pivot)))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()

test_sum(nsim=200)