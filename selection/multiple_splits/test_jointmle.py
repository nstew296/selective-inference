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

def test_lasso_estimate(X, y, sigma, beta, randomizer_scale = 1.):

    while True:
        dispersion = None
        sigma_ = np.std(y)
        print("sigma ", sigma, sigma_)
        n, p = X.shape

        lam_theory = sigma_ * np.mean(np.fabs(np.dot(X[:,1:].T, np.random.standard_normal((n, 2000)))).max(0))

        lasso_sol = lasso.gaussian(X,
                                  y,
                                  feature_weights=np.append(0.001, np.ones(p - 1) * lam_theory),
                                  randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

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

            print(lasso_sol.selective_MLE(observed_target_uni,
                                          cov_target_uni,
                                          cov_target_score_uni,
                                          alternatives))

            _, _, _, _, _, _, mle_comp, var_comp = lasso_sol.selective_MLE(observed_target_uni,
                                                                       cov_target_uni,
                                                                       cov_target_score_uni,
                                                                       alternatives)

            #print("check shape ", np.linalg.pinv(X[:, nonzero])[0,:].shape)
            return np.asscalar(mle_comp), np.asscalar(var_comp), np.asscalar(observed_target_uni),\
                   np.asscalar(cov_target_uni), (sigma) *np.linalg.pinv(X[:, nonzero])[0,:], np.asscalar(beta_target[0])

def test_sum(n=200, p=1000, nval=200, alpha= 2., rho=0.70, s=10, beta_type=1, snr=0.20, randomizer_scale=1., nsim=100):

    _pivot = []
    cov = 0.
    for i in range(nsim):
        X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, alpha=alpha, rho=rho, s=s, beta_type=beta_type,
                                                snr=snr)
        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
        y = y - y.mean()

        mle_comp_1, var_comp_1, ls_1, ls_var_1, ls_covar_1, target_1 = test_lasso_estimate(X, y, sigma, beta, randomizer_scale)
        mle_comp_2, var_comp_2, ls_2, ls_var_2, ls_covar_2, target_2 = test_lasso_estimate(X, y, sigma, beta, randomizer_scale)

        Sigma = np.diag(np.asarray([ls_var_1, ls_var_2]))
        Sigma[0, 1] = ls_covar_1.T.dot(ls_covar_2)
        Sigma[1, 0] = Sigma[0, 1]
        mle = np.array((ls_1, ls_2)) + Sigma.dot(np.array((mle_comp_1, mle_comp_2)))
        inv_info = Sigma + Sigma.dot(np.diag(np.array((var_comp_1, var_comp_2)))).dot(Sigma)

        pooled_est = (mle[0]+mle[1])/2.
        pooled_sd = np.sqrt(inv_info[0,0]+ inv_info[1,1]+ 2*inv_info[0,1])/2.
        pooled_mean = (target_1 + target_2)/2.
        lc_pooled = pooled_est - 1.65*pooled_sd
        uc_pooled = pooled_est + 1.65*pooled_sd
        cov += (lc_pooled< pooled_mean)*(pooled_mean <uc_pooled)
        _pivot.append((pooled_est - pooled_mean)/pooled_sd)

        #lc = mle[1] - 1.65 * np.sqrt(inv_info[1,1])
        #uc = mle[1] + 1.65 * np.sqrt(inv_info[1,1])
        #cov += (lc< target_2)*(target_2 <uc)
        print("coverage so  far ", i+1, cov/float(i+1))
        #_pivot.append((mle[0]-target_1)/np.sqrt(inv_info[0,0]))

    plt.clf()
    ecdf_MLE = ECDF(ndist.cdf(np.asarray(_pivot)))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()

test_sum(nsim=500)