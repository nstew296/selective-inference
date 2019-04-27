from __future__ import division, print_function
import numpy as np

from selection.randomized.lasso import lasso, selected_targets, full_targets, debiased_targets
from selection.multiple_splits.utils import sim_xy, glmnet_lasso_cv1se, glmnet_lasso_cvmin, glmnet_lasso


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

            estimate, _, _, _, _, _, mle_comp, var_comp = lasso_sol.selective_MLE(observed_target_uni,
                                                                                  cov_target_uni,
                                                                                  cov_target_score_uni,
                                                                                  alternatives)

            return np.asscalar(mle_comp), np.asscalar(var_comp), np.asscalar(observed_target_uni),\
                   np.asscalar(cov_target_uni), (sigma) * np.linalg.pinv(X[:, nonzero])[0,:], estimate,\
                   np.asscalar(beta_target[0])

def test_mse(n=200, p=1000, nval=200, alpha= 2., rho=0.35, s=10, beta_type=0, snr=0.20, randomizer_scale=1., nsim=100, B=10):

    mse_marginal = 0.
    mse_full = 0.
    bias_marginal = 0.
    bias_full = 0.
    for i in range(nsim):
        X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, alpha=alpha, rho=rho, s=s, beta_type=beta_type,
                                                snr=snr)
        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
        y = y - y.mean()

        mle_comp = np.zeros(B)
        ls_comp = np.zeros(B)
        Sigma_full_0 = np.zeros((B,B))
        mle_marginal = np.zeros(B)
        covar = np.zeros((n, B))
        for j in range(B):
            mle_comp[j], _, ls_comp[j], Sigma_full_0[j,j], covar[:,j], mle_marginal[j], _ = test_lasso_estimate(X, y, sigma, beta, randomizer_scale)
        #print("check ", mle_comp, ls_comp)
        for k in range(B-1):
            l = k + 1
            for m in range(B-k-1):
                (Sigma_full_0[k, :])[l] = (covar[:,k]).T.dot(covar[:,l])
                l += 1

        Sigma_full = Sigma_full_0 + Sigma_full_0.T - np.diag(Sigma_full_0.diagonal())

        mle_full = ls_comp + Sigma_full.dot(mle_comp)
        mse_full += (np.mean(mle_full)-alpha)**2.

        mse_marginal += (np.mean(mle_marginal) - alpha) ** 2.

        bias_full += (np.mean(mle_full)-alpha)
        bias_marginal += (np.mean(mle_marginal) - alpha)
        print("bias so far ", bias_full / float(i + 1.), bias_marginal / float(i + 1.))
        print("mse so far ", mse_full / float(i + 1.), mse_marginal / float(i + 1.))

    print("var in target so far ", (mse_full/nsim- ((bias_full/nsim)**2.)), (mse_marginal/nsim- ((bias_marginal/nsim)**2.)))

test_mse(randomizer_scale=1., nsim=100, B=5)