from __future__ import division, print_function
import numpy as np
from selection.randomized.lasso import lasso, split_lasso, selected_targets, full_targets, debiased_targets
from selection.B_mle.utils import sim_xy


def test_compare_split_carve(seedn, n=100, p=500, nval=100, alpha=2., rho=0.70, s=10, beta_type=1, snr=0.55,
                             split_proportion=0.5, B=5, nboot = 50):

    X, y, _, _, Sigma, beta, sigma, _ = sim_xy(n=n,
                                               p=p,
                                               nval=nval,
                                               seedn=seedn,
                                               alpha=alpha,
                                               rho=rho,
                                               s=s,
                                               beta_type=beta_type,
                                               snr=snr)
    X -= X.mean(0)[None, :]
    y = y - y.mean()
    scaling = X.std(0)[None, :] * np.sqrt(n / (n - 1.))
    X /= scaling

    sigma_ = np.std(y)

    lam_theory = sigma_ * 0.90 * np.mean(np.fabs(np.dot(X[:, 1:].T, np.random.standard_normal((n, 2000)))).max(0))

    mle_carved = np.zeros(B)
    mle_split = np.zeros(B)
    v_split = np.zeros((B, n))

    alpha_target_carved = np.zeros(B)

    mse_carved = 0.
    mse_split = 0.

    for j in range(B):

        lasso_sol = split_lasso.gaussian(X,
                                         y,
                                         feature_weights=np.append(0.0, lam_theory*np.ones(p-1)),
                                         proportion=split_proportion)

        signs = lasso_sol.fit()
        nonzero = signs != 0
        print("selected ", nonzero.sum(), nonzero[0])

        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(lasso_sol.loglike,
                                          lasso_sol._W,
                                          nonzero)

        observed_target_uni = (observed_target[0]).reshape((1,))
        cov_target_uni = (np.diag(cov_target)[0]).reshape((1, 1))
        cov_target_score_uni = cov_target_score[0, :].reshape((1, p))

        mle = lasso_sol.selective_MLE(observed_target_uni,
                                      cov_target_uni,
                                      cov_target_score_uni)[0]

        mle_carved[j] = mle
        alpha_target_carved[j] = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))[0]

        inf_idx = ~lasso_sol._selection_idx
        y_inf = y[inf_idx]
        X_inf = X[inf_idx, :]
        mle_split[j] = np.linalg.pinv(X_inf[:, nonzero]).dot(y_inf)[0]
        v_split[j, :] = 1 * lasso_sol._selection_idx

    mse_carved += (np.mean(mle_carved) -alpha)**2
    mse_split += (np.mean(mle_split) - alpha)**2

    return mse_carved, mse_split

def main(nsim):

    _mse_carved = 0.
    _mse_split = 0.
    for i in range(nsim):
        seed = i+100
        mse_carved, mse_split = test_compare_split_carve(seedn= seed, n=100, p=500, nval=100, alpha=2.,
                                                         rho=0.70, s=5, beta_type=1, snr=0.70, split_proportion=0.5,
                                                         B=2, nboot = 50)

        _mse_carved += mse_carved
        _mse_split += mse_split
        print("iteration completed ", i, _mse_carved/(i+1.), _mse_split/(i+1.))

main(nsim=1000)