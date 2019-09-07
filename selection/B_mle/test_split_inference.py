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

        mle_carved[j] = lasso_sol.selective_MLE(observed_target_uni,
                                                cov_target_uni,
                                                cov_target_score_uni)[0]
        alpha_target_carved[j] = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))[0]

        inf_idx = ~lasso_sol._selection_idx
        y_inf = y[inf_idx]
        X_inf = X[inf_idx, :]
        mle_split[j] = np.linalg.pinv(X_inf[:, nonzero]).dot(y_inf)[0]

    mse_carved += (np.mean(mle_carved) - alpha)**2
    mse_split += (np.mean(mle_split) - alpha)**2

    mle_carved_boot = np.zeros((nboot, B))
    mle_split_boot = np.zeros((nboot, B))
    alpha_target_carved_boot = np.zeros((nboot, B))

    for b in range(nboot):
        print("iteration of boot ", b, B)
        boot_indices = np.random.choice(n, n, replace=True)
        X_boot = X[boot_indices, :]
        y_boot = y[boot_indices]
        for k in range(B):
            lasso_sol_boot = split_lasso.gaussian(X_boot,
                                                  y_boot,
                                                  feature_weights=np.append(0.0, lam_theory * np.ones(p - 1)),
                                                  proportion=split_proportion)

            signs = lasso_sol_boot.fit()
            nonzero = signs != 0

            (observed_target_boot,
             cov_target_boot,
             cov_target_score_boot,
             alternatives_boot) = selected_targets(lasso_sol_boot.loglike,
                                                   lasso_sol_boot._W,
                                                   nonzero)

            observed_target_uni_boot = (observed_target_boot[0]).reshape((1,))
            cov_target_uni_boot = (np.diag(cov_target_boot)[0]).reshape((1, 1))
            cov_target_score_uni_boot = cov_target_score_boot[0, :].reshape((1, p))

            mle_carved_boot[b, k] = lasso_sol_boot.selective_MLE(observed_target_uni_boot,
                                                                 cov_target_uni_boot,
                                                                 cov_target_score_uni_boot)[0]
            alpha_target_carved_boot[b, k] = np.linalg.pinv(X_boot[:, nonzero]).dot(X.dot(beta))[0]

            inf_idx_boot = ~lasso_sol_boot._selection_idx
            y_inf_boot = y_boot[inf_idx_boot]
            X_inf_boot = X_boot[inf_idx_boot, :]

            mle_split_boot[b, k] = np.linalg.pinv(X_inf_boot[:, nonzero]).dot(y_inf_boot)[0]

    boot_mean_carved = np.mean(mle_carved_boot, axis=1)
    est_std_carved = np.std(boot_mean_carved)

    boot_mean_split = np.mean(mle_split_boot, axis=1)
    est_std_split = np.std(boot_mean_split)

    return mse_carved, mse_split, (2*1.65) * est_std_carved, (2*1.65) * est_std_split

def main(nsim):

    _mse_carved = 0.
    _mse_split = 0.
    _length_carved = 0.
    _length_split = 0.

    for i in range(nsim):
        seed = i+100
        mse_carved, mse_split, length_carved, length_split = test_compare_split_carve(seedn= seed, n=100, p=500, nval=100, alpha=2.,
                                                                                      rho=0.70, s=5, beta_type=1, snr=0.70, split_proportion=0.5,
                                                                                      B=5, nboot = 50)

        _mse_carved += mse_carved
        _mse_split += mse_split
        _length_carved += length_carved
        _length_split += length_split

        print("iteration completed ", i, _mse_carved/(i+1.), _mse_split/(i+1.), _length_carved/(i+1.), _length_split/(i+1.))

main(nsim=1000)