from __future__ import division, print_function
import numpy as np
from selection.randomized.lasso import lasso, split_lasso, selected_targets, full_targets, debiased_targets
from selection.B_mle.utils import sim_xy

def test_boot_splitpivot(seedn, n=100, p=500, nval=100, alpha=2., rho=0.70, s=10, beta_type=1, snr=0.55,
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

    lam_theory = sigma_ * 0.9 * np.mean(np.fabs(np.dot(X[:, 1:].T, np.random.standard_normal((n, 2000)))).max(0))

    mle_split = np.zeros(B)
    alpha_target_split = np.zeros(B)

    for j in range(B):

        lasso_sol = split_lasso.gaussian(X,
                                         y,
                                         feature_weights=np.append(0.0, lam_theory*np.ones(p-1)),
                                         proportion=split_proportion)

        signs = lasso_sol.fit()
        nonzero = signs != 0
        print("selected ", nonzero.sum(), nonzero[0])

        inf_idx = ~lasso_sol._selection_idx
        y_inf = y[inf_idx]
        X_inf = X[inf_idx, :]
        mle_split[j] = np.linalg.pinv(X_inf[:, nonzero]).dot(y_inf)[0]
        alpha_target_split[j] = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))[0]

    mle_split_boot = np.zeros((nboot, B))
    alpha_target_split_boot = np.zeros((nboot, B))

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

            inf_idx_boot = ~lasso_sol_boot._selection_idx
            y_inf_boot = y_boot[inf_idx_boot]
            X_inf_boot = X_boot[inf_idx_boot, :]

            mle_split_boot[b, k] = np.linalg.pinv(X_inf_boot[:, nonzero]).dot(y_inf_boot)[0]
            alpha_target_split_boot[b, k] = np.linalg.pinv(X_boot[:, nonzero]).dot(X.dot(beta))[0]

    boot_mean = np.mean(mle_split_boot, axis=1)
    est_std = np.std(boot_mean)

    lc_pooled = np.mean(mle_split) - 1.65 * est_std
    uc_pooled = np.mean(mle_split) + 1.65 * est_std
    coverage = (lc_pooled < np.mean(alpha_target_split)) * (np.mean(alpha_target_split) < uc_pooled)
    print("check CI", lc_pooled, uc_pooled, np.mean(alpha_target_split), coverage)

    pivot = (np.mean(mle_split) - np.mean(alpha_target_split))/est_std

    return np.asscalar(pivot), 1*coverage

import matplotlib.pyplot as plt
from statsmodels.distributions import ECDF
from scipy.stats import norm as ndist

def main(nsim):

    _pivot = []
    _cover = 0.
    for i in range(nsim):
        seed = i+100
        pivot_, coverage = test_boot_splitpivot(seedn= seed, n=100, p=500, nval=100, alpha=2.,
                                                rho=0.70, s=5, beta_type=1, snr=0.75, split_proportion=0.5,
                                                B=5, nboot = 100)
        _pivot.append(pivot_)
        _cover += coverage
        print("iteration completed ", i, _cover/(i+1.))

    plt.clf()
    split_pivot = ndist.cdf(np.asarray(_pivot))
    split_pivot = 2 * np.minimum(split_pivot, 1. - split_pivot)
    ecdf_split = ECDF(np.asarray(split_pivot))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_split(grid), c='darkcyan', linestyle='-', linewidth=6)
    plt.plot(grid, grid, 'k--', linewidth=3)
    plt.savefig("plot_validity_splitpivot.pdf")

main(nsim=70)
