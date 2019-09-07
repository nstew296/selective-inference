from __future__ import division, print_function

import numpy as np
import nose.tools as nt

from scipy.stats import norm as ndist

import regreg.api as rr

from selection.randomized.lasso import (split_lasso,
                     selected_targets,
                     full_targets,
                     debiased_targets)
from selection.tests.instance import gaussian_instance


def test_split_lasso(n=100,
                     p=200,
                     signal_fac=3,
                     s=5,
                     sigma=3,
                     target='full',
                     rho=0.4,
                     proportion=0.67,
                     orthogonal=False,
                     ndraw=10000,
                     MLE=True,
                     burnin=5000):
    """
    Test data splitting lasso
    """

    inst, const = gaussian_instance, split_lasso.gaussian
    signal = np.sqrt(signal_fac * np.log(p))
    #signal = 15
    X, Y, beta = inst(n=n,
                      p=p,
                      signal=signal,
                      s=s,
                      equicorrelated=False,
                      rho=rho,
                      sigma=sigma,
                      random_signs=False)[:3]

    if orthogonal:
        X = np.linalg.svd(X, full_matrices=False)[0] * np.sqrt(n)
        Y = X.dot(beta) + np.random.standard_normal(n) * sigma

    n, p = X.shape

    sigma_ = np.std(Y)
    W = np.ones(X.shape[1]) * np.sqrt(np.log(p)) * sigma_
    #W = np.ones(X.shape[1])
    #W[0] = 0

    conv = const(X,
                 Y,
                 W,
                 proportion)

    signs = conv.fit()
    nonzero = signs != 0
    print("number selected ", nonzero.sum())

    if nonzero.sum() > 0:

        if target == 'full':
            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = full_targets(conv.loglike,
                                          conv._W,
                                          nonzero,
                                          dispersion=sigma ** 2)
        elif target == 'selected':
            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(conv.loglike,
                                              conv._W,
                                              nonzero)  # ,
            # dispersion=sigma**2)

        elif target == 'debiased':
            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = debiased_targets(conv.loglike,
                                              conv._W,
                                              nonzero,
                                              penalty=conv.penalty,
                                              dispersion=sigma ** 2)

        _, pval, intervals = conv.summary(observed_target,
                                          cov_target,
                                          cov_target_score,
                                          alternatives,
                                          ndraw=ndraw,
                                          burnin=burnin,
                                          compute_intervals=False)

        final_estimator, observed_info_mean,  _, pval, MLE_intervals = conv.selective_MLE(
            observed_target,
            cov_target,
            cov_target_score)[:5]
        print("check intervals ", MLE_intervals, nonzero[0])

        if target == 'selected':
            true_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
        else:
            true_target = beta[nonzero]

        coverage = (true_target > MLE_intervals[:, 0]) * (true_target < MLE_intervals[:, 1])

        MLE_pivot = ndist.cdf((final_estimator - true_target) /
                              np.sqrt(np.diag(observed_info_mean)))
        MLE_pivot = 2 * np.minimum(MLE_pivot, 1. - MLE_pivot)
        print("difference ", MLE_pivot - pval)

        if MLE:
            return MLE_pivot[true_target == 0], MLE_pivot[true_target != 0], coverage[1:], MLE_pivot
        else:
            return pval[true_target == 0], pval[true_target != 0], [], []
    else:
        return [], [], [], []


def test_all_targets(n=100, p=20, signal_fac=1.5, s=5, sigma=3, rho=0.4):
    for target in ['full', 'selected', 'debiased']:
        test_split_lasso(n=n,
                         p=p,
                         signal_fac=signal_fac,
                         s=s,
                         sigma=sigma,
                         rho=rho,
                         target=target)


def main(nsim=100, n=100, p=500, target='selected', sigma=5., s=5):
    import matplotlib.pyplot as plt
    P0, PA, cover, Pivot = [], [], [], []
    from statsmodels.distributions import ECDF

    for i in range(nsim):
        p0, pA, cover_, MLE_pivot = test_split_lasso(n=n, p=p, target=target, sigma=sigma, s=s)
        print(len(p0), len(pA))
        if not (len(pA) < s and target == 'selected'):
            P0.extend(p0)
            PA.extend(pA)
            cover.extend(cover_)
            Pivot.extend(MLE_pivot)

        print("coverage so far ", np.mean(cover))
        P0_clean = np.array(P0)

        P0_clean = P0_clean[P0_clean > 1.e-5]  #
        print(np.mean(P0_clean), np.std(P0_clean), np.mean(np.array(PA) < 0.05), np.sum(np.array(PA) < 0.05) / (i + 1),
              np.mean(np.array(P0) < 0.05), np.mean(P0_clean < 0.05), np.mean(np.array(P0) < 1e-5),
              'null pvalue + power + failure')

    #     if i % 3 == 0 and i > 0:
    #         U = np.linspace(0, 1, 101)
    #         plt.clf()
    #         if len(P0_clean) > 0:
    #             plt.plot(U, ECDF(P0_clean)(U))
    #         if len(PA) > 0:
    #             plt.plot(U, ECDF(PA)(U), 'r')
    #         plt.plot([0, 1], [0, 1], 'k--')
    #         plt.savefig("plot.pdf")
    # plt.show()

        if i % 3 == 0 and i > 0:
            plt.clf()
            ecdf_MLE = ECDF(np.asarray(Pivot))
            grid = np.linspace(0, 1, 101)
            plt.plot(grid, ecdf_MLE(grid), c='darkcyan', linestyle='-', linewidth=6)
            plt.plot(grid, grid, 'k--', linewidth=3)
            plt.savefig("plot.pdf")
    plt.show()

main(nsim = 500)