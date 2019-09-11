from __future__ import division, print_function
import numpy as np
from selection.randomized.lasso import lasso, split_lasso, selected_targets, full_targets, debiased_targets
from selection.B_mle.utils import sim_xy



def test_HOLP_boot_pivot(seedn, n=100, p=500, nval=100, alpha=2., rho=0.70, s=10, beta_type=1, snr=0.55,
                         split_proportion=0.5, B=5, nboot = 50, boot= True):

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

    sel_mle = np.zeros(B)
    alpha_target_carved = np.zeros(B)
    delta = 0.1
    for j in range(B):
        perturb_indices = np.zeros(n, np.bool)
        ps_n = int(split_proportion * n)
        perturb_indices[:ps_n] = True
        np.random.shuffle(perturb_indices)

        X_ind = X[~perturb_indices, :]
        Y_ind = y[~perturb_indices]
        lambda_val = 1./np.fabs(X_ind.T.dot(np.linalg.inv(X_ind.dot(X_ind.T) + delta*np.identity(ps_n))).dot(Y_ind))

        lasso_sol = split_lasso.gaussian(X,
                                         y,
                                         feature_weights=np.append(0.0, lambda_val[1:]),
                                         proportion=split_proportion)


        signs = lasso_sol.fit(perturb = perturb_indices)
        nonzero = signs != 0
        print("selected ", nonzero.sum(), nonzero[0])

        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(lasso_sol.loglike,
                                          lasso_sol._W,
                                          nonzero)

        mle = lasso_sol.selective_MLE(observed_target,
                                      cov_target,
                                      cov_target_score)[0]

        sel_mle[j] = mle[0]
        alpha_target_carved[j] = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))[0]

    sel_mle_boot = np.zeros((nboot, B))
    alpha_target_carved_boot = np.zeros((nboot, B))
    if boot == True:
        for b in range(nboot):
            print("iteration of boot ", b, B)
            boot_indices = np.random.choice(n, n, replace=True)
            X_boot = X[boot_indices, :]
            y_boot = y[boot_indices]
            for k in range(B):
                perturb_indices_boot = np.zeros(n, np.bool)
                perturb_indices_boot[:ps_n] = True
                np.random.shuffle(perturb_indices_boot)

                X_ind_boot = X_boot[~perturb_indices_boot, :]
                Y_ind_boot = y_boot[~perturb_indices_boot]
                lambda_val_boot = 1. / np.fabs(X_ind_boot.T.dot(np.linalg.inv(X_ind_boot.dot(X_ind_boot.T)
                                                                              + delta*np.identity(ps_n))).dot(Y_ind_boot))

                lasso_sol_boot = split_lasso.gaussian(X_boot,
                                                      y_boot,
                                                      feature_weights=np.append(0.0,
                                                                                lambda_val_boot[1:]),
                                                      proportion=split_proportion)

                signs = lasso_sol_boot.fit(perturb=perturb_indices_boot)
                nonzero = signs != 0
                print("selected ", nonzero.sum(), nonzero[0])

                (observed_target_boot,
                 cov_target_boot,
                 cov_target_score_boot,
                 alternatives_boot) = selected_targets(lasso_sol_boot.loglike,
                                                       lasso_sol_boot._W,
                                                       nonzero)

                mle_boot = lasso_sol_boot.selective_MLE(observed_target_boot,
                                                        cov_target_boot,
                                                        cov_target_score_boot)[0]

                sel_mle_boot[b, k] = mle_boot[0]
                alpha_target_carved_boot[b, k] = np.linalg.pinv(X_boot[:, nonzero]).dot(X.dot(beta))[0]

        boot_mean = np.mean(sel_mle_boot, axis=1)
        est_std = np.std(boot_mean)

        lc_pooled = np.mean(sel_mle) - 1.65 * est_std
        uc_pooled = np.mean(sel_mle) + 1.65 * est_std
        coverage = (lc_pooled < np.mean(alpha_target_carved)) * (np.mean(alpha_target_carved) < uc_pooled)
        print("check CI", lc_pooled, uc_pooled, np.mean(alpha_target_carved), coverage)

        pivot = (np.mean(sel_mle) - np.mean(alpha_target_carved)) / est_std
        return np.asscalar(pivot), 1 * coverage

    else:
        pivot = (np.mean(sel_mle) - np.mean(alpha_target_carved))
        return np.asscalar(pivot), 0.

def test_boot_pivot(seedn, n=100, p=500, nval=100, alpha=2., rho=0.70, s=10, beta_type=1, snr=0.55,
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

    lam_theory = sigma_ * 0.85 * np.mean(np.fabs(np.dot(X[:, 1:].T, np.random.standard_normal((n, 2000)))).max(0))

    sel_mle = np.zeros(B)
    alpha_target_carved = np.zeros(B)

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

        sel_mle[j] = mle
        alpha_target_carved[j] = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))[0]

    sel_mle_boot = np.zeros((nboot, B))
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

            mle_boot = lasso_sol_boot.selective_MLE(observed_target_uni_boot,
                                                    cov_target_uni_boot,
                                                    cov_target_score_uni_boot)[0]

            sel_mle_boot[b, k] = mle_boot
            alpha_target_carved_boot[b, k] = np.linalg.pinv(X_boot[:, nonzero]).dot(X.dot(beta))[0]

    boot_mean = np.mean(sel_mle_boot, axis=1)
    est_std = np.std(boot_mean)

    lc_pooled = np.mean(sel_mle) - 1.65 * est_std
    uc_pooled = np.mean(sel_mle) + 1.65 * est_std
    coverage = (lc_pooled < np.mean(alpha_target_carved)) * (np.mean(alpha_target_carved) < uc_pooled)
    print("check CI", lc_pooled, uc_pooled, np.mean(alpha_target_carved), coverage)

    pivot = (np.mean(sel_mle) - np.mean(alpha_target_carved))/est_std

    return np.asscalar(pivot), 1*coverage

import matplotlib.pyplot as plt
from statsmodels.distributions import ECDF
from scipy.stats import norm as ndist

def main(nsim):

    _pivot = []
    _cover = 0.
    for i in range(nsim):
        seed = i+200
        pivot_, coverage = test_boot_pivot(seedn= seed, n=100, p=500, nval=100, alpha=2.,
                                           rho=0.35, s=5, beta_type=2, snr=0.71, split_proportion=0.5,
                                           B=3, nboot = 100)
        _pivot.append(pivot_)
        _cover += coverage
        print("iteration completed ", i, _cover/(i+1.))

    plt.clf()
    MLE_pivot = ndist.cdf(np.asarray(_pivot))
    MLE_pivot = 2 * np.minimum(MLE_pivot, 1. - MLE_pivot)
    ecdf_MLE = ECDF(np.asarray(MLE_pivot))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='darkcyan', linestyle='-', linewidth=6)
    plt.plot(grid, grid, 'k--', linewidth=3)
    plt.savefig("plot_validity_pivot.pdf")

main(nsim =100)

def main_test(nsim):

    import seaborn as sns, pylab
    import scipy.stats as stats

    _pivot = []
    _cover = 0.
    for i in range(nsim):
        seed = i+100
        pivot_, coverage = test_HOLP_boot_pivot(seedn= seed, n=100, p=500, nval=100, alpha=2.,
                                           rho=0.70, s=5, beta_type=1, snr=0.75, split_proportion=0.5,
                                           B=3, nboot = 100, boot= False)
        _pivot.append(pivot_)
        _cover += coverage
        print("iteration completed ", i, _cover/(i+1.))

    sns.distplot(np.asarray(_pivot))
    plt.savefig("plot.pdf")

    stats.probplot(np.asarray(_pivot), dist="norm", plot=pylab)
    pylab.show()

#main_test(nsim=500)
