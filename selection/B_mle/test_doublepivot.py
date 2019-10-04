from __future__ import division, print_function
import numpy as np
from selection.randomized.lasso import lasso, split_lasso, selected_targets, full_targets, debiased_targets
from selection.B_mle.utils import sim_xyd
from selection.algorithms.lasso import lasso_full

def test_validity_pivot(seedn, n=100, p=500, nval=100, alpha=2., rho=0.70,
                        s_y=10, beta_type_y=1, snr_y=0.55,
                        s_d=10, beta_type_d=1, snr_d=0.55,
                        split_proportion=0.5, B=5):

    x, y, D, _, _, Sigma, beta, gamma, sigma_y, sigma_d,  _ = sim_xyd(n=n,
                                                                      p=p,
                                                                      nval=nval,
                                                                      seedn=seedn,
                                                                      alpha=alpha,
                                                                      rho=rho,
                                                                      s_y=s_y,
                                                                      beta_type_y=beta_type_y,
                                                                      snr_y=snr_y,
                                                                      s_d=s_d,
                                                                      beta_type_d=beta_type_d,
                                                                      snr_d=snr_d)

    true_mean = np.hstack((D.reshape((n, 1)), x)).dot(beta)
    x -= x.mean(0)[None, :]
    y = y - y.mean()
    d = D - D.mean()

    sigma_d_ = np.std(d)
    sigma_y_ = np.std(y)

    theory = np.mean(np.fabs(np.dot(x.T, np.random.standard_normal((n, 2000)))).max(0))
    lam_theory_y = sigma_y_ * 0.80 * theory
    lam_theory_d = sigma_d_ * 0.70 * theory

    sel_mle = np.zeros(B)
    alpha_target_carved = np.zeros(B)

    for j in range(B):
        lasso_d = lasso_full.gaussian(x, d, lam_theory_d * np.ones(p - 1))
        signs_d = lasso_d.fit()
        nonzero_d = signs_d != 0

        X = np.hstack((d.reshape((n,1)), x))
        lasso_sol = split_lasso.gaussian(X,
                                         y,
                                         feature_weights= np.append(0.,lam_theory_y*np.ones(p-1)),
                                         proportion=split_proportion)

        signs_y = lasso_sol.fit()
        nonzero_y = signs_y != 0
        nonzero = np.append(True, np.logical_or(nonzero_y[1:], nonzero_d))

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
        alpha_target_carved[j] = np.linalg.pinv(X[:,nonzero]).dot(true_mean)[0]
        print("SEE: mle and observed target", mle[0], observed_target[0])

    print("check ", np.mean(sel_mle), np.mean(alpha_target_carved), observed_target[0])
    pivot = np.mean(sel_mle) - np.mean(alpha_target_carved)

    return np.asscalar(pivot)

def test_boot_pivot(seedn, n=100, p=500, nval=100, alpha=2., rho=0.70,
                    s_y=10, beta_type_y=1, snr_y=0.55,
                    s_d=10, beta_type_d=1, snr_d=0.55,
                    split_proportion=0.5, B=5, nboot = 50,
                    adaptive = True, lam_frac_d = 5., lam_frac_y=8. ):

    x, y, D, _, _, Sigma, beta, gamma, sigma_y, sigma_d, _ = sim_xyd(n=n,
                                                                     p=p,
                                                                     nval=nval,
                                                                     seedn=seedn,
                                                                     alpha=alpha,
                                                                     rho=rho,
                                                                     s_y=s_y,
                                                                     beta_type_y=beta_type_y,
                                                                     snr_y=snr_y,
                                                                     s_d=s_d,
                                                                     beta_type_d=beta_type_d,
                                                                     snr_d=snr_d)

    true_mean = np.hstack((D.reshape((n, 1)), x)).dot(beta)
    x -= x.mean(0)[None, :]
    y = y - y.mean()
    d = D - D.mean()

    sigma_d_ = np.std(d)
    sigma_y_ = np.std(y)

    sel_mle = np.zeros(B)
    alpha_target_carved = np.zeros(B)
    sel_mle_boot = np.zeros((nboot, B))

    if adaptive == False:
        theory = np.mean(np.fabs(np.dot(x.T, np.random.standard_normal((n, 2000)))).max(0))
        lam_theory_y = sigma_y_ * 0.80 * theory
        lam_theory_d = sigma_d_ * 0.80 * theory

        for j in range(B):
            lasso_d = lasso_full.gaussian(x, d, lam_theory_d * np.ones(p - 1))
            signs_d = lasso_d.fit()
            nonzero_d = signs_d != 0

            X = np.hstack((d.reshape((n, 1)), x))
            lasso_sol = split_lasso.gaussian(X,
                                             y,
                                             feature_weights= np.append(0., lam_theory_y * np.ones(p - 1)),
                                             proportion=split_proportion)

            signs_y = lasso_sol.fit()
            nonzero_y = signs_y != 0
            nonzero = np.append(True, np.logical_or(nonzero_y[1:], nonzero_d))
            print("selected ", nonzero_y.sum(), nonzero_d.sum(), nonzero.sum())

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
            alpha_target_carved[j] = np.linalg.pinv(X[:,nonzero]).dot(true_mean)[0]

        for b in range(nboot):
            print("iteration of boot ", b, B)
            boot_indices = np.random.choice(n, n, replace=True)
            x_boot = x[boot_indices, :]
            y_boot = y[boot_indices]
            d_boot = d[boot_indices]
            X_boot = np.hstack((d_boot.reshape((n, 1)), x_boot))

            for k in range(B):
                lasso_d_boot = lasso_full.gaussian(x_boot, d_boot, lam_theory_d * np.ones(p - 1))
                signs_d_boot = lasso_d_boot.fit()
                nonzero_d_boot = signs_d_boot != 0

                lasso_sol_boot = split_lasso.gaussian(X_boot,
                                                      y_boot,
                                                      feature_weights=np.append(0., lam_theory_y * np.ones(p - 1)),
                                                      proportion=split_proportion)

                signs_y_boot = lasso_sol_boot.fit()
                nonzero_y_boot = signs_y_boot != 0
                nonzero = np.append(True, np.logical_or(nonzero_y_boot[1:], nonzero_d_boot))

                (observed_target_boot,
                 cov_target_boot,
                 cov_target_score_boot,
                 alternatives) = selected_targets(lasso_sol_boot.loglike,
                                                  lasso_sol_boot._W,
                                                  nonzero)

                mle_boot = lasso_sol_boot.selective_MLE(observed_target_boot,
                                                        cov_target_boot,
                                                        cov_target_score_boot)[0]

                sel_mle_boot[b, k] = mle_boot[0]

        boot_mean = np.mean(sel_mle_boot, axis=1)
        est_std = np.std(boot_mean)

        lc_pooled = np.mean(sel_mle) - 1.65 * est_std
        uc_pooled = np.mean(sel_mle) + 1.65 * est_std
        coverage = (lc_pooled < np.mean(alpha_target_carved)) * (np.mean(alpha_target_carved) < uc_pooled)
        coverage_true = (lc_pooled < alpha) * (alpha < uc_pooled)
        print("check CI", lc_pooled, uc_pooled, np.mean(alpha_target_carved), coverage, coverage_true)

        pivot = (np.mean(sel_mle) - np.mean(alpha_target_carved)) / est_std

    else:
        delta = 0.1
        lambda_val_d = lam_frac_d / np.fabs(x.T.dot(np.linalg.inv(x.dot(x.T) + delta * np.identity(n))).dot(d))

        perturb_indices = np.zeros(n, np.bool)
        ps_n = int(split_proportion * n)
        perturb_indices[:ps_n] = True
        np.random.shuffle(perturb_indices)

        X = np.hstack((d.reshape((n, 1)), x))
        X_ind = X[~perturb_indices, :]
        Y_ind = y[~perturb_indices]
        lambda_val = lam_frac_y / np.fabs(X_ind[:, 1:].T.dot(np.linalg.inv(X_ind[:, 1:].dot(X_ind[:, 1:].T) + delta * np.identity(ps_n))).dot(Y_ind))

        for j in range(B):
            lasso_d = lasso_full.gaussian(x, d, lambda_val_d * np.ones(p - 1))
            signs_d = lasso_d.fit()
            nonzero_d = signs_d != 0

            lasso_sol = split_lasso.gaussian(X,
                                             y,
                                             feature_weights=np.append(0., lambda_val),
                                             proportion=split_proportion)

            signs_y = lasso_sol.fit(perturb=perturb_indices)
            nonzero_y = signs_y != 0
            nonzero = np.append(True, np.logical_or(nonzero_y[1:], nonzero_d))
            print("selected ", nonzero_y.sum(), nonzero_d.sum(), nonzero.sum())

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
            alpha_target_carved[j] = np.linalg.pinv(X[:, nonzero]).dot(true_mean)[0]

        for b in range(nboot):
            print("iteration of boot ", b, B)
            boot_indices = np.random.choice(n, n, replace=True)
            x_boot = x[boot_indices, :]
            y_boot = y[boot_indices]
            d_boot = d[boot_indices]
            X_boot = np.hstack((d_boot.reshape((n, 1)), x_boot))

            lambda_val_d_boot = lam_frac_d / np.fabs(x_boot.T.dot(np.linalg.inv(x_boot.dot(x_boot.T) + delta * np.identity(n))).dot(d_boot))

            perturb_indices_boot = np.zeros(n, np.bool)
            perturb_indices_boot[:ps_n] = True
            np.random.shuffle(perturb_indices_boot)

            X_ind_boot = X_boot[~perturb_indices_boot, :]
            Y_ind_boot = y_boot[~perturb_indices_boot]
            lambda_val_boot = lam_frac_y / np.fabs(
                X_ind_boot[:, 1:].T.dot(np.linalg.inv(X_ind_boot[:, 1:].dot(X_ind_boot[:, 1:].T) + delta * np.identity(ps_n))).dot(
                    Y_ind_boot))

            for k in range(B):
                lasso_d_boot = lasso_full.gaussian(x_boot, d_boot, lambda_val_d_boot)
                signs_d_boot = lasso_d_boot.fit()
                nonzero_d_boot = signs_d_boot != 0

                lasso_sol_boot = split_lasso.gaussian(X_boot,
                                                      y_boot,
                                                      feature_weights=np.append(0., lambda_val_boot),
                                                      proportion=split_proportion)

                signs_y_boot = lasso_sol_boot.fit(perturb=perturb_indices_boot)
                nonzero_y_boot = signs_y_boot != 0
                nonzero = np.append(True, np.logical_or(nonzero_y_boot[1:], nonzero_d_boot))

                (observed_target_boot,
                 cov_target_boot,
                 cov_target_score_boot,
                 alternatives) = selected_targets(lasso_sol_boot.loglike,
                                                  lasso_sol_boot._W,
                                                  nonzero)

                mle_boot = lasso_sol_boot.selective_MLE(observed_target_boot,
                                                        cov_target_boot,
                                                        cov_target_score_boot)[0]

                sel_mle_boot[b, k] = mle_boot[0]

        boot_mean = np.mean(sel_mle_boot, axis=1)
        est_std = np.std(boot_mean)

        lc_pooled = np.mean(sel_mle) - 1.65 * est_std
        uc_pooled = np.mean(sel_mle) + 1.65 * est_std
        coverage = (lc_pooled < np.mean(alpha_target_carved)) * (np.mean(alpha_target_carved) < uc_pooled)
        coverage_true = (lc_pooled < alpha) * (alpha < uc_pooled)
        print("check CI", lc_pooled, uc_pooled, np.mean(alpha_target_carved), coverage, coverage_true)

        pivot = (np.mean(sel_mle) - np.mean(alpha_target_carved)) / est_std

    return np.asscalar(pivot), 1*coverage, 1*coverage_true

import matplotlib.pyplot as plt
from statsmodels.distributions import ECDF
from scipy.stats import norm as ndist

def main(nsim):

    _pivot = []
    _cover = 0.
    _cover_true = 0.
    for i in range(nsim):
        seed = i+501
        pivot_, coverage, coverage_true = test_boot_pivot(seedn= seed, n=120, p=500, nval=120, alpha=2., rho=0.90,
                                                          s_y=5, beta_type_y=2, snr_y=0.90,
                                                          s_d=5, beta_type_d=3, snr_d=0.90,
                                                          split_proportion=0.5,
                                                          B=3, nboot = 100, adaptive=True)
        _pivot.append(pivot_)
        _cover += coverage
        _cover_true += coverage_true
        print("iteration completed ", i, _cover/(i+1.), _cover_true/(i+1.))

    plt.clf()
    MLE_pivot = ndist.cdf(np.asarray(_pivot))
    MLE_pivot = 2 * np.minimum(MLE_pivot, 1. - MLE_pivot)
    ecdf_MLE = ECDF(np.asarray(MLE_pivot))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='darkcyan', linestyle='-', linewidth=6)
    plt.plot(grid, grid, 'k--', linewidth=3)
    plt.savefig("plot_validity_pivot.pdf")

main(nsim =100)

def main_validity(nsim):
    import seaborn as sns
    import matplotlib.pyplot as plt
    _pivot = []
    for i in range(nsim):
        seed = i+101
        pivot_ = test_validity_pivot(seedn=seed, n=120, p=500, nval=120, alpha=2., rho=0.70,
                                      s_y=5, beta_type_y=2, snr_y=0.90,
                                      s_d=5, beta_type_d=2, snr_d=0.90,
                                      split_proportion=0.5, B=1)
        _pivot.append(pivot_)
        print("iteration completed ", i+1)
    sns.distplot(np.asarray(_pivot))
    plt.savefig("plot_validity_pivot.pdf")
    plt.show()

#main_validity(nsim =500)
