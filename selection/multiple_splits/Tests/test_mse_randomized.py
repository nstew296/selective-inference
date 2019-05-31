from __future__ import division, print_function
import numpy as np, os
import pandas as pd

from selection.randomized.lasso import lasso, selected_targets, full_targets, debiased_targets
from selection.multiple_splits.utils import sim_xy, glmnet_lasso_cv1se, glmnet_lasso_cvmin, glmnet_lasso, \
    glmnet_lasso_cv
from selection.algorithms.lasso import lasso_full


def test_mse(n=200, p=1000, nval=200, alpha=2., rho=0.70, s=10, beta_type=1, snr=1.0,
             randomizer_scale=1., split_fraction=0.67,
             nsim=100, B=100):

    bias_tar_randomized = 0.
    bias_randomized = 0.
    mse_randomized = 0.
    fourth_moment_randomized = 0.

    bias_tar_split = 0.
    bias_split = 0.
    mse_split = 0.
    fourth_moment_split = 0.

    for i in range(nsim):

        X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, alpha=alpha, rho=rho, s=s, beta_type=beta_type,
                                                snr=snr)
        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
        y = y - y.mean()

        dispersion = None
        sigma_ = np.std(y)
        lam_theory = sigma_ * 0.85 *  np.mean(np.fabs(np.dot(X[:, 1:].T, np.random.standard_normal((n, 2000)))).max(0))

        alpha_target_randomized = np.zeros(B)
        sel_mle = np.zeros(B)

        alpha_target_split = np.zeros(B)
        est_split = np.zeros(B)

        for j in range(B):
            lasso_sol = lasso.gaussian(X,
                                       y,
                                       feature_weights=np.append(0.001, np.ones(p - 1) * lam_theory),
                                       randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)
            signs = lasso_sol.fit()
            nonzero = signs != 0

            if nonzero[0] == 1 and nonzero.sum()<30:
                alpha_target_randomized[j] = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))[0]
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

                mle, _, _, _, _, _, _, _ = lasso_sol.selective_MLE(observed_target_uni,
                                                                   cov_target_uni,
                                                                   cov_target_score_uni,
                                                                   alternatives)

                sel_mle[j] = mle

            subsample_size = int(split_fraction * n)
            sel_idx = np.zeros(n, np.bool)
            sel_idx[:subsample_size] = 1
            np.random.shuffle(sel_idx)
            inf_idx = ~sel_idx
            y_inf = y[inf_idx]
            X_inf = X[inf_idx, :]
            y_sel = y[sel_idx]
            X_sel = X[sel_idx, :]

            lam_theory_split = sigma_ * 0.85* np.mean(
                np.fabs(np.dot(X_sel[:, 1:].T, np.random.standard_normal((subsample_size, 2000)))).max(0))
            lasso_split = lasso_full.gaussian(X_sel, y_sel, np.append(0.001, np.ones(p - 1) * lam_theory_split))
            lasso_soln = lasso_split.fit()
            active_LASSO = (lasso_soln != 0)
            nactive_LASSO = active_LASSO.sum()

            if active_LASSO[0] == 1 and nactive_LASSO<30:
                alpha_target_split[j] = np.linalg.pinv(X_inf[:, active_LASSO]).dot(X_inf.dot(beta))[0]
                est_split[j] = np.linalg.pinv(X_inf[:, active_LASSO]).dot(y_inf)[0]

        alpha_target_randomized = alpha_target_randomized[alpha_target_randomized != 0]
        sel_mle = sel_mle[sel_mle != 0]

        avg_target_randomized = np.mean(alpha_target_randomized)
        bias_tar_randomized += (avg_target_randomized - alpha)
        bias_randomized += (np.mean(sel_mle) - alpha)
        mse_randomized += ((np.mean(sel_mle) - alpha) ** 2)
        fourth_moment_randomized += ((np.mean(sel_mle) - alpha) ** 4)

        alpha_target_split = alpha_target_split[alpha_target_split != 0]
        est_split = est_split[est_split != 0]

        avg_target_split = np.mean(alpha_target_split)
        bias_tar_split += (avg_target_split - alpha)
        bias_split += (np.mean(est_split) - alpha)
        mse_split += ((np.mean(est_split) - alpha) ** 2)
        fourth_moment_split += ((np.mean(est_split) - alpha) ** 4)

        print("iteration completed ", i+1, B)
        print("theoretical sigma ", sigma, sigma_, (sigma ** 2) / (n * Sigma[0, 0]))

    stderr_bias_randomized = np.sqrt(mse_randomized/float(nsim **2))
    stderr_mse_randomized = np.sqrt((fourth_moment_randomized - ((mse_randomized/float(nsim))**2.))/ float(nsim ** 2))

    stderr_bias_split = np.sqrt(mse_split / float(nsim ** 2))
    stderr_mse_split = np.sqrt((fourth_moment_split - ((mse_split / float(nsim)) ** 2.)) / float(nsim ** 2))

    bias_tar_randomized /= float(nsim)
    bias_randomized /= float(nsim)
    mse_randomized /= float(nsim)

    bias_tar_split /= float(nsim)
    bias_split /= float(nsim)
    mse_split /= float(nsim)

    return np.vstack((bias_tar_randomized,
                      bias_tar_split,
                      bias_randomized,
                      bias_split,
                      mse_randomized,
                      mse_split,
                      stderr_bias_randomized,
                      stderr_bias_split,
                      stderr_mse_randomized,
                      stderr_mse_split))

def output_file(n=200, p=1000, nval=200, alpha= 2., rho=0.35, s=10, beta_type=0, snr=0.71,
                randomizer_scale=1., split_fraction=0.67, nsim=10, Bval=np.array([1,2,3,5,10,20,25]),
                outpath= None):

    df_mse = pd.DataFrame()
    for B_agg in Bval:
        output = test_mse(n=n,
                          p=p,
                          nval=nval,
                          alpha=alpha,
                          rho=rho,
                          s=s,
                          beta_type= beta_type,
                          snr=snr,
                          randomizer_scale=randomizer_scale,
                          split_fraction=split_fraction,
                          nsim=nsim,
                          B=B_agg)

        df_mse_B = pd.DataFrame(data=output.reshape((1, 10)),
                                columns=['bias_tar_randomized', 'bias_tar_split',
                                         'bias_randomized', 'bias_split',
                                         'mse_randomized', 'mse_split',
                                         'std_bias_randomized', 'std_bias_split',
                                         'std_mse_randomized', 'std_mse_split'])

        df_mse = df_mse.append(df_mse_B, ignore_index=True)

    df_mse['n'] = n
    df_mse['p'] = p
    df_mse['s'] = s
    df_mse['rho'] = rho
    df_mse['beta-type'] = beta_type
    df_mse['snr'] = snr
    df_mse['B'] = pd.Series(np.asarray(Bval))
    print("check final ", df_mse)

    if outpath is None:
        outpath = os.path.dirname(__file__)

    outfile_inf_html = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_mse_normal_" + str(beta_type) +  "_rho_" + str(rho) + ".html")
    outfile_inf_csv = os.path.join(outpath,"dims_" + str(n) + "_" + str(p) + "_mse_normal_" + str(beta_type) + "_rho_" + str(rho) + ".csv")

    df_mse.to_csv(outfile_inf_csv, index=False)
    df_mse.to_html(outfile_inf_html)

output_file(n=100, p=500, nval=100, alpha= 2., rho=0.35, s=5, beta_type=1, snr=0.71,
            randomizer_scale=1.,
            split_fraction=0.67,
            nsim= 1000,
            Bval= np.array([1, 2, 3, 5, 10, 20, 25, 50]),
            outpath= None)



