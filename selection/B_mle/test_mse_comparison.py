from __future__ import division, print_function
import numpy as np
from selection.randomized.lasso import lasso, split_lasso, selected_targets, full_targets, debiased_targets
from selection.B_mle.utils import sim_xy

def test_compare_split_carve_adaptive(seedn, n=100, p=500, nval=100, alpha=2., rho=0.70, s=10, beta_type=1, snr=0.55,
                                      split_proportion=0.5, B=5, delta = 0.1, lam_frac = 6.):

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

    mle_carved = np.zeros(B)
    mle_split = np.zeros(B)
    mle_split_DTE = np.zeros(B)

    alpha_target_carved = np.zeros(B)
    alpha_target_split = np.zeros(B)
    alpha_target_split_DTE = np.zeros(B)

    for j in range(B):
        perturb_indices = np.zeros(n, np.bool)
        ps_n = int(split_proportion * n)
        perturb_indices[:ps_n] = True
        np.random.shuffle(perturb_indices)

        X_ind = X[~perturb_indices, :]
        Y_ind = y[~perturb_indices]
        lambda_val = lam_frac / np.fabs(X_ind.T.dot(np.linalg.inv(X_ind.dot(X_ind.T) + delta * np.identity(ps_n))).dot(Y_ind))

        X_sel = X[perturb_indices, :]
        Y_sel = y[perturb_indices]
        lambda_val_split = lam_frac / np.fabs(X_sel.T.dot(np.linalg.inv(X_sel.dot(X_sel.T) + delta * np.identity(ps_n))).dot(Y_sel))

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
        mle_carved[j] = mle[0]
        alpha_target_carved[j] = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))[0]

        lasso_sol_split = split_lasso.gaussian(X,
                                               y,
                                               feature_weights=np.append(0.0, lambda_val_split[1:]),
                                               proportion=split_proportion)

        signs_split = lasso_sol_split.fit(perturb=perturb_indices)
        nonzero_split = signs_split != 0
        inf_idx_split = ~lasso_sol_split._selection_idx

        y_inf = y[inf_idx_split]
        X_inf = X[inf_idx_split, :]
        mle_split[j] = np.linalg.pinv(X_inf[:, nonzero_split]).dot(y_inf)[0]
        alpha_target_split[j] = np.linalg.pinv(X_inf[:, nonzero_split]).dot(X_inf.dot(beta))[0]


        perturb_indices_DTE = np.zeros(n, np.bool)
        ps_n_DTE = int(0.67 * n)
        perturb_indices_DTE[:ps_n_DTE] = True
        np.random.shuffle(perturb_indices_DTE)

        X_ind_DTE = X[perturb_indices_DTE, :]
        Y_ind_DTE = y[perturb_indices_DTE]
        lambda_val_DTE = lam_frac / np.fabs(X_ind_DTE.T.dot(np.linalg.inv(X_ind_DTE.dot(X_ind_DTE.T) + delta * np.identity(ps_n_DTE))).dot(Y_ind_DTE))

        lasso_sol_DTE = split_lasso.gaussian(X,
                                             y,
                                             feature_weights= np.append(0.0, lambda_val_DTE[1:]),
                                             proportion=0.67)

        signs_DTE = lasso_sol_DTE.fit(perturb = perturb_indices_DTE)
        nonzero_DTE = signs_DTE != 0
        print("selected by Splits 50% and 67% ", nonzero_split.sum(), nonzero_DTE.sum())
        inf_idx_DTE = ~lasso_sol_DTE._selection_idx

        y_inf_DTE = y[inf_idx_DTE]
        X_inf_DTE = X[inf_idx_DTE, :]
        mle_split_DTE[j] = np.linalg.pinv(X_inf_DTE[:, nonzero_DTE]).dot(y_inf_DTE)[0]
        alpha_target_split_DTE[j] = np.linalg.pinv(X_inf_DTE[:, nonzero_DTE]).dot(X_inf_DTE.dot(beta))[0]

    bias_tar_carved = (np.mean(alpha_target_carved) - alpha)
    bias_carved = (np.mean(mle_carved) - alpha)
    mse_carved = (np.mean(mle_carved) - alpha) ** 2
    fourth_moment_carved = ((np.mean(mle_carved) - alpha) ** 4)

    bias_tar_split = (np.mean(alpha_target_split) - alpha)
    bias_split = (np.mean(mle_split) - alpha)
    mse_split = (np.mean(mle_split) - alpha) ** 2
    fourth_moment_split = ((np.mean(mle_split) - alpha) ** 4)

    bias_tar_split_DTE = (np.mean(alpha_target_split_DTE) - alpha)
    bias_split_DTE = (np.mean(mle_split_DTE) - alpha)
    mse_split_DTE = (np.mean(mle_split_DTE) - alpha) ** 2
    fourth_moment_split_DTE = ((np.mean(mle_split_DTE) - alpha) ** 4)

    print("check ", mse_carved, mse_split)
    print("check ", bias_carved, bias_split, (sigma ** 2) / (n * Sigma[0, 0]))

    return mse_carved, mse_split, mse_split_DTE, bias_carved, bias_split, bias_split_DTE, \
           bias_tar_carved, bias_tar_split, bias_tar_split_DTE

def test_compare_split_carve(seedn, n=100, p=500, nval=100, alpha=2., rho=0.70, s=10, beta_type=1, snr=0.55,
                             split_proportion=0.5, B=5):

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

    mle_carved = np.zeros(B)
    mle_split = np.zeros(B)
    mle_split_DTE = np.zeros(B)

    alpha_target_carved = np.zeros(B)
    alpha_target_split = np.zeros(B)
    alpha_target_split_DTE = np.zeros(B)

    for j in range(B):

        lasso_sol = split_lasso.gaussian(X,
                                         y,
                                         feature_weights=np.append(0.0, lam_theory * np.ones(p - 1)),
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

        mle = lasso_sol.selective_MLE(observed_target,
                                      cov_target,
                                      cov_target_score)[0]
        mle_carved[j] = mle[0]
        alpha_target_carved[j] = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))[0]

        inf_idx = ~lasso_sol._selection_idx
        y_inf = y[inf_idx]
        X_inf = X[inf_idx, :]
        mle_split[j] = np.linalg.pinv(X_inf[:, nonzero]).dot(y_inf)[0]
        alpha_target_split[j] = np.linalg.pinv(X_inf[:, nonzero]).dot(X_inf.dot(beta))[0]

        lasso_sol_DTE = split_lasso.gaussian(X,
                                             y,
                                             feature_weights=np.append(0.0, lam_theory * np.ones(p - 1)),
                                             proportion=0.67)
        signs_DTE = lasso_sol_DTE.fit()
        nonzero_DTE = signs_DTE != 0
        inf_idx_DTE = ~lasso_sol_DTE._selection_idx
        y_inf_DTE = y[inf_idx_DTE]
        X_inf_DTE = X[inf_idx_DTE, :]
        mle_split_DTE[j] = np.linalg.pinv(X_inf_DTE[:, nonzero_DTE]).dot(y_inf_DTE)[0]
        alpha_target_split_DTE[j] = np.linalg.pinv(X_inf_DTE[:, nonzero_DTE]).dot(X_inf_DTE.dot(beta))[0]

    bias_tar_carved = (np.mean(alpha_target_carved) - alpha)
    bias_carved = (np.mean(mle_carved) - alpha)
    mse_carved = (np.mean(mle_carved) - alpha) ** 2
    fourth_moment_carved = ((np.mean(mle_carved) - alpha) ** 4)

    bias_tar_split = (np.mean(alpha_target_split) - alpha)
    bias_split = (np.mean(mle_split) - alpha)
    mse_split = (np.mean(mle_split) - alpha) ** 2
    fourth_moment_split = ((np.mean(mle_split) - alpha) ** 4)

    bias_tar_split_DTE = (np.mean(alpha_target_split_DTE) - alpha)
    bias_split_DTE = (np.mean(mle_split_DTE) - alpha)
    mse_split_DTE = (np.mean(mle_split_DTE) - alpha) ** 2
    fourth_moment_split_DTE = ((np.mean(mle_split_DTE) - alpha) ** 4)

    print("check ", mse_carved, mse_split)
    print("check ", bias_carved, bias_split, (sigma ** 2) / (n * Sigma[0, 0]))

    return mse_carved, mse_split, mse_split_DTE, bias_carved, bias_split, bias_split_DTE, \
           bias_tar_carved, bias_tar_split, bias_tar_split_DTE

    # return np.vstack((bias_tar_split,
    #                   bias_split,
    #                   mse_split,
    #                   fourth_moment_split,
    #                   bias_tar_split_DTE,
    #                   bias_split_DTE,
    #                   mse_split_DTE,
    #                   fourth_moment_split_DTE,
    #                   bias_tar_carved,
    #                   bias_carved,
    #                   mse_carved,
    #                   fourth_moment_carved,
    #                   (sigma ** 2) / (n * Sigma[0, 0])))



def main(nsim, adaptive= False):

    _mse_carved = 0.
    _mse_split = 0.
    _mse_split_DTE = 0.
    _bias_carved = 0.
    _bias_split = 0.
    _bias_split_DTE = 0.
    _bias_tar_carved = 0.
    _bias_tar_split = 0.
    _bias_tar_split_DTE = 0.
    nvar_DTE = 0.
    mse_split_DTE_nvar = 0.

    for i in range(nsim):
        seed = i+100
        if adaptive == True:
            mse_carved, mse_split, mse_split_DTE, bias_carved, \
            bias_split, bias_split_DTE, bias_tar_carved, \
            bias_tar_split, bias_tar_split_DTE = test_compare_split_carve_adaptive(seedn= seed, n=120, p=500, nval=120, alpha=2.,
                                                                                   rho=0.90, s=5, beta_type=2, snr=0.90, split_proportion=0.50,
                                                                                   B=5)
            if mse_split_DTE>5:
                nvar_DTE += 1
                mse_split_DTE_nvar += mse_split_DTE
                mse_split_DTE = 0.
        else:
            mse_carved, mse_split, mse_split_DTE, bias_carved, \
            bias_split, bias_split_DTE, bias_tar_carved, \
            bias_tar_split, bias_tar_split_DTE = test_compare_split_carve(seedn=seed, n=100, p=500, nval=100, alpha=2.,
                                                                          rho=0.35, s=5, beta_type=2, snr=0.90, split_proportion=0.50,
                                                                          B=5)

        _mse_carved += mse_carved
        _mse_split += mse_split
        _mse_split_DTE += mse_split_DTE
        _bias_carved += bias_carved
        _bias_split += bias_split
        _bias_split_DTE += bias_split_DTE
        _bias_tar_carved += bias_tar_carved
        _bias_tar_split +=  bias_tar_split
        _bias_tar_split_DTE += bias_tar_split_DTE

        _var_carved = _mse_carved/(i+1.) - ((_bias_carved/(i+1.))**2)
        _var_split = _mse_split / (i + 1.) - ((_bias_split / (i + 1.)) ** 2)
        _var_split_DTE = _mse_split_DTE / (i + 1.-nvar_DTE) - ((_bias_split_DTE / (i + 1.)) ** 2)

        print("iteration completed ", i, _mse_carved / (i + 1.), _mse_split / (i + 1.),
              _mse_split_DTE / (i + 1. - nvar_DTE), mse_split_DTE_nvar / max(float(nvar_DTE), 1.), (_mse_split_DTE+mse_split_DTE_nvar)/ (i + 1.))
        print("bias ", _bias_carved / (i + 1.), _bias_split / (i + 1.), _bias_split_DTE / (i + 1.))
        print("var so far ", _var_carved, _var_split, _var_split_DTE)
        print("bias in target so far ", _bias_tar_carved / (i + 1.), _bias_tar_split / (i + 1.),
              _bias_tar_split_DTE / (i + 1.))

main(nsim=1000, adaptive= True)


# def main(nsim = 500):
#
#     import pandas as pd
#
#     df_mse = pd.DataFrame()
#
#     for i in range(nsim):
#         seed = i
#         output += np.squeeze(test_compare_split_carve(seedn= seed, n=100, p=500, nval=100, alpha=2.,
#                                                       rho=0.70, s=5, beta_type=2, snr=0.71, split_proportion=0.5, B=5))




