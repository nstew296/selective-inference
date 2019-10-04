from __future__ import division, print_function
import numpy as np
from selection.randomized.lasso import lasso, split_lasso, selected_targets, full_targets, debiased_targets
from selection.B_mle.utils import sim_xyd
from selection.algorithms.lasso import lasso_full

def test_compare_split_carve_adaptive(seedn, n=100, p=500, nval=100, alpha=2., rho=0.70,
                                      s_y=10, beta_type_y=1, snr_y=0.55,
                                      s_d=10, beta_type_d=1, snr_d=0.55,
                                      split_proportion=0.5, B=5,
                                      delta=0.1, lam_frac_y=6., lam_frac_d=4.):

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

    mle_carved = np.zeros(B)
    mle_split = np.zeros(B)
    mle_split_DTE = np.zeros(B)

    alpha_target_carved = np.zeros(B)
    alpha_target_split = np.zeros(B)
    alpha_target_split_DTE = np.zeros(B)

    for j in range(B):
        lambda_val_d = lam_frac_d / np.fabs(x.T.dot(np.linalg.inv(x.dot(x.T) + delta * np.identity(n))).dot(d))
        lasso_d = lasso_full.gaussian(x, d, lambda_val_d)
        signs_d = lasso_d.fit()
        nonzero_d = signs_d != 0

        perturb_indices = np.zeros(n, np.bool)
        ps_n = int(split_proportion * n)
        perturb_indices[:ps_n] = True
        np.random.shuffle(perturb_indices)

        X = np.hstack((d.reshape((n, 1)), x))
        X_ind = X[~perturb_indices, :]
        Y_ind = y[~perturb_indices]
        lambda_val = lam_frac_y / np.fabs(X_ind[:, 1:].T.dot(np.linalg.inv(X_ind[:, 1:].dot(X_ind[:, 1:].T) + delta * np.identity(ps_n))).dot(Y_ind))

        X_sel = X[perturb_indices, :]
        Y_sel = y[perturb_indices]
        lambda_val_split = lam_frac_y / np.fabs(X_sel[:, 1:].T.dot(np.linalg.inv(X_sel[:, 1:].dot(X_sel[:, 1:].T) + delta * np.identity(ps_n))).dot(Y_sel))

        lasso_sol = split_lasso.gaussian(X,
                                         y,
                                         feature_weights= np.append(0., lambda_val),
                                         proportion=split_proportion)

        signs_y = lasso_sol.fit(perturb=perturb_indices)
        nonzero_y = signs_y != 0
        nonzero = np.append(True, np.logical_or(nonzero_y[1:], nonzero_d))
        print("selected carved ", nonzero_y.sum(), nonzero_d.sum(), nonzero.sum())

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
        alpha_target_carved[j] = np.linalg.pinv(X[:,nonzero]).dot(true_mean)[0]
        print("check mle ", mle[0], alpha_target_carved[j], observed_target[0])

        lasso_sol_split = split_lasso.gaussian(X,
                                               y,
                                               feature_weights= np.append(0., lambda_val_split),
                                               proportion=split_proportion)

        signs_y_split = lasso_sol_split.fit(perturb=perturb_indices)
        nonzero_y_split = signs_y_split != 0
        nonzero_split = np.append(True, np.logical_or(nonzero_y_split[1:], nonzero_d))
        print("selected by split 50%", nonzero_y_split.sum(), nonzero_d.sum(), nonzero_split.sum())

        inf_idx_split = ~lasso_sol_split._selection_idx
        y_inf = y[inf_idx_split]
        mle_split[j] = np.linalg.pinv(X[inf_idx_split,:][:, nonzero_split]).dot(y_inf)[0]
        alpha_target_split[j] = np.linalg.pinv(X[inf_idx_split,:][:, nonzero_split]).dot(true_mean[inf_idx_split])[0]

        perturb_indices_DTE = np.zeros(n, np.bool)
        ps_n_DTE = int(0.67 * n)
        perturb_indices_DTE[:ps_n_DTE] = True
        np.random.shuffle(perturb_indices_DTE)

        X_ind_DTE = X[perturb_indices_DTE, :]
        Y_ind_DTE = y[perturb_indices_DTE]
        lambda_val_DTE = lam_frac_y / np.fabs(X_ind_DTE[:,1:].T.dot(np.linalg.inv(X_ind_DTE[:,1:].dot(X_ind_DTE[:,1:].T)
                                                                                  + delta * np.identity(ps_n_DTE))).dot(Y_ind_DTE))

        lasso_sol_DTE = split_lasso.gaussian(X,
                                             y,
                                             feature_weights= np.append(0., lambda_val_DTE),
                                             proportion=0.67)

        signs_y_DTE = lasso_sol_DTE.fit(perturb = perturb_indices_DTE)
        nonzero_y_DTE = signs_y_DTE != 0
        nonzero_DTE = np.append(True, np.logical_or(nonzero_y_DTE[1:], nonzero_d))
        print("selected by split 67%", nonzero_y_DTE.sum(), nonzero_d.sum(), nonzero_DTE.sum())

        inf_idx_DTE = ~lasso_sol_DTE._selection_idx
        y_inf_DTE = y[inf_idx_DTE]
        mle_split_DTE[j] = np.linalg.pinv(X[inf_idx_DTE, :][:, nonzero_DTE]).dot(y_inf_DTE)[0]
        alpha_target_split_DTE[j] = np.linalg.pinv(X[inf_idx_DTE, :][:, nonzero_DTE]).dot(true_mean[inf_idx_DTE])[0]

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
    print("check ", bias_carved, bias_split, (sigma_y ** 2) / (n * Sigma[0, 0]))

    return mse_carved, mse_split, mse_split_DTE, bias_carved, bias_split, bias_split_DTE, \
           bias_tar_carved, bias_tar_split, bias_tar_split_DTE

def main(nsim):

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

        mse_carved, mse_split, mse_split_DTE, bias_carved, \
        bias_split, bias_split_DTE, bias_tar_carved, \
        bias_tar_split, bias_tar_split_DTE = test_compare_split_carve_adaptive(seedn=seed, n=120, p=500, nval=120, alpha=2., rho=0.90,
                                                                               s_y=5, beta_type_y=2, snr_y=0.90,
                                                                               s_d=5, beta_type_d=3, snr_d=0.90,
                                                                               split_proportion=0.5, B=5,
                                                                               delta=0.1, lam_frac_y=8., lam_frac_d=5.)

        if mse_split_DTE > 5:
            nvar_DTE += 1
            mse_split_DTE_nvar += mse_split_DTE
            mse_split_DTE = 0.

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
              _mse_split_DTE / (i + 1. - nvar_DTE), mse_split_DTE_nvar / max(float(nvar_DTE), 1.))
        print("bias ", _bias_carved / (i + 1.), _bias_split / (i + 1.), _bias_split_DTE / (i + 1.))
        print("var so far ", _var_carved, _var_split, _var_split_DTE)
        print("bias in target so far ", _bias_tar_carved / (i + 1.), _bias_tar_split / (i + 1.),
              _bias_tar_split_DTE / (i + 1.))

main(nsim=1000)