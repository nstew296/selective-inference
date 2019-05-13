from __future__ import division, print_function
import numpy as np

from selection.randomized.lasso import lasso, selected_targets, full_targets, debiased_targets
from selection.multiple_splits.utils import sim_xy, glmnet_lasso_cv1se, glmnet_lasso_cvmin, glmnet_lasso, glmnet_lasso_cv
from scipy.stats import norm as ndist
from selection.randomized.selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C

def boot_mle(boot_target,
             observed_target,
             cov_target,
             cov_target_score,
             init_soln,
             cond_mean,
             cond_cov,
             logdens_linear,
             linear_part,
             offset,
             solve_args={'tol': 1.e-15}):


    if np.asarray(observed_target).shape in [(), (0,)]:
        raise ValueError('no target specified')

    observed_target = np.atleast_1d(observed_target)
    prec_target = np.linalg.inv(cov_target)
    target_lin = - logdens_linear.dot(cov_target_score.T.dot(prec_target))

    prec_opt = np.linalg.inv(cond_cov)
    solver = solve_barrier_affine_C

    cond_mean_boot = target_lin.dot(boot_target) + (cond_mean - target_lin.dot(observed_target))
    conjugate_arg = prec_opt.dot(cond_mean_boot)

    _, soln, _ = solver(conjugate_arg,
                        prec_opt,
                        init_soln,
                        linear_part,
                        offset,
                        **solve_args)

    return boot_target + cov_target.dot(target_lin.T.dot(prec_opt.dot(cond_mean_boot - soln)))


def test_ci(n=200, p=1000, nval=200, alpha= 2., rho=0.70, s=10, beta_type=0, snr=1.0, randomizer_scale=1., B=5, nboot = 5000):

    X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, alpha=alpha, rho=rho, s=s, beta_type=beta_type,
                                            snr=snr)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
    y = y - y.mean()
    dispersion = None
    sigma_ = np.std(y)

    lam_theory = sigma_ * np.mean(np.fabs(np.dot(X[:, 1:].T, np.random.standard_normal((n, 2000)))).max(0))

    sel_mle = np.zeros(B)
    boot_sample = np.zeros((nboot, B))
    resid = np.zeros((n, B))

    nonzero_list = []
    observed_target_list = []
    cov_target_list = []
    cov_target_score_list = []
    observed_opt_state_list = []
    cond_mean_list = []
    cond_cov_list = []
    logdens_linear_list = []
    A_scaling_list = []
    b_scaling_list = []
    for j in range(B):
        lasso_sol = lasso.gaussian(X,
                                   y,
                                   feature_weights=np.append(0.001, np.ones(p - 1) * lam_theory),
                                   randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)
        signs = lasso_sol.fit()
        nonzero = signs != 0
        nonzero_list.append(nonzero)

        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(lasso_sol.loglike,
                                          lasso_sol._W,
                                          nonzero,
                                          dispersion=dispersion)

        full_mle, _, _, _, _, _, _, _ = lasso_sol.selective_MLE(observed_target,
                                                                cov_target,
                                                                cov_target_score,
                                                                alternatives)

        sel_mle[j] = full_mle[0]
        observed_target_list.append(observed_target)
        cov_target_list.append(cov_target)
        cov_target_score_list.append(cov_target_score)
        observed_opt_state_list.append(lasso_sol.observed_opt_state)
        cond_mean_list.append(lasso_sol.cond_mean)
        cond_cov_list.append(lasso_sol.cond_cov)
        logdens_linear_list.append(lasso_sol.logdens_linear)
        A_scaling_list.append(lasso_sol.A_scaling)
        b_scaling_list.append(lasso_sol.b_scaling)

        # observed_target_uni = (observed_target[0]).reshape((1,))
        # cov_target_uni = (np.diag(cov_target)[0]).reshape((1, 1))
        # cov_target_score_uni = cov_target_score[0, :].reshape((1, p))
        #
        # sel_mle[j], var, _, _, _, _, _, _ = lasso_sol.selective_MLE(observed_target_uni,
        #                                                             cov_target_uni,
        #                                                             cov_target_score_uni,
        #                                                             alternatives)

        resid[:, j] = y - X[:, nonzero].dot(observed_target)

    for b in range(nboot):
        boot_indices = np.random.choice(n, n, replace=True)
        for k in range(B):
            boot_vector = (X[boot_indices, :][:, nonzero_list[k]]).T.dot(resid[:, k][boot_indices])
            target_boot = np.linalg.inv(X[:, nonzero_list[k]].T.dot(X[:, nonzero_list[k]])).dot(boot_vector) + observed_target_list[k]
            #boot_observed_target = target_boot[0].reshape((1,))

            full_boot_mle = boot_mle(target_boot,
                                     observed_target_list[k],
                                     cov_target_list[k],
                                     cov_target_score_list[k],
                                     observed_opt_state_list[k],
                                     cond_mean_list[k],
                                     cond_cov_list[k],
                                     logdens_linear_list[k],
                                     A_scaling_list[k],
                                     b_scaling_list[k])

            boot_sample[b, k] = full_boot_mle[0]

    cc_estimate = np.mean(sel_mle)
    boot_cc_estimate = np.mean(boot_sample, axis=1)
    std_estimate = boot_cc_estimate.std()

    quantile = 1.65
    intervals = np.vstack([cc_estimate - quantile * std_estimate,
                           cc_estimate + quantile * std_estimate]).T

    print("intervals", intervals)

    return intervals

def check(n=200, p=1000, nval=200, alpha= 2., rho=0.35, s=10, beta_type=0, snr=0.71, randomizer_scale=1., nsim =200):

    coverage = 0.
    for niter in range(nsim):

        intervals = test_ci(n=n,
                            p=p,
                            nval=nval,
                            alpha=alpha,
                            rho=rho,
                            s=s,
                            beta_type= beta_type,
                            snr=snr,
                            randomizer_scale=randomizer_scale,
                            B=5)

        coverage += (alpha > intervals[:, 0]) * (alpha < intervals[:, 1])

        print("coverage so far ", coverage/(niter+1.), niter+1)

check()


