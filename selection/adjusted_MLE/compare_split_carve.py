import numpy as np, sys, time

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from scipy.stats import norm as ndist
from selection.randomized.lasso import lasso, split_lasso, full_targets, selected_targets, debiased_targets
from selection.adjusted_MLE.cv_MLE import coverage, BHfilter

def sim_xy(n, p, nval, rho=0, s=5, beta_type=2, snr=1):
    robjects.r('''
    #library(bestsubset)
    source('~/best-subset/bestsubset/R/sim.R')
    sim_xy = sim.xy
    ''')

    r_simulate = robjects.globalenv['sim_xy']
    sim = r_simulate(n, p, nval, rho, s, beta_type, snr)
    X = np.array(sim.rx2('x'))
    y = np.array(sim.rx2('y'))
    X_val = np.array(sim.rx2('xval'))
    y_val = np.array(sim.rx2('yval'))
    Sigma = np.array(sim.rx2('Sigma'))
    beta = np.array(sim.rx2('beta'))
    sigma = np.array(sim.rx2('sigma'))

    return X, y, X_val, y_val, Sigma, beta, sigma


def glmnet_lasso(X, y, lambda_val):
    robjects.r('''
                library(glmnet)
                glmnet_LASSO = function(X,y, lambda){
                y = as.matrix(y)
                X = as.matrix(X)
                lam = as.matrix(lambda)[1,1]
                n = nrow(X)

                fit = glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate = coef(fit, s=lam, exact=TRUE, x=X, y=y)[-1]
                fit.cv = cv.glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate.1se = coef(fit.cv, s='lambda.1se', exact=TRUE, x=X, y=y)[-1]
                estimate.min = coef(fit.cv, s='lambda.min', exact=TRUE, x=X, y=y)[-1]
                return(list(estimate = estimate, estimate.1se = estimate.1se, estimate.min = estimate.min, lam.min = fit.cv$lambda.min, lam.1se = fit.cv$lambda.1se))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_lam = robjects.r.matrix(lambda_val, nrow=1, ncol=1)

    estimate = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate'))
    estimate_1se = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate.1se'))
    estimate_min = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate.min'))
    lam_min = np.asscalar(np.array(lambda_R(r_X, r_y, r_lam).rx2('lam.min')))
    lam_1se = np.asscalar(np.array(lambda_R(r_X, r_y, r_lam).rx2('lam.1se')))
    return estimate, estimate_1se, estimate_min, lam_min, lam_1se

def compare_split_MLE(n=100, p=500, nval=100, rho=0.40, s=5, beta_type=1, snr=0.55, target= "selected",
                      full_dispersion=False, split_proportion=0.50):

    X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
    print("snr", snr)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
    y = y - y.mean()
    true_set = np.asarray([u for u in range(p) if beta[u] != 0])

    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)
    print("estimated and true sigma", sigma, sigma_, split_proportion)

    lam_theory = sigma_ * 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))

    randomized_lasso = split_lasso.gaussian(X,
                                            y,
                                            feature_weights= lam_theory * np.ones(p),
                                            proportion= split_proportion)

    signs = randomized_lasso.fit(estimate_dispersion= True)
    nonzero = signs != 0
    sys.stderr.write("active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")
    active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])
    active_rand_bool = np.asarray([(np.in1d(active_set_rand[x], true_set).sum() > 0) for x in range(nonzero.sum())],
                                  np.bool)
    nreport = 0.

    if nonzero.sum() > 0:
        target_randomized = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(randomized_lasso.loglike,
                                          randomized_lasso._W,
                                          nonzero)

        toc = time.time()
        MLE_estimate, _, _, MLE_pval, MLE_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(observed_target,
                                                                                                             cov_target,
                                                                                                             cov_target_score)
        tic = time.time()
        time_MLE = tic - toc

        cov_MLE, selective_MLE_power = coverage(MLE_intervals, MLE_pval, target_randomized, beta[nonzero])
        length_MLE = np.mean(MLE_intervals[:, 1] - MLE_intervals[:, 0])
        power_MLE = ((active_rand_bool) * (np.logical_or((0. < MLE_intervals[:, 0]), (0. > MLE_intervals[:, 1])))).sum() / float((beta != 0).sum())
        fdr_MLE = ((~active_rand_bool) * (np.logical_or((0. < MLE_intervals[:, 0]), (0. > MLE_intervals[:, 1])))).sum() / float(nonzero.sum())
        MLE_discoveries = BHfilter(MLE_pval, q=0.1)
        power_MLE_BH = (MLE_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
        fdr_MLE_BH = (MLE_discoveries * ~active_rand_bool).sum() / float(max(MLE_discoveries.sum(), 1.))
        bias_MLE = np.mean(MLE_estimate - target_randomized)

    else:
        nreport += 1
        cov_MLE, length_MLE, power_MLE, fdr_MLE, power_MLE_BH, fdr_MLE_BH, bias_MLE, selective_MLE_power, time_MLE = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
        MLE_discoveries = np.zeros(1)

    sel_idx = randomized_lasso._selection_idx
    inf_idx = ~sel_idx
    y_inf = y[inf_idx]
    X_inf = X[inf_idx, :]

    active_LASSO_split = nonzero
    nactive_LASSO_split = active_LASSO_split.sum()
    sys.stderr.write("active variables selected by split LASSO " + str(nactive_LASSO_split) + "\n" + "\n")
    active_set_LASSO_split = np.asarray([r for r in range(p) if active_LASSO_split[r]])
    active_LASSO_bool_split = np.asarray([(np.in1d(active_set_LASSO_split[z], true_set).sum() > 0) for z in range(nactive_LASSO_split)],np.bool)
    nreport_split = 0.

    if nactive_LASSO_split > 0:
        target_split = np.linalg.pinv(X[:, active_LASSO_split]).dot(X.dot(beta))
        rel_LASSO_split = np.zeros(p)
        rel_LASSO_split[active_LASSO_split] = np.linalg.pinv(X_inf[:, active_LASSO_split]).dot(y_inf)
        post_LASSO_OLS_split = np.linalg.pinv(X_inf[:, active_LASSO_split]).dot(y_inf)
        sd_split = sigma_ * np.sqrt(
            np.diag((np.linalg.inv(X_inf[:, active_LASSO_split].T.dot(X_inf[:, active_LASSO_split])))))
        intervals_split = np.vstack([post_LASSO_OLS_split - 1.65 * sd_split,
                                     post_LASSO_OLS_split + 1.65 * sd_split]).T
        pval_split = 2. *(1.-ndist.cdf(np.abs(post_LASSO_OLS_split) / sd_split))
        cov_split, selective_power_split = coverage(intervals_split, pval_split, target_split,
                                                    beta[active_LASSO_split])
        length_split = np.mean(intervals_split[:, 1] - intervals_split[:, 0])
        power_split = ((active_LASSO_bool_split) * (
            np.logical_or((0. < intervals_split[:, 0]), (0. > intervals_split[:, 1])))).sum() / float((beta != 0).sum())
        fdr_split = ((~active_LASSO_bool_split) * (
            np.logical_or((0. < intervals_split[:, 0]), (0. > intervals_split[:, 1])))).sum() / float(
            nactive_LASSO_split)

        discoveries_split = BHfilter(pval_split, q=0.1)
        power_split_BH = (discoveries_split * active_LASSO_bool_split).sum() / float((beta != 0).sum())
        fdr_split_BH = (discoveries_split * ~active_LASSO_bool_split).sum() / float(
            max(discoveries_split.sum(), 1.))
        bias_split = np.mean(rel_LASSO_split[active_LASSO_split] - target_split)

    elif nactive_LASSO_split == 0:
        nreport_split += 1
        cov_split, length_split, power_split, fdr_split, power_split_BH, fdr_split_BH, bias_split, selective_power_split = [0., 0., 0., 0., 0., 0., 0., 0.]
        discoveries_split = np.zeros(1)

    MLE_inf = np.vstack((cov_MLE, length_MLE, 0., nonzero.sum(), bias_MLE, selective_MLE_power, time_MLE, fdr_MLE,
                         power_MLE, power_MLE_BH, fdr_MLE_BH, MLE_discoveries.sum()))

    split_inf = np.vstack((cov_split, length_split, 0., nactive_LASSO_split, bias_split, selective_power_split, 0., fdr_split,
                           power_split, power_split_BH, fdr_split_BH, discoveries_split.sum()))

    print("MLE-inference", MLE_inf, nreport)
    print("split inference", split_inf, nreport_split)
    return np.vstack((MLE_inf, split_inf, nreport, nreport_split))

def main(nsim=100):
    output_overall = np.zeros(26)
    for i in range(nsim):
        output_overall += np.squeeze(compare_split_MLE(n=100, p=500, nval=100, rho=0.70, s=5, beta_type=1, snr=0.70, target= "selected",
                                     full_dispersion=False, split_proportion=0.50))
        nreport = output_overall[24]
        nreport_split = output_overall[25]
    carved_inf = np.hstack(((output_overall[0:8] / float(nsim - nreport)).reshape((1, 8)),
                            (output_overall[8:12] / float(nsim)).reshape((1, 4))))
    split_inf = np.hstack(((output_overall[12:20] / float(nsim - nreport_split)).reshape((1, 8)),
                            (output_overall[20:24] / float(nsim)).reshape((1, 4))))

    print("check carved ", carved_inf, nreport)
    print("check split ", split_inf, nreport_split, nsim - nreport)

main(nsim=200)