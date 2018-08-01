import numpy as np, sys

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from scipy.stats import norm as ndist
from selection.randomized.lasso import lasso, full_targets, selected_targets, debiased_targets
from selection.algorithms.lasso import lasso_full

def sim_xy(n, p, nval, rho=0, s=5, beta_type=2, snr=1):
    robjects.r('''
    library(bestsubset)
    sim_xy = bestsubset::sim.xy
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

def selInf_R(X, y, beta, lam, sigma, Type, alpha=0.1):
    robjects.r('''
               library("selectiveInference")
               selInf = function(X, y, beta, lam, sigma, Type, alpha= 0.1){
               y = as.matrix(y)
               X = as.matrix(X)
               beta = as.matrix(beta)
               lam = as.matrix(lam)[1,1]
               sigma = as.matrix(sigma)[1,1]
               Type = as.matrix(Type)[1,1]
               if(Type == 1){
                   type = "full"} else{
                   type = "partial"}
               inf = fixedLassoInf(x = X, y = y, beta = beta, lambda=lam, family = "gaussian",
                                   intercept=FALSE, sigma=sigma, alpha=alpha, type=type)
               return(list(ci = inf$ci, pvalue = inf$pv))}
               ''')

    inf_R = robjects.globalenv['selInf']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_beta = robjects.r.matrix(beta, nrow=p, ncol=1)
    r_lam = robjects.r.matrix(lam, nrow=1, ncol=1)
    r_sigma = robjects.r.matrix(sigma, nrow=1, ncol=1)
    r_Type = robjects.r.matrix(Type, nrow=1, ncol=1)
    output = inf_R(r_X, r_y, r_beta, r_lam, r_sigma, r_Type)
    ci = np.array(output.rx2('ci'))
    pvalue = np.array(output.rx2('pvalue'))
    return ci, pvalue

def glmnet_lasso(X, y):
    robjects.r('''
                library(glmnet)
                glmnet_LASSO = function(X,y){
                y = as.matrix(y)
                X = as.matrix(X)
                n = nrow(X)
                fit.cv = cv.glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate.1se = coef(fit.cv, s='lambda.1se', exact=TRUE, x=X, y=y)[-1]
                estimate.min = coef(fit.cv, s='lambda.min', exact=TRUE, x=X, y=y)[-1]
                return(list(estimate.1se = estimate.1se, estimate.min = estimate.min, lam.min = fit.cv$lambda.min, lam.1se = fit.cv$lambda.1se))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    estimate_1se = np.array(lambda_R(r_X, r_y).rx2('estimate.1se'))
    estimate_min = np.array(lambda_R(r_X, r_y).rx2('estimate.min'))
    lam_min = np.asscalar(np.array(lambda_R(r_X, r_y).rx2('lam.min')))
    lam_1se = np.asscalar(np.array(lambda_R(r_X, r_y).rx2('lam.1se')))
    return estimate_1se, estimate_min, lam_min, lam_1se

def coverage(intervals, pval, truth):
    if (truth!=0).sum()!=0:
        avg_power = np.mean(pval[truth != 0])
    else:
        avg_power = 0.
    return np.mean((truth > intervals[:, 0])*(truth < intervals[:, 1])), avg_power

def BHfilter(pval, q=0.2):
    robjects.r.assign('pval', pval)
    robjects.r.assign('q', q)
    robjects.r('Pval = p.adjust(pval, method="BH")')
    robjects.r('S = which((Pval < q)) - 1')
    S = robjects.r('S')
    ind = np.zeros(pval.shape[0], np.bool)
    ind[np.asarray(S, np.int)] = 1
    return ind

def relative_risk(est, truth, Sigma):
    return (est - truth).T.dot(Sigma).dot(est - truth) / truth.T.dot(Sigma).dot(truth)

def comparison_cvmetrics_selected(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=1, snr=0.20,
                                  randomizer_scale = np.sqrt(0.25), full_dispersion = True,
                                  tuning_nonrand = "lambda.min", tuning_rand = "lambda.1se"):

    X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
    y = y - y.mean()
    true_signals = np.zeros(p, np.bool)
    true_signals[beta != 0] = 1
    true_set = np.asarray([u for u in range(p) if true_signals[u]])

    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)
    print("estimated and true sigma", sigma, sigma_)

    glm_LASSO_1se, glm_LASSO_min, lam_min, lam_1se = glmnet_lasso(X, y)
    if tuning_nonrand == "lambda.min":
        lam_LASSO = lam_min
        glm_LASSO = glm_LASSO_min
    else:
        lam_LASSO = lam_1se
        glm_LASSO = glm_LASSO_1se
    rel_LASSO = np.zeros(p)
    active_LASSO = (glm_LASSO != 0)
    nactive_LASSO = active_LASSO.sum()
    active_set_LASSO = np.asarray([r for r in range(p) if active_LASSO[r]])
    active_LASSO_bool = np.zeros(nactive_LASSO, np.bool)
    for z in range(nactive_LASSO):
        active_LASSO_bool[z] = (np.in1d(active_set_LASSO[z], true_set).sum() > 0)

    if nactive_LASSO>0:
        rel_LASSO[active_LASSO] = np.linalg.pinv(X[:, active_LASSO]).dot(y)
        Lee_target = np.linalg.pinv(X[:, active_LASSO]).dot(X.dot(beta))
        Lee_intervals, Lee_pval = selInf_R(X, y, glm_LASSO, n * lam_LASSO, sigma_, Type=0, alpha=0.1)
        if (Lee_pval.shape[0] == Lee_target.shape[0]):

            cov_Lee, _ = coverage(Lee_intervals, Lee_pval, Lee_target)
            inf_entries = np.isinf(Lee_intervals[:, 1] - Lee_intervals[:, 0])
            if inf_entries.sum() == nactive_LASSO:
                length_Lee = 0.
            else:
                length_Lee = np.mean((Lee_intervals[:, 1] - Lee_intervals[:, 0])[~inf_entries])
            power_Lee = ((active_LASSO_bool) * (np.logical_or((0. < Lee_intervals[:, 0]), (0. > Lee_intervals[:, 1])))).sum()
            Lee_discoveries = BHfilter(Lee_pval, q=0.1)
            power_Lee_BH = (Lee_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
            fdr_Lee_BH = (Lee_discoveries * ~active_LASSO_bool).sum() / float(max(Lee_discoveries.sum(), 1.))

            post_LASSO_OLS = np.linalg.pinv(X[:, active_LASSO]).dot(y)
            naive_sd = sigma_ * np.sqrt(np.diag((np.linalg.inv(X[:, active_LASSO].T.dot(X[:, active_LASSO])))))
            naive_intervals = np.vstack([post_LASSO_OLS - 1.65 * naive_sd,
                                        post_LASSO_OLS + 1.65 * naive_sd]).T
            naive_pval = ndist.cdf(post_LASSO_OLS / naive_sd)
            cov_naive, _ = coverage(naive_intervals, naive_pval, Lee_target)
            length_naive = np.mean(naive_intervals[:, 1] - naive_intervals[:, 0])
            power_naive = ((active_LASSO_bool) * (np.logical_or((0. < naive_intervals[:, 0]), (0. > naive_intervals[:, 1])))).sum()
            naive_discoveries = BHfilter(naive_pval, q=0.1)
            power_naive_BH = (naive_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
            fdr_naive_BH = (naive_discoveries * ~active_LASSO_bool).sum() / float(max(naive_discoveries.sum(), 1.))
        else:
            Lee_nreport = 1
            naive_nreport = 1
    elif nactive_LASSO == 0:
        Lee_nreport = 1
        naive_nreport = 1

    if tuning_rand == "lambda.min":
        randomized_lasso = lasso(X,
                                 y,
                                 n * lam_min * np.ones(p),
                                 randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)
    else:
        randomized_lasso = lasso(X,
                                 y,
                                 n * lam_1se * np.ones(p),
                                 randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

    signs = randomized_lasso.fit()
    nonzero = signs != 0
    active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])
    active_rand_bool = np.zeros(nonzero.sum(), np.bool)
    for x in range(nonzero.sum()):
        active_rand_bool[x] = (np.in1d(active_set_rand[x], true_set).sum() > 0)
    sel_MLE = np.zeros(p)
    ind_est = np.zeros(p)
    randomized_lasso_est = np.zeros(p)
    randomized_rel_lasso_est = np.zeros(p)

    if nonzero.sum()>0:
        target_randomized = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(randomized_lasso.loglike,
                                          randomized_lasso._W,
                                          nonzero,
                                          dispersion=dispersion)
        MLE_estimate, _, _, MLE_pval, MLE_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(observed_target,
                                                                                                             cov_target,
                                                                                                             cov_target_score,
                                                                                                             alternatives)
        sel_MLE[nonzero] = MLE_estimate
        ind_est[nonzero] = ind_unbiased_estimator
        randomized_lasso_est[nonzero] = randomized_lasso.initial_soln
        randomized_rel_lasso_est[nonzero] = randomized_lasso._beta_full

        cov_MLE, _ = coverage(MLE_intervals, MLE_pval, target_randomized)
        length_MLE = np.mean(MLE_intervals[:, 1] - MLE_intervals[:, 0])
        power_MLE = ((active_rand_bool) * (np.logical_or((0. < MLE_intervals[:, 0]), (0. > MLE_intervals[:, 1])))).sum()
        MLE_discoveries = BHfilter(MLE_pval, q=0.1)
        power_MLE_BH = (MLE_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
        fdr_MLE_BH = (MLE_discoveries * ~active_rand_bool).sum() / float(max(MLE_discoveries.sum(), 1.))
    else:
        MLE_nreport = 1

    return np.vstack((relative_risk(sel_MLE, beta, Sigma),
                      relative_risk(ind_est, beta, Sigma),
                      relative_risk(randomized_lasso_est, beta, Sigma),
                      relative_risk(randomized_rel_lasso_est, beta, Sigma),
                      relative_risk(rel_LASSO, beta, Sigma),
                      relative_risk(glm_LASSO, beta, Sigma),
                      cov_MLE,
                      cov_Lee,
                      cov_naive,
                      length_MLE,
                      length_Lee,
                      inf_entries.sum() / float(nactive_LASSO),
                      length_naive,
                      power_MLE / float((beta != 0).sum()),
                      power_Lee / float((beta != 0).sum()),
                      power_naive / float((beta != 0).sum()),
                      power_MLE_BH,
                      power_Lee_BH,
                      power_naive_BH,
                      fdr_MLE_BH,
                      fdr_Lee_BH,
                      fdr_naive_BH,
                      nonzero.sum(),
                      nactive_LASSO,
                      MLE_discoveries.sum(),
                      Lee_discoveries.sum(),
                      naive_discoveries.sum(),
                      np.mean(MLE_estimate - target_randomized)))

















