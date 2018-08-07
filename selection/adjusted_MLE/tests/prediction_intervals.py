import numpy as np, sys

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import selection.randomized.lasso as L; reload(L)
from selection.randomized.lasso import highdim
from scipy.stats import norm as ndist
from selection.algorithms.lasso import lasso_full
from selection.tests.instance import gaussian_instance

from scipy.stats import pearsonr

def glmnet_lasso(X, y):
    robjects.r('''
                library(glmnet)
                glmnet_LASSO = function(X,y){
                y = as.matrix(y)
                X = as.matrix(X)
                n = nrow(X)
                fit.cv = cv.glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate = coef(fit.cv, s='lambda.1se', exact=TRUE, x=X, y=y)[-1]
                estimate.min = coef(fit.cv, s='lambda.min', exact=TRUE, x=X, y=y)[-1]
                return(list(estimate = estimate, estimate.min = estimate.min, lam.min = fit.cv$lambda.min, lam.1se = fit.cv$lambda.1se))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    estimate = np.array(lambda_R(r_X, r_y).rx2('estimate'))
    estimate_min = np.array(lambda_R(r_X, r_y).rx2('estimate.min'))
    lam_min = np.asscalar(np.array(lambda_R(r_X, r_y).rx2('lam.min')))
    lam_1se = np.asscalar(np.array(lambda_R(r_X, r_y).rx2('lam.1se')))
    return estimate, estimate_min, lam_min, lam_1se

def sim_xy(n, p, nval, rho=0, s=5, beta_type=2, snr=1):
    robjects.r('''
    #library(bestsubset)
    source('~/best-subset/bestsubset/R/sim.R')
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

def tuned_lasso(X, y, X_val, y_val):
    robjects.r('''
        source('~/best-subset/bestsubset/R/lasso.R')
        tuned_lasso_estimator = function(X,Y,X.val,Y.val){
        Y = as.matrix(Y)
        X = as.matrix(X)
        Y.val = as.vector(Y.val)
        X.val = as.matrix(X.val)
        rel.LASSO = lasso(X,Y,intercept=TRUE, nrelax=10, nlam=50, standardize=TRUE)
        LASSO = lasso(X,Y,intercept=TRUE,nlam=50, standardize=TRUE)
        beta.hat.rellasso = as.matrix(coef(rel.LASSO))
        beta.hat.lasso = as.matrix(coef(LASSO))
        min.lam = min(rel.LASSO$lambda)
        max.lam = max(rel.LASSO$lambda)

        lam.seq = exp(seq(log(max.lam),log(min.lam),length=rel.LASSO$nlambda))

        muhat.val.rellasso = as.matrix(predict(rel.LASSO, X.val))
        muhat.val.lasso = as.matrix(predict(LASSO, X.val))
        err.val.rellasso = colMeans((muhat.val.rellasso - Y.val)^2)
        err.val.lasso = colMeans((muhat.val.lasso - Y.val)^2)

        opt_lam = ceiling(which.min(err.val.rellasso)/10)
        lambda.tuned.rellasso = lam.seq[opt_lam]
        lambda.tuned.lasso = lam.seq[which.min(err.val.lasso)]
        fit = glmnet(X, Y, standardize=TRUE, intercept=TRUE)
        estimate.tuned = coef(fit, s=lambda.tuned.lasso, exact=TRUE, x=X, y=Y)[-1]
        beta.hat.lasso = (beta.hat.lasso[,which.min(err.val.lasso)])[-1]
        return(list(beta.hat.rellasso = (beta.hat.rellasso[,which.min(err.val.rellasso)])[-1],
        beta.hat.lasso = beta.hat.lasso,
        lambda.tuned.rellasso = lambda.tuned.rellasso, lambda.tuned.lasso= lambda.tuned.lasso,
        lambda.seq = lam.seq))
        }''')

    r_lasso = robjects.globalenv['tuned_lasso_estimator']

    n, p = X.shape
    nval, _ = X_val.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_X_val = robjects.r.matrix(X_val, nrow=nval, ncol=p)
    r_y_val = robjects.r.matrix(y_val, nrow=nval, ncol=1)

    tuned_est = r_lasso(r_X, r_y, r_X_val, r_y_val)
    estimator_rellasso = np.array(tuned_est.rx2('beta.hat.rellasso'))
    estimator_lasso = np.array(tuned_est.rx2('beta.hat.lasso'))
    lam_tuned_rellasso = np.asscalar(np.array(tuned_est.rx2('lambda.tuned.rellasso')))
    lam_tuned_lasso = np.asscalar(np.array(tuned_est.rx2('lambda.tuned.lasso')))
    lam_seq = np.array(tuned_est.rx2('lambda.seq'))
    return estimator_rellasso, estimator_lasso, lam_tuned_rellasso, lam_tuned_lasso, lam_seq

def prediction_intervals_coverage(n=104, p=152, nval=104, rho=0.5, s=20, beta_type=1, snr=0.8,
                                  randomizer_scale=np.sqrt(0.25), target = "selected",
                                  full_dispersion = False, alpha = 0.10):

    while True:
        X, y, X_test, y_test, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
        y = y - y.mean()
        X_test -= X_test.mean(0)[None, :]
        X_test /= (X_test.std(0)[None, :] * np.sqrt(n / (n - 1.)))
        y_test = y_test - y_test.mean()

        if full_dispersion:
            dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
            sigma_ = np.sqrt(dispersion)
        else:
            dispersion = None
            sigma_ = np.std(y)
        print("estimated and true sigma", sigma, sigma_)


        _, glm_LASSO, lam_min , lam_1se = glmnet_lasso(X, y)
        active_LASSO = (glm_LASSO != 0)
        nactive_LASSO = active_LASSO.sum()
        print("correlation between estimate and response", pearsonr(X.dot(glm_LASSO), y))

        randomized_lasso = highdim.gaussian(X,
                                            y,
                                            n * lam_1se * np.ones(p),
                                            randomizer_scale= np.sqrt(n) * randomizer_scale * sigma_)

        signs = randomized_lasso.fit()
        nonzero = signs != 0
        sys.stderr.write("active variables selected by cv LASSO  " + str(nactive_LASSO) + "\n")
        sys.stderr.write("active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")

        if nactive_LASSO>0 and nonzero.sum()>0:

            estimate, _, _, sel_pval, sel_intervals, ind_unbiased_estimator, hessian = randomized_lasso.\
                selective_MLE(target=target,dispersion=dispersion)
            sel_MLE = np.zeros(p)
            sel_MLE[nonzero] = estimate

            quantile = ndist.ppf(1 - alpha / 2.)
            sel_sd_vector = np.sqrt(np.diag(X_test[:, nonzero].dot(hessian).dot(X_test[:, nonzero].T))+ sigma_**2)
            sel_prediction_intervals = np.vstack([X_test.dot(sel_MLE) - quantile * sel_sd_vector,
                                                  X_test.dot(sel_MLE) + quantile * sel_sd_vector]).T

            unad_est = np.zeros(p)
            post_LASSO_OLS = np.linalg.pinv(X[:, active_LASSO]).dot(y)
            unad_est[active_LASSO] = post_LASSO_OLS
            unad_sd_vector = np.sqrt((sigma_**2) * np.diag(X_test[:, active_LASSO]
                                                    .dot(np.linalg.inv(X[:, active_LASSO].T.dot(X[:, active_LASSO])))
                                                    .dot(X_test[:, active_LASSO].T)) + 1.)
            unad_prediction_intervals = np.vstack([X_test.dot(unad_est) - quantile * unad_sd_vector,
                                                   X_test.dot(unad_est) + quantile * unad_sd_vector]).T

            cov_adjusted = np.mean((y_test > sel_prediction_intervals[:, 0])*(y_test < sel_prediction_intervals[:, 1]))
            cov_unadjusted = np.mean((y_test > unad_prediction_intervals[:, 0])*(y_test < unad_prediction_intervals[:, 1]))
            print("coverages", cov_adjusted, cov_unadjusted)
            break

    if True:
        return np.vstack((cov_adjusted,
                          cov_unadjusted))

prediction_intervals_coverage()