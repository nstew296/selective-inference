import numpy as np, sys, os

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from scipy.stats import norm as ndist
from scipy.stats import pearsonr
from selection.randomized.lasso import highdim

def glmnet_lasso(X, y):
    robjects.r('''
                library('glmnet', lib.loc='/Users/snigdhapanigrahi/anaconda/lib/R/library')
                glmnet_LASSO = function(X,y){
                y = as.matrix(y)
                X = as.matrix(X)
                n = nrow(X)
                fit.cv = cv.glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate = coef(fit.cv, s='lambda.1se', exact=TRUE, x=X, y=y)[-1]
                estimate.min = coef(fit.cv, s='lambda.min', exact=TRUE, x=X, y=y)[-1]
                return(list(estimate = estimate, estimate.min = estimate.min,
                            lam.min = fit.cv$lambda.min, lam.1se = fit.cv$lambda.1se))
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

def naive_coverage(inpath, alpha = 0.10):

    X = np.load(os.path.join(inpath, "predictors.npy"))
    y = np.load(os.path.join(inpath, "response.npy"))
    ntrain = (y.shape[0]-105)+1

    cov_unadjusted = 0.
    cov_adjusted = 0.
    for i in range(33):

        indx = np.arange(104) + (i+79+(4*52))
        Y_train = y[indx]
        X_train = X[indx, :]
        n, p = X_train.shape

        mean_effect = Y_train.mean()
        col_means_X = X_train.mean(0)[None, :]
        X_train -= X_train.mean(0)[None, :]
        Y_train = Y_train - Y_train.mean()
        Y_train = Y_train.reshape((Y_train.shape[0],))

        Y_test = y[(104+i+79+(4*52))] - mean_effect
        X_test = X[(104+i+79+(4*52)),:] - col_means_X
        print("shapes", Y_train.shape, X_train.shape, Y_test.shape, X_test.shape)

        _, glm_LASSO, lam_min, lam_1se = glmnet_lasso(X_train, Y_train)
        active_LASSO = (glm_LASSO != 0)
        nactive_LASSO = active_LASSO.sum()
        print("correlation between estimate and response ", pearsonr(X_train.dot(glm_LASSO), Y_train)[0])
        print("number of active variables selected by LASSO ", nactive_LASSO, lam_min)

        dispersion = np.linalg.norm(Y_train - mean_effect - (X_train - col_means_X).dot(glm_LASSO)) ** 2. / (
            n - nactive_LASSO)
        sigma_ = np.sqrt(dispersion)
        _sigma_ = np.std(Y_train) / np.sqrt(2.)
        print("sigma", sigma_, _sigma_)

        # randomized_lasso = highdim.gaussian(X_train,
        #                                     Y_train,
        #                                     n * lam_min * np.ones(p),
        #                                     randomizer_scale= None)
        # signs = randomized_lasso.fit()
        # nonzero = signs != 0
        # estimate, observed_info_mean, _, _, _, _ = randomized_lasso.selective_MLE(target= "selected",
        #                                                                           dispersion=sigma_**2)

        unad_est = np.zeros(p)
        post_LASSO_OLS = np.linalg.pinv(X_train[:, active_LASSO]).dot(Y_train)
        unad_est[active_LASSO] = post_LASSO_OLS
        unad_sd_vector = np.sqrt((sigma_ ** 2) * (np.diag(X_test[:, active_LASSO]
                                                          .dot(np.linalg.inv(X[:, active_LASSO].T.dot(X[:, active_LASSO])))
                                                          .dot(X_test[:, active_LASSO].T)) + 1.))
        print("ratio of variances", np.diag(X_test[:, active_LASSO]
                                                          .dot(np.linalg.inv(X[:, active_LASSO].T.dot(X[:, active_LASSO])))
                                                          .dot(X_test[:, active_LASSO].T)), 1.)
        quantile = ndist.ppf(1 - alpha / 2.)
        unad_prediction_intervals = np.vstack([X_test.dot(unad_est) - quantile * unad_sd_vector,
                                               X_test.dot(unad_est) + quantile * unad_sd_vector]).T

        # ad_est = np.zeros(p)
        # ad_est[nonzero] = estimate
        # ad_sd_vector = np.sqrt((sigma_ ** 2) * (np.diag(X_test[:, nonzero].dot(observed_info_mean)
        #                                                 .dot(X_test[:, nonzero].T)) + 1.))
        # ad_prediction_intervals = np.vstack([X_test.dot(ad_est) - quantile * ad_sd_vector,
        #                                      X_test.dot(ad_est) + quantile * ad_sd_vector]).T

        print("unad prediction intervals", unad_prediction_intervals)
              #, ad_prediction_intervals)
        print("Y test", Y_test)
        cov_unadjusted += np.mean((Y_test > unad_prediction_intervals[:, 0]) * (Y_test < unad_prediction_intervals[:, 1]))
        #cov_adjusted += np.mean((Y_test > ad_prediction_intervals[:, 0]) * (Y_test < ad_prediction_intervals[:, 1]))
        print("coverage so far", cov_unadjusted/(i+1))
              #cov_adjusted/(i+1))

def naive_coverage_increment(inpath, alpha = 0.10):

    X = np.load(os.path.join(inpath, "predictors.npy"))
    y = np.load(os.path.join(inpath, "response.npy"))
    ntrain = (y.shape[0]-105)+1

    cov_unadjusted = 0.
    for i in range(ntrain):
        indx = np.arange(104) + i
        Y_train = y[indx]
        X_train = X[indx, :]
        n, p = X_train.shape

        mean_effect = Y_train.mean()
        col_means_X = X_train.mean(0)[None, :]
        X_train -= X_train.mean(0)[None, :]
        Y_train = Y_train - Y_train.mean()
        Y_train = Y_train.reshape((Y_train.shape[0],))

        Y_test = y[(104+i)] - y[104]
        X_test = (X[(104+i),:] - col_means_X)-(X[104,:]- col_means_X)
        print("shapes", Y_train.shape, X_train.shape, Y_test.shape, X_test.shape)

        _, glm_LASSO, lam_min, lam_1se = glmnet_lasso(X_train, Y_train)
        active_LASSO = (glm_LASSO != 0)
        nactive_LASSO = active_LASSO.sum()
        print("correlation between estimate and response ", pearsonr(X_train.dot(glm_LASSO), Y_train)[0])
        print("number of active variables selected by LASSO ", nactive_LASSO, lam_min)

        dispersion = np.linalg.norm(Y_train-mean_effect- (X_train- col_means_X).dot(glm_LASSO)) ** 2. / (n - nactive_LASSO)
        sigma_ = np.sqrt(dispersion)
        _sigma_ = np.std(Y_train) / np.sqrt(2.)
        print("sigma", _sigma_, sigma_)

        unad_est = np.zeros(p)
        post_LASSO_OLS = np.linalg.pinv(X_train[:, active_LASSO]).dot(Y_train)
        unad_est[active_LASSO] = post_LASSO_OLS
        unad_sd_vector = np.sqrt((sigma_ ** 2) * (np.diag(X_test[:, active_LASSO]
                                                          .dot(np.linalg.inv(X[:, active_LASSO].T.dot(X[:, active_LASSO])))
                                                          .dot(X_test[:, active_LASSO].T)) + 2.))
        print("ratio of variances", np.diag(X_test[:, active_LASSO]
                                                          .dot(np.linalg.inv(X[:, active_LASSO].T.dot(X[:, active_LASSO])))
                                                          .dot(X_test[:, active_LASSO].T)), 2.)
        quantile = ndist.ppf(1 - alpha / 2.)
        unad_prediction_intervals = np.vstack([X_test.dot(unad_est) - quantile * unad_sd_vector,
                                               X_test.dot(unad_est) + quantile * unad_sd_vector]).T

        print("unad prediction intervals", unad_prediction_intervals)
        print("Y test", Y_test)
        cov_unadjusted += np.mean((Y_test > unad_prediction_intervals[:, 0]) * (Y_test < unad_prediction_intervals[:, 1]))
        print("coverage so far", cov_unadjusted/(i+1))

naive_coverage(inpath='/Users/snigdhapanigrahi/Documents/Research/Prediction_argo/', alpha = 0.10)