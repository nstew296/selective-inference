import numpy as np, sys, os

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import selection.randomized.lasso as L; reload(L)
from selection.randomized.lasso import highdim
from selection.randomized.randomization import randomization
import pandas as pd

def generate_data():
    robjects.r('''
                library('grf')
                generate_X_y = function(inpath="/Users/snigdhapanigrahi/Documents/Research/Effect_modification/synthetic_data.csv")
                {
                   data <- read.csv()
                   data$C1 <- factor(data$C1)
                   data$C2 <- factor(data$C2)
                   data$C3 <- factor(data$C3)
                   data$XC <- factor(data$XC)
                   print("here")

                   y <- data$Y
                   t <- data$Z
                   x <- model.matrix(~ ., data[, -c(1:3)])[, -1]
                   x <- scale(x, scale = FALSE)
                   set.seed("20180511")
                   mu.y <- regression_forest(x, y, tune.parameters = TRUE)
                   mu.t <- regression_forest(x, t, tune.parameters = TRUE)
                   print("here")
                   y.tilde <- y - predict(mu.y)$predictions
                   t.tilde <- t - predict(mu.t, x)$predictions

                   x.tilde <- x
                   x.tilde <- x.tilde[, apply(x.tilde, 2, sd) > 0]
                   x.tilde <- x.tilde[, !duplicated(t(x.tilde))]
                   x.tilde <- scale(x.tilde, scale = FALSE) * as.vector(t.tilde)
                   print("here")

                   return(list(y=y.tilde, X=x.tilde))
                   }''')

    gen_data_R = robjects.globalenv['generate_X_y']
    y = np.array(gen_data_R().rx2('y'))
    X = np.array(gen_data_R().rx2('X'))
    return y, X

def glmnet_lasso(X, y, lambda_val):
    robjects.r('''
                library('glmnet')
                glmnet_LASSO = function(X,y,lambda){
                y = as.matrix(y)
                X = as.matrix(X)
                lam = as.matrix(lambda)[1,1]
                n = nrow(X)
                fit = glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                fit.cv = cv.glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate = coef(fit, s=lam, exact=TRUE, x=X, y=y)[-1]
                estimate.min = coef(fit.cv, s='lambda.min', exact=TRUE, x=X, y=y)[-1]
                estimate.1se = coef(fit.cv, s='lambda.1se', exact=TRUE, x=X, y=y)[-1]
                return(list(estimate = estimate, estimate.1se = estimate.1se, estimate.min = estimate.min,
                            lam.min = fit.cv$lambda.min, lam.1se = fit.cv$lambda.1se))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_lam = robjects.r.matrix(lambda_val, nrow=1, ncol=1)
    estimate = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate'))
    lam_min = np.array(lambda_R(r_X, r_y, r_lam).rx2('lam.min'))
    lam_1se = np.array(lambda_R(r_X, r_y, r_lam).rx2('lam.1se'))
    estimate_min = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate.min'))
    estimate_1se = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate.1se'))
    return estimate, lam_min, lam_1se, estimate_min, estimate_1se

def randomized_inference(X, y, randomizer_scale = np.sqrt(0.50), target = "full", tuning = "theory", dispersion = None):

    n, p = X.shape
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
    y = y - y.mean()
    y= y.reshape((y.shape[0], ))

    if dispersion is None:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2. / (n - p)

    sigma_ = np.sqrt(dispersion)
    lam_theory = sigma_ * 1.1 * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    if tuning == "theory":
        glm_LASSO, _, _, _, _ = glmnet_lasso(X, y, lam_theory / float(n))
        lam = lam_theory/float(n)
    elif tuning == "lambda.1se":
        _, _, lam_1se, _, glm_LASSO = glmnet_lasso(X, y, lam_theory / float(n))
        lam = lam_1se
    elif tuning == "lambda.min":
        _, lam_min, _, glm_LASSO, _ = glmnet_lasso(X, y, lam_theory / float(n))
        lam = lam_min

    sys.stderr.write("lam in randomized LASSO " + str(lam) + "\n" + "\n")
    active_LASSO = (glm_LASSO != 0)
    active_set = np.asarray([z for z in range(p) if active_LASSO[z]])

    ###running randomized LASSO at lam
    randomized_lasso = highdim.gaussian(X,
                                        y,
                                        n * lam * np.ones(p),
                                        randomizer_scale= np.sqrt(n) * randomizer_scale * sigma_)
    signs = randomized_lasso.fit(solve_args={'tol': 1.e-5, 'min_its': 100})
    ini_perturb = randomized_lasso._initial_omega
    nonzero = signs != 0
    active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])

    ###comparison of active sets by both randomized and non-randomized LASSO
    sys.stderr.write(str(active_LASSO.sum()) + " active variables selected by non-randomized LASSO " + str(active_set) + "\n" + "\n")
    sys.stderr.write(str(nonzero.sum()) + " active variables selected by randomized LASSO " + str(active_set_rand) + "\n" + "\n")

    estimate, _, _, _, _, _ = randomized_lasso.selective_MLE(target=target,
                                                             dispersion=dispersion)

    _, pval, intervals = randomized_lasso.summary(target=target, dispersion=sigma_, compute_intervals=True, level=0.90, ndraw=100000)
    sys.stderr.write("theoretical lambda: pvals based on sampler " + str(pval) + "\n" + "\n")
    sys.stderr.write("theoretical lambda: intervals based on sampler " + str(intervals.T) + "\n" + "\n")

    return np.vstack((active_set_rand.astype(int),
                      estimate,
                      pval,
                      intervals[:, 0],
                      intervals[:, 1])).T

def main(inpath, outpath = None, randomizer_scale= np.sqrt(0.50), target = "selected", tuning = "theory"):

    if inpath is None:
        y, X = generate_data()
    else:
        X = np.load(os.path.join(inpath, "predictors.npy"))
        y = np.load(os.path.join(inpath, "response.npy"))

    np.random.seed(0)
    n, p = X.shape

    dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2. / (n - p)
    sigma_ = np.sqrt(dispersion)
    ini_perturb = randomization.isotropic_gaussian((p,), np.sqrt(n) * randomizer_scale * sigma_).sample()
    output = randomized_inference(X, y, randomizer_scale=randomizer_scale, target= target, tuning = tuning,
                                  dispersion=dispersion)

    output_df = pd.DataFrame(data=output[:,1:],columns=['sel-MLE', 'pval', 'lower_ci', 'upper_ci'])
    output_df['active-vars'] = (output[:,0]).astype(int)

    if outpath is None:
        outpath = os.path.dirname(__file__)

    outfile_html = os.path.join(outpath, "selective_inference_" + str(tuning) + ".html")
    outfile_csv = os.path.join(outpath, "selective_inference_" + str(tuning) +".csv")
    output_df.to_csv(outfile_csv, index=False)
    output_df.to_html(outfile_html)

main(inpath = '/Users/snigdhapanigrahi/Documents/Research/Effect_modification/',
     outpath= '/Users/snigdhapanigrahi/Documents/Research/Effect_modification/')

