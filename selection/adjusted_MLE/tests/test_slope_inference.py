from rpy2.robjects.packages import importr
from rpy2 import robjects
SLOPE = importr('SLOPE')

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import numpy as np, sys
from selection.randomized.slope import slope

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

def test_slope_R(X, Y, W = None, normalize = True, choice_weights = "gaussian", sigma = None):
    robjects.r('''
    slope = function(X, Y, W , normalize, choice_weights, sigma, fdr = NA){
      if(is.na(sigma)){
      sigma=NULL} else{
      sigma = as.matrix(sigma)[1,1]}
      if(is.na(fdr)){
      fdr = 0.1 }
      if(normalize=="TRUE"){
       normalize = TRUE} else{
       normalize = FALSE}
      if(is.na(W))
      {
        if(choice_weights == "gaussian"){
        lambda = "gaussian"} else{
        lambda = "bhq"}
        result = SLOPE(X, Y, fdr = fdr, lambda = lambda, normalize = normalize, sigma = sigma)
       } else{
        result = SLOPE(X, Y, fdr = fdr, lambda = W, normalize = normalize, sigma = sigma)
      }
      return(list(beta = result$beta, E = result$selected, lambda_seq = result$lambda, sigma = result$sigma))
    }''')

    r_slope = robjects.globalenv['slope']

    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_Y = robjects.r.matrix(Y, nrow=n, ncol=1)

    if normalize is True:
        r_normalize = robjects.StrVector('True')
    else:
        r_normalize = robjects.StrVector('False')

    if W is None:
        r_W = robjects.NA_Logical
        if choice_weights is "gaussian":
            r_choice_weights  = robjects.StrVector('gaussian')
        elif choice_weights is "bhq":
            r_choice_weights = robjects.StrVector('bhq')
    else:
        r_W = robjects.r.matrix(W, nrow=p, ncol=1)

    if sigma is None:
        r_sigma = robjects.NA_Logical
    else:
        r_sigma = robjects.r.matrix(sigma, nrow=1, ncol=1)

    result = r_slope(r_X, r_Y, r_W, r_normalize, r_choice_weights, r_sigma)

    return np.asarray(result.rx2('beta')), np.asarray(result.rx2('E')), \
           np.asarray(result.rx2('lambda_seq')), np.asscalar(np.array(result.rx2('sigma')))

def inference_slope(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=2, snr=0.20,
                    randomizer_scale=np.sqrt(0.25), target = "selected", full_dispersion = True):
    while True:
        X, y, X_val, y_val, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho,
                                                        s=s, beta_type=beta_type, snr=snr)
        true_mean = X.dot(beta)

        if full_dispersion:
            dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        else:
            dispersion = None

        sigma_ = np.sqrt(dispersion)
        print("estimated and true sigma", sigma, sigma_)
        y /= sigma_
        beta /= sigma_
        true_mean /= sigma_
        sigma_ = 1.

        r_beta, r_E, r_lambda_seq, r_sigma = test_slope_R(X,
                                                          y,
                                                          W=None,
                                                          normalize=True,
                                                          choice_weights="gaussian",
                                                          sigma=sigma_)
        
        conv = slope.gaussian(X,
                              y,
                              slope_weights= np.sqrt(n) * sigma_ * r_lambda_seq,
                              randomizer_scale= np.sqrt(n)* randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())
        if nonzero.sum() > 0:
            if target == "selected":
                beta_target =  np.linalg.pinv(X[:, nonzero]).dot(true_mean)
            elif target == "full":
                beta_target = beta[nonzero]

            estimate, _, _, pval, intervals, _ = conv.selective_MLE(target=target, dispersion=sigma_)
            coverage = np.mean((beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1]))
            break

    if True:
        return np.vstack((coverage,
                          np.mean(intervals[:, 1] - intervals[:, 0])))

def main():
    ndraw = 50
    output_overall = np.zeros(2)

    target = "selected"
    n, p, rho, s, beta_type, snr = 500, 100, 0., 5, 1, 0.15

    for i in range(ndraw):
        output = inference_slope(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type, snr=snr,
                                 randomizer_scale=np.sqrt(0.25), target=target, full_dispersion=True)
        output_overall += np.squeeze(output)

        sys.stderr.write("overall selective coverage " + str(output_overall[0] / float(i + 1)) + "\n")
        #sys.stderr.write("overall unad coverage " + str(output_overall[8] / float(i + 1)) + "\n" + "\n")

        sys.stderr.write("overall selective length " + str(output_overall[1] / float(i + 1)) + "\n")
        #sys.stderr.write("overall unad length " + str(output_overall[11] / float(i + 1)) + "\n" + "\n")

        sys.stderr.write("iteration completed " + str(i + 1) + "\n")

main()


