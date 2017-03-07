from __future__ import print_function
import sys
import time
import random
import glob
import numpy as np
from selection.tests.instance import gaussian_instance
from selection.bayesian.initial_soln import selection, instance
from selection.randomized.api import randomization
from selection.bayesian.cisEQTLS.Simes_selection import simes_selection
from scipy.stats import norm as normal
from selection.bayesian.cisEQTLS.Simes_selection import BH_q
from selection.bayesian.cisEQTLS.inference_2sels import selection_probability_genes_variants, \
    sel_prob_gradient_map_simes_lasso, selective_inf_simes_lasso
from selection.bayesian.cisEQTLS.inference_per_gene import selection_probability_variants, \
    sel_prob_gradient_map_lasso, selective_inf_lasso

def evaluate_hierarchical_results(result, X, s, snr):
    """
    evaluate the output for hierarhical with ground truth information
    """
    # Inputs:
    #   result: np matrix of results (rows: genes, 
    #           cols: [lasso_sel, ***, unadj_l, unadj_r, adj_l, adj_r, **, **]) 
    #   X: np matrix of the data input
    #   s: integer of value of the signal (the first s variables are signals)
    #   snr: float of value of the snr 
    # Outputs:
    #   None

    if result is None:
        sys.stderr.write("Result is None (likely Lasso did not select any variables)\n")
        return None
        

    FDR = 0.
    power = 0.

    n, p = X.shape
    true_beta = np.zeros(p)
    true_beta[:s] = snr

    discoveries = np.array(result[:, 1], np.bool)

    if true_beta[0] > 0:

        true_discoveries = discoveries[:s].sum()

    else:
        true_discoveries = 0

    false_discoveries = discoveries[s:].sum()
    FDR += false_discoveries / max(float(discoveries.sum()), 1.)
    if s == 0:
        power = 0
    else:
        power += true_discoveries / float(s)

    active_ind = np.array(result[:, 0], np.bool)
    nactive = active_ind.sum()

    projection_active = X[:, active_ind].dot(np.linalg.inv(X[:, active_ind].T.dot(X[:, active_ind])))
    true_val = projection_active.T.dot(X.dot(true_beta))

    coverage_ad = np.zeros(true_val.shape[0])
    coverage_unad = np.zeros(true_val.shape[0])

    adjusted_intervals = np.zeros((2,nactive))
    adjusted_intervals[0,:] = (result[:, 2])[active_ind]
    adjusted_intervals[1,:] = (result[:, 3])[active_ind]

    unadjusted_intervals = np.zeros((2, nactive))
    unadjusted_intervals[0, :] = (result[:, 4])[active_ind]
    unadjusted_intervals[1, :] = (result[:, 5])[active_ind]

    for l in range(nactive):
        if (adjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= adjusted_intervals[1, l]):
            coverage_ad[l] += 1
        if (unadjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= unadjusted_intervals[1, l]):
            coverage_unad[l] += 1

    adjusted_coverage = float(coverage_ad.sum() / nactive)
    unadjusted_coverage = float(coverage_unad.sum() / nactive)

    return adjusted_coverage, unadjusted_coverage, FDR, power


def hierarchical_inference(X,           # input X data
                           y,           # input y data
                           index,       # index of the lead variable that passes simes TODO
                           simes_level, #simes level divided by number of genes
                           pgenes,      # proportion of egenes in total number of genes
                           J,           # rejection list
                           t_0,         # order index of the lead variable that passes simes 
                           T_sign,      # the sign of the test statistic of the lead variable
                           seed_n = 19, # seed to run the procedure
                           bh_level=0.1,# benjamini-hotchburg level to control the FDR
                           selection_method = "single", # method selection
                           lambda_method = "theoretical",
                           just_selection=False):

    np.random.seed(seed_n)
    n, p = X.shape
    T_sign = T_sign * np.ones(1)

    if t_0 == 0:
        threshold = normal.ppf(1. - simes_level / (2. * p)) * np.ones(1)
    else:
        J_card = J.shape[0]
        threshold = np.zeros(J_card + 1)
        threshold[:J_card] = normal.ppf(1. - (simes_level / (2. * p)) * (np.arange(J_card) + 1.))
        threshold[J_card] = normal.ppf(1. - (simes_level / (2. * p)) * t_0)

    random_Z = np.random.standard_normal(p)

    sel = selection(X, y, random_Z, method=lambda_method)

    # terminate selection if lasso did not select any variables
    if sel is None:
        sys.stderr.write("Lasso did not select any variables\n")
        return None

    if just_selection:
        sys.stderr.write("Warning: only returning selection; no inference will be returned\n")
        return sel 

    lam, epsilon, active, betaE, cube, initial_soln = sel
    lagrange = lam * np.ones(p)
    active_sign = np.sign(betaE)
    nactive = active.sum()
    noise_variance = 1.
    randomizer = randomization.isotropic_gaussian((p,), 1.)
    generative_X = X[:, active]
    prior_variance = 1000.

    if selection_method == "double":
        # account for selection and randomlized lasso
        feasible_point = np.append(1, np.fabs(betaE))
        grad_map = sel_prob_gradient_map_simes_lasso(X,
                                                     feasible_point,
                                                     index,
                                                     J,
                                                     active,
                                                     T_sign,
                                                     active_sign,
                                                     lagrange,
                                                     threshold,
                                                     generative_X,
                                                     noise_variance,
                                                     randomizer,
                                                     epsilon)
        inf = selective_inf_simes_lasso(y, grad_map, prior_variance)

    elif selection_method == "single":
        # account for only randomized lasso with a more stringent level
        bh_level = bh_level * pgenes
        feasible_point = np.fabs(betaE)
        grad_map = sel_prob_gradient_map_lasso(X,
                                               feasible_point,
                                               active,
                                               active_sign,
                                               lagrange,
                                               generative_X,
                                               noise_variance,
                                               randomizer,
                                               epsilon)
        inf = selective_inf_lasso(y, grad_map, prior_variance)
    else:
        sys.stderr.write("Method '"+selection_method+"' does not exist (use 'double' or 'single')\n")
        sys.exit(1)

    samples = inf.posterior_samples()

    adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])

    selective_mean = np.mean(samples, axis=0)

    projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))
    M_1 = prior_variance * (X.dot(X.T)) + noise_variance * np.identity(n)
    M_2 = prior_variance * ((X.dot(X.T)).dot(projection_active))
    M_3 = prior_variance * (projection_active.T.dot(X.dot(X.T)).dot(projection_active))
    post_mean = M_2.T.dot(np.linalg.inv(M_1)).dot(y)
    post_var = M_3 - M_2.T.dot(np.linalg.inv(M_1)).dot(M_2)

    unadjusted_intervals = np.vstack([post_mean - 1.65 * (np.sqrt(post_var.diagonal())),
                                      post_mean + 1.65 * (np.sqrt(post_var.diagonal()))])

    nerr = 0.

    active_set = [i for i in xrange(p) if active[i]]
    active_ind = np.zeros(p)
    active_ind[active_set] = 1

    # compute intervals
    ad_lower_credible = np.zeros(p)
    ad_upper_credible = np.zeros(p)
    unad_lower_credible = np.zeros(p)
    unad_upper_credible = np.zeros(p)
    ad_mean = np.zeros(p)
    unad_mean = np.zeros(p)
    for l in xrange(int(nactive)):
        ad_lower_credible[active_set[l]] = adjusted_intervals[0, l]
        ad_upper_credible[active_set[l]] = adjusted_intervals[1, l]
        unad_lower_credible[active_set[l]] = unadjusted_intervals[0, l]
        unad_upper_credible[active_set[l]] = unadjusted_intervals[1, l]
        ad_mean[active_set[l]] = selective_mean[l]
        unad_mean[active_set[l]] = post_mean[l]

    ngrid = 1000
    quantiles = np.zeros((ngrid, nactive))

    for i in xrange(ngrid):
        quantiles[i, :] = np.percentile(samples, (i * 100.) / ngrid, axis=0)

    index_grid = np.argmin(np.abs(quantiles - np.zeros((ngrid, nactive))), axis=0)
    p_value = 2 * np.minimum(np.true_divide(index_grid, ngrid), 1. - np.true_divide(index_grid, ngrid))
    p_BH = BH_q(p_value, bh_level)

    # selection 
    D_BH = np.zeros(p)
    if p_BH is not None:
        for indx in p_BH[1]:
            D_BH[active_set[indx]] = 1

    list_results = np.transpose( np.vstack((active_ind,
                                            D_BH,
                                            ad_lower_credible,
                                            ad_upper_credible,
                                            unad_lower_credible,
                                            unad_upper_credible,
                                            ad_mean,
                                            unad_mean)))

    return list_results

if __name__ == "__main__":
    np.random.seed(0)
    # s = 2
    s = 2
    snr = 5.0
    X, y, true_beta, nonzero, noise_variance = gaussian_instance(n=10, p=4, s=s, sigma=1, rho=0, snr=snr)
    for method in ("single", "double"):
        result = hierarchical_inference(X, y, 0, 0.01, 0.8, J=[], t_0=0, T_sign=1, selection_method = method)
        print(result)
        print(evaluate_hierarchical_results(result, X, s, snr))

