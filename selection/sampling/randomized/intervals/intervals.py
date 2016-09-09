import numpy as np
from selection.sampling.randomized.tests.test_lasso_fixedX_saturated import selection
from selection.sampling.randomized.tests.test_lasso_fixedX_saturated import test_lasso

from selection.sampling.randomized.intervals.estimation import estimation, instance


class intervals(estimation):

    def __init__(self, X, y, initial_soln, cube, epsilon, lam, sigma, tau):
        estimation.__init__(self, X, y, initial_soln, cube, epsilon, lam, sigma, tau)
        estimation.setup_estimation(self)

    def setup_samples(self, samples, observed, variances):
        (self.samples,
         self.observed,
         self.variances) = (samples,
                            observed,
                            variances)

    def empirical_exp(self, j, param, ref):
        tilted_samples = np.exp(self.samples[j,:] * np.true_divide(param-ref, 2*self.eta_norm_sq[j]*self.sigma))
        return np.true_divide(np.sum(tilted_samples), self.samples.shape[1])


    def log_ratio_selection_prob(self, j, param, ref):
        Sigma_inv_mu_param = self.Sigma_inv_mu[j].copy()
        Sigma_inv_mu_param[0] += param / (self.eta_norm_sq[j] * (self.sigma ** 2))
        mu_param = np.dot(self.Sigma_full[j], Sigma_inv_mu_param)
        Sigma_inv_mu_ref = self.Sigma_inv_mu[j].copy()
        Sigma_inv_mu_ref += ref / (self.eta_norm_sq[j] * (self.sigma ** 2))
        mu_ref = np.dot(self.Sigma_full[j], Sigma_inv_mu_ref)
        log_gaussian_part = -np.inner(mu_param, Sigma_inv_mu_param)+np.inner(mu_ref, Sigma_inv_mu_ref)
        return log_gaussian_part*self.empirical_exp(j, param, ref)

    def pvalue_by_tilting(self, j, param, ref):
        indicator = np.array(self.samples[j] < self.observed[j], dtype =int)
        gaussian_tilt = np.true_divide(self.samples[j] * (param - ref) - (param ** 2 - (ref ** 2)), 2 * self.variances[j])
        log_LR = np.multiply(gaussian_tilt, self.log_ratio_selection_prob(j, param, ref))
        return np.clip(np.sum(np.multiply(indicator, np.exp(log_LR))) / float(indicator.shape[0]), 0, 1)

    def pvalues_all(self, param_vec, ref_vector):
        pvalues = []
        for j in range(self.nactive):
            pvalues.append(self.pvalue_by_tilting(j, param_vec[j], ref_vector[j]))
        return pvalues



def test_intervals(n=200, p=10, s=0):
    pvalues = []
    tau = 1.
    data_instance = instance(n, p, s)
    X, y, true_beta, nonzero, sigma = data_instance.generate_response()
    random_Z = np.random.standard_normal(p)
    lam, epsilon, active, betaE, cube, initial_soln = selection(X,y, random_Z)

    int_class = intervals(X, y, initial_soln, cube, epsilon, lam, sigma, tau)

    _, _, all_observed, all_variances, all_samples = test_lasso(X, y, nonzero, sigma, lam, epsilon, active, betaE,
                                                                cube, random_Z, beta_reference=int_class.mle.copy(),
                                                                randomization_distribution="normal",
                                                                Langevin_steps=20000, burning=2000)
    if lam < 0:
            print "no active covariates"
    else:

        int_class.setup_samples(all_samples, all_observed, all_variances)

        pvalues.extend(int_class.pvalues_all(np.zeros(active.sum()), int_class.mle.copy()))
        print pvalues
        return pvalues


if __name__ == "__main__":
    P0 = []
    for i in range(50):
        print "iteration", i
        pvalues = test_intervals()
        if pvalues is not None:
            P0.append(pvalues)

    from matplotlib import pyplot as plt
    from scipy.stats import laplace, probplot, uniform

    plt.figure()
    probplot(P0, dist=uniform, sparams=(0,1), plot=plt, fit=True)
    plt.plot([0,1], color='k', linestyle='-', linewidth=2)
    plt.show()

