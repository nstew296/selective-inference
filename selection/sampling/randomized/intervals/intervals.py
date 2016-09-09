import numpy as np
from selection.sampling.randomized.tests.test_lasso_fixedX_saturated import selection
from selection.sampling.randomized.tests.test_lasso_fixedX_saturated import test_lasso

from selection.sampling.randomized.intervals.estimation import estimation, instance


class intervals(estimation):

    def __init__(self, X, y, active, betaE, cube, epsilon, lam, sigma, tau):
        estimation.__init__(self, X, y, active, betaE, cube, epsilon, lam, sigma, tau)
        estimation.setup_estimation(self)

    def setup_samples(self, ref_vec, samples, observed, variances):
        (self.ref_vec,
         self.samples,
         self.observed,
         self.variances) = (ref_vec,
                            samples,
                            observed,
                            variances)
        self.nsamples = self.samples.shape[1]

    def empirical_exp(self, j, param):
        ref = self.ref_vec[j]
        tilted_samples = np.exp(self.samples[j,:] * np.true_divide(param-ref, 2*self.eta_norm_sq[j]*self.sigma_sq))
        #print self.samples.shape[1]
        return np.sum(tilted_samples)/float(self.nsamples)


    def log_ratio_selection_prob(self, j, param):
        ref = self.ref_vec[j]
        Sigma_inv_mu_param, Sigma_inv_mu_ref = self.Sigma_inv_mu[j].copy(), self.Sigma_inv_mu[j].copy()
        Sigma_inv_mu_param[0] += param / (self.eta_norm_sq[j] * self.sigma_sq)
        mu_param = np.dot(self.Sigma_full[j], Sigma_inv_mu_param)
        Sigma_inv_mu_ref[0] += ref / (self.eta_norm_sq[j] * self.sigma_sq)
        mu_ref = np.dot(self.Sigma_full[j], Sigma_inv_mu_ref)
        log_gaussian_part = (-np.inner(mu_param, Sigma_inv_mu_param)+np.inner(mu_ref, Sigma_inv_mu_ref))/float(2)
        return log_gaussian_part*np.log(self.empirical_exp(j, param, ref))


    def pvalue_by_tilting(self, j, param):
        ref = self.ref_vec[j]
        indicator = np.array(self.samples[j,:] < self.observed[j], dtype =int)
        log_gaussian_tilt = 2*np.array(self.samples[j,:]) * (param - ref) - (param ** 2) + (ref ** 2),
        log_gaussian_tilt /= 2*self.eta_norm_sq[j]*self.sigma_sq
        log_ratio_sel_prob_tilt = self.log_ratio_selection_prob(j, param, ref)
        log_LR = log_gaussian_tilt-log_ratio_sel_prob_tilt
        return np.clip(np.sum(np.multiply(indicator, np.exp(log_LR))) / float(self.nsamples), 0, 1)


    def pvalues_param(self, param_vec):
        pvalues = []
        for j in range(self.nactive):
            pvalues.append(self.pvalue_by_tilting(j, param_vec[j], self.ref_vec[j]))
        return pvalues

    def pvalues_ref(self):
        pvalues = []
        for j in range(self.nactive):
            indicator = np.array(self.samples[j, :] < self.observed[j], dtype=int)
            pvalues.append(np.sum(indicator)/float(self.nsamples))
        return pvalues



def test_intervals(n=100, p=10, s=0):

    tau = 1.
    data_instance = instance(n, p, s)
    X, y, true_beta, nonzero, sigma = data_instance.generate_response()
    random_Z = np.random.standard_normal(p)
    lam, epsilon, active, betaE, cube, initial_soln = selection(X,y, random_Z)
    if lam < 0:
        return None
    int_class = intervals(X, y, active, betaE, cube, epsilon, lam, sigma, tau)
    ref_vec = int_class.mle.copy()
    #ref_vec = np.ones(np.sum(active))/2
    _, _, all_observed, all_variances, all_samples = test_lasso(X, y, nonzero, sigma, lam, epsilon, active, betaE,
                                                                cube, random_Z, beta_reference=ref_vec.copy(),
                                                                randomization_distribution="normal",
                                                                Langevin_steps=20000, burning=2000)

    int_class.setup_samples(ref_vec.copy(), all_samples, all_observed, all_variances)

    pvalues_ref = int_class.pvalues_ref()
    pvalues_param = int_class.pvalues_param(ref_vec.copy())
    print pvalues_param
    return pvalues_ref, pvalues_param



if __name__ == "__main__":
    P0 = []
    P1 = []
    for i in range(50):
        print "iteration", i
        pvalues = test_intervals()
        if pvalues is not None:
            P0.extend(pvalues[1])
            P1.extend(pvalues[0])

    from matplotlib import pyplot as plt
    from scipy.stats import laplace, probplot, uniform
    import statsmodels.api as sm

    fig = plt.figure()
    plot_pvalues0 = fig.add_subplot(121)
    plot_pvalues1 = fig.add_subplot(122)

    P0 = np.asarray(P0, dtype=np.float32)
    ecdf = sm.distributions.ECDF(P0)
    x = np.linspace(min(P0), max(P0))
    y = ecdf(x)
    plot_pvalues0.plot(x, y, '-o', lw=2)
    plot_pvalues0.plot([0, 1], [0, 1], 'k-', lw=2)
    plot_pvalues0.set_title("P values at the truth")
    plot_pvalues0.set_xlim([0, 1])
    plot_pvalues0.set_ylim([0, 1])

    P1 = np.asarray(P1, dtype=np.float32)
    ecdf = sm.distributions.ECDF(P1)
    x = np.linspace(min(P1), max(P1))
    y = ecdf(x)
    plot_pvalues1.plot(x, y, '-o', lw=2)
    plot_pvalues1.plot([0, 1], [0, 1], 'k-', lw=2)
    plot_pvalues1.set_title("P values at the reference")
    plot_pvalues1.set_xlim([0, 1])
    plot_pvalues1.set_ylim([0, 1])

    plt.show()
    plt.savefig("P values from the intervals file.png")
