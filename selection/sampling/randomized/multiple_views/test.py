import numpy as np
import regreg.api as rr
from regreg.smooth.glm import glm as regreg_glm, logistic_loglike
import selection.sampling.randomized.api as randomized
from selection.sampling.langevin_new import projected_langevin

from selection.distributions.discrete_family import discrete_family
from scipy.stats import norm as ndist, percentileofscore
from selection.algorithms.randomized import logistic_instance
from matplotlib import pyplot as plt
from scipy.stats import laplace, probplot, uniform


def test(s=0, n=200, p=10, Langevin_steps=10000, burning=2000):
    """
    """

    randomization = randomized.randomization.base.laplace((p,), scale=0.5)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0, snr=7)
    print 'true_beta', beta
    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = regreg_glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), np.ones(p) * lam)), lagrange=1.)

    M_est = randomized.M_est.glm(loss, epsilon, penalty, randomization)
    M_est.solve()
    if np.sum(M_est.initial_soln==0)==p:
        print "no var selected"

    M_est.setup_sampler()
    cov = M_est.form_covariance(M_est.target_bootstrap)

    print cov
    #print "covariance size", cov.shape

    overall = M_est.overall
    noverall = overall.sum()

    data_transform = rr.linear_transform(np.linalg.inv(cov))
    #objectives = [M_est]
    #multiple_views = multiple_views(objectives, rr.linear_transform(np.identity(p)),
    #                                rr.linear_transform(np.identity(p)))
    #multiple_views.setup_sampler()
    #multiple_sampler = projected_langevin(multiple_views._initial_data_state, multiple_views._initial_opt_state,
    #                                      multiple_views.gradient, multiple_views.projection, data_transform,
    #                                      stepsize=1. / p)

    #multiple_sampler.next()

    result = []
    for _ in range(10):
        indices = np.random.choice(n, size=(n,), replace=True)
        result.append(M_est.bootstrap_score(indices))

    #print(np.array(result).shape)

    ndata = p
    data0 = M_est._initial_data_state.copy()

    #null = []
    #alt = []

    overall_set = np.where(overall)[0]

    print "true nonzero ", nonzero, "active set", overall_set
    pval = 0
    if set(nonzero).issubset(overall_set):
        #for j, idx in enumerate(overall_set):

            #eta = np.zeros(noverall)
            #eta[j] = 1
            #sigma_eta_sq = Sigma[j,j]

            #linear_part = np.zeros((ndata, ndata))
            #linear_part[:noverall,:noverall]  = np.identity(noverall) - (np.outer(np.dot(Sigma, eta), eta) / sigma_eta_sq)
            #saturated model
            #linear_part[nactive:,nactive:] = np.identity(ndata - nactive)

            #linear_part = L

            #P = np.dot(linear_part.T, np.linalg.pinv(linear_part).T)
            #I = np.identity(linear_part.shape[1])
            #R = I - P



            sampler = projected_langevin(M_est._initial_data_state, M_est._initial_opt_state,
                                         M_est.gradient, M_est.projection, data_transform,
                                         stepsize=1. / p)
            samples = []

            for i in range(Langevin_steps):
                if (i>burning):
                    sampler.next()
                    samples.append(sampler.data_state.copy())

            samples = np.array(samples)
            data_samples = samples[:, :ndata]

            #eta1 = np.zeros(ndata)
            #eta1[:noverall] = eta
            pop = [np.linalg.norm(z[:noverall]) for z in data_samples]
            obs = np.linalg.norm(data0[:noverall])

            fam = discrete_family(pop, np.ones_like(pop))
            pval = fam.cdf(0, obs)
            pval = 2 * min(pval, 1-pval)
            print "observed: ", obs, "p value: ", pval

            #if pval < 0.0001:
            #    print obs, pval, np.percentile(pop, [0.2,0.4,0.6,0.8,1.0])
            #if idx in nonzero:
            #    alt.append(pval)
            #else:
            #    null.append(pval)
    return pval

if __name__ == "__main__":

    P0, PA = [], []
    plt.figure()
    plt.ion()

    for i in range(20):
        print "iteration", i
        p0 = test()
        P0.append(p0) #PA.extend(pA)
        plt.clf()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        probplot(P0, dist=uniform, sparams=(0, 1), plot=plt,fit=False)
        plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
        plt.pause(0.01)


    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)
    plt.figure()
    probplot(P0, dist=uniform, sparams=(0,1), plot=plt, fit=True)
    plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
    plt.suptitle("Logistic")

    while True:
        plt.pause(0.05)

plt.show()
