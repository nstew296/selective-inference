import numpy as np
import regreg.api as rr
from regreg.smooth.glm import glm as regreg_glm, logistic_loglike
import selection.sampling.randomized.api as randomized
from selection.sampling.langevin_new import projected_langevin
from selection.algorithms.randomized import logistic_instance
from selection.distributions.discrete_family import discrete_family
from matplotlib import pyplot as plt
from scipy.stats import laplace, probplot, uniform


class multiple_views(object):

    def __init__(self, objectives):

        self.objectives = objectives
        self.nviews = len(self.objectives)
        # print "obj num", self.nviews
        self.model=1

    def solve(self):
        self.initial_soln = []
        for i in range(self.nviews):
            self.objectives[i].solve()
            self.initial_soln.append(self.objectives[i].initial_soln)
            if self.objectives[i].model==0:
                self.model = 0


    def setup_sampler(self):

        self.num_opt_var = 0
        self.opt_slice = []

        for i in range(self.nviews):
            self.objectives[i].setup_sampler()
            self.opt_slice.append(slice(self.num_opt_var, self.num_opt_var+self.objectives[i].num_opt_var))
            self.num_opt_var += self.objectives[i].num_opt_var

        self._initial_opt_state = np.zeros(self.num_opt_var)

        for i in range(self.nviews):
            self._initial_opt_state[self.opt_slice[i]] = self.objectives[i]._initial_opt_state


    def projection(self, opt_state):
        new_opt_state = np.zeros_like(opt_state)
        for i in range(self.nviews):
            new_opt_state[self.opt_slice[i]] = self.objectives[i].projection(opt_state[self.opt_slice[i]])
        return new_opt_state


    def gradient(self, data_state, data_transform, opt_state):

        data_grad, opt_grad = np.zeros_like(data_state), np.zeros_like(opt_state)

        for i in range(self.nviews):
            data_grad_curr, opt_grad[self.opt_slice[i]] = \
                self.objectives[i].gradient(data_state, data_transform[i], opt_state[self.opt_slice[i]])
            data_grad += data_grad_curr.copy()

        return data_grad, opt_grad



def test():

    from selection.algorithms.randomized import logistic_instance
    from selection.sampling.langevin import projected_langevin

    s, n, p = 0, 200, 20

    randomization = randomized.randomization.base.laplace((p,), scale=0.5)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0, snr=7)
    # print 'true_beta', beta
    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = regreg_glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), np.ones(p)*lam)), lagrange=1.)

    # first randomization
    M_est = randomized.M_est.glm(loss, epsilon, penalty, randomization)
    # second randomization
    M_est2 = randomized.M_est.glm(loss, epsilon, penalty, randomization)
    #multiple views
    mv = multiple_views([M_est, M_est2])
    mv.solve()
    if mv.model==0: # checks if there exists a model that didn't select anything
        return -1
    mv.setup_sampler()
    # for exposition, we will just take
    # the target from first randomization
    # should really do something different

    cov = M_est.form_covariance(M_est.bootstrap_target)
    cov2 = M_est2.form_covariance(M_est.bootstrap_target)
    target_cov = M_est.form_target_cov()

    #print cov.shape, target_cov.shape

    target_initial = M_est._initial_score_state[:M_est.active.sum()]

    # for second coefficient
    A1, b1 = M_est.condition(cov[0], target_cov[0,0], target_initial[0])
    A2, b2 = M_est2.condition(cov2[0], target_cov[0,0], target_initial[0])

    target_inv_cov = 1. / target_cov[0,0]

    obs = target_initial[0].copy()
    initial_state = np.hstack([target_initial[0],
                               mv._initial_opt_state])

    target_slice = slice(0,1)
    opt_slice = slice(1, 2*p+1)

    data_transform = [(A1,b1), (A2,b2)]


    def target_gradient(state):
        # with many samplers, we will add up the `target_slice` component
        # many target_grads
        # and only once do the Gaussian addition of full_grad

        target = state[target_slice]
        opt_state = state[opt_slice]
        target_grad = mv.gradient(target, data_transform, opt_state)

        full_grad = np.zeros_like(state)
        full_grad[target_slice] = target_grad[0]
        full_grad[opt_slice] = target_grad[1]

        full_grad[target_slice] -= target * target_inv_cov

        return full_grad

    def target_projection(state):
        opt_state = state[opt_slice]
        state[opt_slice] = M_est.projection(opt_state)
        return state

    target_langevin = projected_langevin(initial_state.copy(),
                                         target_gradient,
                                         target_projection,
                                         1. / p)

    Langevin_steps = 10000
    burning = 2000
    samples = []
    for i in range(Langevin_steps):
        if (i>burning):
            target_langevin.next()
            samples.append(target_langevin.state[target_slice].copy())

    samples = np.array(samples)
    data_samples = samples[:, 0]
    #obs = target_initial[0]

    fam = discrete_family(data_samples, np.ones_like(data_samples))
    pval = fam.cdf(0, obs)
    pval = 2 * min(pval, 1 - pval)

    print "pvalue", pval
    return pval


if __name__ == "__main__":

    pvalues = []
    for i in range(10):
        print "iteration", i
        pval = test()
        if pval >-1:
            pvalues.append(pval)

    plt.clf()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    probplot(pvalues, dist=uniform, sparams=(0, 1), plot=plt, fit=False)
    plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
    plt.pause(0.01)

    while True:
        plt.pause(0.05)
    plt.show()


