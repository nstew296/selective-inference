import numpy as np
import regreg.api as rr
from regreg.smooth.glm import glm as regreg_glm, logistic_loglike
import selection.sampling.randomized.api as randomized
from selection.sampling.langevin_new import projected_langevin


class multiple_views(object):

    def __init__(self, samplers, data_construct_map, data_reconstruct_map):

        (self.samplers,
        self.data_construct_map,
        self.data_reconstruct_map) = (samplers,
                                      data_construct_map,
                                      data_reconstruct_map)

        self.nviews = len(self.samplers)


    def solve(self):

        self.initial_soln = []
        for i in range(self.nviews):
            self.samplers[i].solve()
            self.initial_soln.append(self.samplers[i].initial_soln)


    def setup_sampling(self):

        self.num_opt_var, self.num_data_var = 0, 0
        self.opt_slice, self.data_slice = [], []

        for i in range(self.nviews):
            self.samplers[i].setup_sampler()
            self.opt_slice.append(slice(self.num_opt_var, self.num_opt_var+self.samplers[i].num_opt_var))
            self.data_slice.append(slice(self.num_data_var, self.num_data_var+self.samplers[i].num_data_var))
            self.num_opt_var += self.samplers[i].num_opt_var
            self.num_data_var += self.samplers[i].num_data_var

        _init_reconstruct_data_state = np.zeros(self.num_data_var)
        self.init_opt_state = np.zeros(self.num_opt_var)
        for i in range(self.nviews):
            self.init_opt_state[self.opt_slice[i]] = self.samplers[i].init_opt_state
            _init_reconstruct_data_state[self.data_slice[i]] = self.samplers[i].init_data_state

        self.data_state = self.data_construct_map.affine_map(_init_reconstruct_data_state)


    def projection(self, opt_state):
        new_opt_state = np.zeros_like(opt_state)
        for i in range(self.nviews):
            new_opt_state[self.opt_slice[i]] = self.samplers[i].projection(opt_state[self.opt_slice[i]])
        return new_opt_state


    def gradient(self, data_state, opt_state):
        opt_grad, data_grad = np.zeros_like(data_state), np.zeros_like(opt_state)
        _data_reconstruct = self.data_reconstruct_map.affine_map(data_state)
        _reconstruct_data_grad = np.zeros(self.num_data_var)
        for i in range(self.nviews):
            data = _data_reconstruct[self.data_slice[i]]
            data_transform = rr.linear_transform(np.identity(data.shape[0]))
            _reconstruct_data_grad[self.data_slice[i]], opt_grad[self.opt_slice[i]] =\
                self.samplers[i].gradient(data, data_transform, opt_state[self.opt_slice[i]])

        data_grad = self.data_construct_map.affine_map(_reconstruct_data_grad)
        return data_grad, opt_grad



if __name__ == "__main__":


    from selection.algorithms.randomized import logistic_instance
    #from selection.sampling.randomized.randomization import base

    s, n, p = 5, 200, 20

    randomization = randomized.randomization.base.laplace((p,), scale=0.5)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=7)
    print 'true_beta', beta
    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = regreg_glm.logistic(X, y)
    epsilon = 1.

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), np.ones(p)*lam)), lagrange=1.)

    M_est = randomized.M_est.glm(loss, epsilon, penalty, randomization)
    M_est.solve()
    M_est.setup_sampler()
    cov = M_est.form_covariance(M_est.target_bootstrap)
    print cov.shape
    result = []

    data_transform = rr.linear_transform(np.identity(p))
    sampler = projected_langevin(M_est._initial_data_state, M_est._initial_state,
                                M_est.gradient, M_est.projection, data_transform, stepsize=1./p)

    sampler.next()

    for _ in range(10):
        indices = np.random.choice(n, size=(n,), replace=True)
        result.append(M_est.bootstrap_score(indices))

    print(np.array(result).shape)