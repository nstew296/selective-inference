import numpy as np
import regreg.api as rr
from selection.bayesian.credible_intervals import projected_langevin
from selection.bayesian.marginal_screening import selection_probability_objective_ms

class sel_prob_gradient_map_ms(rr.smooth_atom):
    def __init__(self,
                 X,
                 active,
                 active_signs,
                 threshold,
                 generative_X,
                 noise_variance,
                 randomizer,
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.dim = active.sum()

        self.noise_variance = noise_variance

        (self.X, self.active, self.active_signs, self.threshold, self.generative_X, self.noise_variance,
         self.randomizer) = (X, active, active_signs, threshold, generative_X, noise_variance, randomizer)

        rr.smooth_atom.__init__(self,
                                (self.dim,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

        self.p = self.threshold.shape[0]

        w, v = np.linalg.eig(self.X.T.dot(self.X))
        var_half_inv = (v.T.dot(np.diag(np.power(w, -0.5)))).dot(v)
        self.scale_coef = var_half_inv.dot(self.X.T)

    def smooth_objective(self, true_param, mode='both', check_feasibility=False, tol=1.e-6):

        true_param = self.apply_offset(true_param)

        mean_parameter = np.squeeze(self.generative_X.dot(true_param))

        mean = self.scale_coef.dot(mean_parameter)

        sol = selection_probability_objective_ms(self.active,
                                                 self.active_signs,
                                                 self.threshold, # a vector in R^p
                                                 mean,  # in R^p
                                                 self.noise_variance,
                                                 self.randomizer)

        sel_prob_primal = sol.minimize2(nstep=60)[::-1]
        optimal_primal = (sel_prob_primal[1])[:self.p]
        sel_prob_val = -sel_prob_primal[0]
        optimizer = (self.generative_X.T.dot(self.scale_coef.T)).dot(np.true_divide(optimal_primal - mean, self.noise_variance))

        if mode == 'func':
            return sel_prob_val
        elif mode == 'grad':
            return optimizer
        elif mode == 'both':
            return sel_prob_val, optimizer
        else:
            raise ValueError('mode incorrectly specified')


class selective_map_credible_ms(rr.smooth_atom):
    def __init__(self,
                 y,
                 X,
                 active,
                 active_signs,
                 threshold,
                 generative_X,
                 noise_variance,
                 prior_variance,
                 randomizer,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 nstep=10):

        self.param_shape = generative_X.shape[1]

        y = np.squeeze(y)
        self.X = X
        w, v = np.linalg.eig(self.X.T.dot(self.X))
        var_half_inv = (v.T.dot(np.diag(np.power(w, -0.5)))).dot(v)
        scale_coef = var_half_inv.dot(self.X.T)

        scaled_Z = scale_coef.dot(y)

        self.E = active.sum()

        self.generative_X = generative_X

        initial = np.ones(self.E)

        rr.smooth_atom.__init__(self,
                                (self.param_shape,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)


        self.coefs[:] = initial

        self.initial_state = initial

        self.set_likelihood(scaled_Z, noise_variance, scale_coef, generative_X)

        self.set_prior(prior_variance)

        self.log_sel_prob = sel_prob_gradient_map_ms(X,
                                                     active,
                                                     active_signs,
                                                     threshold,
                                                     generative_X,
                                                     noise_variance,
                                                     randomizer)

        self.total_loss = rr.smooth_sum([self.likelihood_loss,
                                         self.log_prior_loss,
                                         self.log_sel_prob])

    def set_likelihood(self, scaled_Z, noise_variance, scale_coef, generative_X):
        likelihood_loss = rr.signal_approximator(scaled_Z, coef=1. / noise_variance)
        self.likelihood_loss = rr.affine_smooth(likelihood_loss, scale_coef.dot(generative_X))

    def set_prior(self, prior_variance):
        self.log_prior_loss = rr.signal_approximator(np.zeros(self.param_shape), coef=1. / prior_variance)

    def smooth_objective(self, true_param, mode='both', check_feasibility=False):

        true_param = self.apply_offset(true_param)

        if mode == 'func':
            f = self.total_loss.smooth_objective(true_param, 'func')
            return self.scale(f)
        elif mode == 'grad':
            g = self.total_loss.smooth_objective(true_param, 'grad')
            return self.scale(g)
        elif mode == 'both':
            f, g = self.total_loss.smooth_objective(true_param, 'both')
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def map_solve_2(self, step=1, nstep=100, tol=1.e-8):

        current = self.coefs[:]
        current_value = np.inf

        objective = lambda u: self.smooth_objective(u, 'func')
        grad = lambda u: self.smooth_objective(u, 'grad')

        for itercount in range(nstep):

            newton_step = grad(current)
                          #* self.noise_variance

            # make sure proposal is a descent
            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)

                if proposed_value <= current_value:
                    break
                step *= 0.5

            # stop if relative decrease is small

            if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value):
                current = proposal
                current_value = proposed_value
                break

            current = proposal
            current_value = proposed_value

            if itercount % 4 == 0:
                step *= 2

        value = objective(current)
        return current, value

    def posterior_samples(self, Langevin_steps = 10000, burnin = 1000):
        state = self.initial_state
        gradient_map = lambda x: -self.smooth_objective(x, 'grad')
        projection_map = lambda x: x
        stepsize = 1. / self.E
        sampler = projected_langevin(state, gradient_map, projection_map, stepsize)

        samples = []

        for i in range(Langevin_steps):
            sampler.next()
            samples.append(sampler.state.copy())
            print i, sampler.state.copy()

        samples = np.array(samples)
        return samples[burnin:, :]