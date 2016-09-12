"""
Projected Langevin sampler of `http://arxiv.org/abs/1507.02564`_
"""

import numpy as np
from scipy.stats import norm as ndist

class projected_langevin(object):

    def __init__(self,
                 initial_opt_state,
                 initial_data_state,
                 gradient_map,
                 projection_map,
                 data_transform,
                 stepsize):

        (self.opt_state,
         self.data_state,
         self.gradient_map,
         self.projection_map,
         self.data_transform,
         self.stepsize) = (np.copy(initial_opt_state),
                           np.copy(initial_data_state),
                           gradient_map,
                           projection_map,
                           data_transform,
                           stepsize)

        self.num_opt_var = self.opt_state.shape[0]
        self.num_data_var = self.data_state.shape[0]
        self._sqrt_step = np.sqrt(self.stepsize)
        self._noise = ndist(loc=0, scale=1)

    def __iter__(self):
        return self

    def next(self):
        while True:

            data_gradient, opt_gradient = self.gradient_map(self.data_state, self.opt_state, self.data_transform)
            data_candidate = (self.data_state
                             + 0.5 * self.stepsize * data_gradient
                             + self._noise.rvs(self.num_data_var) * self._sqrt_step)
            #data_candidate = self.data_transform(data_candidate)
            proj_opt_arg = (self.opt_state
                            + 0.5 * self.stepsize * opt_gradient
                            + self._noise.rvs(self.num_opt_var) * self._sqrt_step)

            opt_candidate = self.projection_map(proj_opt_arg)
            if not np.all(np.isfinite(self.gradient_map(data_candidate, opt_candidate, self.data_transform))):
                print data_candidate, opt_candidate, self._sqrt_step
                self._sqrt_step *= 0.8
            else:
                self.data_state[:] = data_candidate
                self.opt_state[:] = opt_candidate
                break
