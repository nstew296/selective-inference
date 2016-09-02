import numpy as np
from regreg.smooth.glm import glm as regreg_glm
import regreg.api as rr


class multiple_views(object):

    def __init__(self, samplers):

        self.samplers=samplers
        self.nviews = len(samplers)
        self.opt_length = 0
        self.data_length = 0
        opt_slice = []
        data_slice = []
        for i in range(self.nviews):
            self.opt_slice.append(slice(self.opt_length, self.opt_length+samplers[i].opt_length))
            self.data_slice.append(slice(self.data_length, self.data_length+samplers[i].data_length))
            self.opt_length += samplers[i].opt_length
            self.data_length +=samplers[i].data_length


    def projection(self, data_state, opt_state):
        new_opt_state = np.zeros_like(opt_state)
        for i in range(self.nviews):
            new_opt_state[self.opt_slice[i]] = \
                self.samplers[i].projection(data_state[self.data_slice[i]], opt_state[self.opt_slice[i]])[1]
        return data_state.copy(), new_opt_state


    def gradient(self, data_state, opt_state):
        data_grad = np.zeros(self.data_length)
        opt_grad = np.zeros(self.opt_length)
        for i in range(self.nviews):
            data_grad[self.data_slice[i]], opt_grad[self.opt_slice[i]] =\
                self.samplers[i].gradient(data_state[self.data_slice[i]], opt_state[self.opt_slice[i]])

        return data_grad, opt_grad