import numpy as np


class multiple_views(object):

    def __init__(self, samplers, data_construct_map, data_reconstruct_map):

        (self.samplers,
        self.data_construct_map,
        self.data_reconstruct_map) = (samplers,
                                      data_construct_map,
                                      data_reconstruct_map)

        self.nviews = len(self.samplers)


    def solve(self):

        self.initial_soln=[]
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

        _init_reconstruct_data_state = np.zeros(self.data_length)
        self.init_opt_state = np.zeros(self.opt_length)
        for i in range(self.nviews):
            self.init_opt_state[self.opt_slice[i]] = self.samplers[i].init_opt_state
            _init_reconstruct_data_state[self.data_slice[i]] = self.samplers[i].init_data_state

        self.data_state = self.data_construct_map.affine_map(init_reconstruct_data_state)


    def projection(self, opt_state):
        new_opt_state = np.zeros_like(opt_state)
        for i in range(self.nviews):
            new_opt_state[self.opt_slice[i]] = self.samplers[i].projection(opt_state[self.opt_slice[i]])
        return new_opt_state


    def gradient(self, data_state, opt_state):
        opt_grad, data_grad = np.zeros_like(data_state), np.zeros_like(opt_state)
        _data_reconstruct = self.data_reconstruct_map.affine_map(data_state)
        _reconstruct_data_grad = np.zeros(self.data_length)
        for i in range(self.nviews):
            data_transform =
            _reconstruct_data_grad[self.data_slice[i]], opt_grad[self.opt_slice[i]] =\
                self.samplers[i].gradient(_data_reconstruct[self.data_slice[i]], data_transform,opt_state[self.opt_slice[i]])

        data_grad = self.data_construct_map.affine_map(_reconstruct_data_grad)
        return data_grad, opt_grad


