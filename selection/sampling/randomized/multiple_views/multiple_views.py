import numpy as np


class multiple_views(object):

    def __init__(self, samplers, data_construct_map, data_reconstruct_map):

        self.samplers = samplers
        self.data_construct_map = data_construct_map
        self.data_reconstruct_map = data_reconstruct_map
        self.nviews = len(samplers)
        self.opt_length, self.data_length = 0, 0
        self.opt_slice, self.data_slice = [], []

        for i in range(self.nviews):
            self.opt_slice.append(slice(self.opt_length, self.opt_length+samplers[i].opt_length))
            self.data_slice.append(slice(self.data_length, self.data_length+samplers[i].data_length))
            self.opt_length += samplers[i].opt_length
            self.data_length +=samplers[i].data_length

        _init_reconstruct_data_state = np.zeros(self.data_length)
        self.init_opt_state = np.zeros(self.opt_length)
        for i in range(self.nviews):
            self.init_opt_state[self.opt_slice[i]] = samplers[i].init_opt_state
            _init_reconstruct_data_state[self.data_slice[i]] = samplers[i].init_data_state

        self.data_state = np.dot(data_construct_map, _init_reconstruct_data_state)


    def projection(self, data_state, opt_state):
        _data_reconstruct = np.dot(self.data_reconstruct_map, data_state)
        new_opt_state = np.zeros_like(opt_state)
        for i in range(self.nviews):
            new_opt_state[self.opt_slice[i]] = \
                self.samplers[i].projection(_data_reconstruct[self.data_slice[i]], opt_state[self.opt_slice[i]])[1]
        return data_state.copy(), new_opt_state


    def gradient(self, data_state, opt_state):
        opt_grad = np.zeros(self.opt_length)
        _data_reconstruct = np.dot(self.data_reconstruct_map, data_state)
        _reconstruct_data_grad = np.zeros(self.data_length)
        for i in range(self.nviews):
            _reconstruct_data_grad[self.data_slice[i]], opt_grad[self.opt_slice[i]] =\
                self.samplers[i].gradient(_data_reconstruct[self.data_slice[i]], opt_state[self.opt_slice[i]])

        data_grad = np.dot(self.data_construct_map, _reconstruct_data_grad)
        return data_grad, opt_grad


