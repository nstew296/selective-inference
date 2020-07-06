import numpy as np

import regreg.api as rr
from selectinf.randomized.query import gaussian_query
from selectinf.randomized.randomization import randomization

class multi_task_lasso(gaussian_query):

    def __init__(self,
                 loglike_dict,
                 ridge_term_list,
                 randomizer_list,
                 lam = 0,
                 perturb_matrix=None):

        self.loglike_dict = loglike_dict
        self.nfeature_vec = [self.loglike_dict[key].shape[0] for key in loglike_dict]
        self.lam = lam
        self.feature_weights_list = np.full((self.nfeature_vec[0],),lam)
        self.penalty = rr.weighted_l1norm(self.feature_weights_list, lagrange=1.)

        self.K = len(loglike_dict)
        self.ridge_term_list = ridge_term_list
        self._initial_omega = perturb_matrix  # random perturbation
        self.randomizer = randomizer_list

    def _solve_randomized_problem(self,
                                  perturb_matrix=None,
                                  solve_args={'tol': 1.e-12, 'min_its': 50}):

        # take a new perturbation if supplied
        if perturb_matrix is not None:
            self._initial_omega = perturb_matrix
        if self._initial_omega is None:
            self._initial_omega = np.array([[self.randomizer[i].sample() for i in range(self.K)]]).reshape(self.nfeature_vec[0],-1)
        quad_list = [rr.identity_quadratic(self.ridge_term_list[i],
                                     0,
                                     -self._initial_omega[:,i],
                                     0) for i in range(self.K)]

        problem_list = [rr.simple_problem(self.loglike_dict[i], self.penalty) for i in range(self.K)]


        initial_soln = np.array([problem_list[i].solve(quad_list[i], **solve_args) for i in range(self.K)])
        initial_subgrad = np.array([-(self.loglike_dict[i].smooth_objective(initial_soln[i,:],
                                                        'grad') +
                            quad_list[i].objective(initial_soln[i,:], 'grad')) for i in range(self.K)])

        return initial_soln, initial_subgrad

    def get_solution(self, num_iter=50):
        initial_solution = self._solve_randomized_problem()
        beta = initial_solution[0].transpose()
        Lambda = self.lam

        for iteration in range(num_iter):
            sum_all_tasks = np.sum(np.absolute(beta), axis=1)
            penalty_weight = 1 / np.maximum(np.sqrt(sum_all_tasks), 10 ** -10)
            self.feature_weights_list = Lambda * penalty_weight
            self.penalty = rr.weighted_l1norm(self.feature_weights_list, lagrange=1.)
            update = self._solve_randomized_problem()
            beta = update[0].transpose()
            subgrad = update[1].transpose()

        return (beta, subgrad)

    @staticmethod
    def gaussian(data_dict,
        quadratic_list,
        ridge_term_list,
        randomizer_scale_list,
        sigma = 1.,
        Lambda = 0):

        K = len(data_dict)

        loglike_dict = {i: rr.glm.gaussian(data_dict[i]['X'],
                        data_dict[i]['Y'],
                        coef=1. / sigma ** 2,
                        quadratic=quadratic_list[i]) for i in range(K)}
        n_vec = [data_dict[i]['X'].shape[0] for i in range(K)]
        p_vec = [data_dict[i]['X'].shape[1] for i in range(K)]

        mean_diag_list = [np.mean((data_dict[i]['X'] ** 2).sum(0)) for i in range(K)]
        for i in range(len(ridge_term_list)):
            if ridge_term_list[i] is None:
                ridge_term_list[i] = np.std(data_dict[i]['Y']) * np.sqrt(mean_diag_list[i]) / np.sqrt(n_vec[i] - 1)

        for i in range(len(randomizer_scale_list)):
            if randomizer_scale_list[i] is None:
                randomizer_scale_list[i] = np.sqrt(mean_diag_list[i]) * 0.5 * np.std(data_dict[i]['Y']) * np.sqrt(n_vec[i] / (n_vec[i] - 1.))


        randomizer_list = [randomization.isotropic_gaussian((p_vec[i],), randomizer_scale_list[i]) for i in range(K)]

        return multi_task_lasso(loglike_dict,
                         ridge_term_list,
                         randomizer_list, lam=Lambda)


def main():
    K = 4
    n = (50, 50, 50,50,50)
    p = 5
    beta = np.random.random((p, K))
    global_sparsity_rate = .2
    task_sparsity_rate = .2
    global_zeros = np.random.choice(p,int(round(global_sparsity_rate*p)))
    beta[global_zeros,:] = np.zeros((K,))
    for i in np.delete(range(p),global_zeros):
        beta[i,np.random.choice(K,int(round(task_sparsity_rate*K)))] = 0
    print(beta)


    none_list = [None for i in range(K)]
    penalty = np.array([0.01 for i in range(p)])
    print(penalty)

    list_X = [np.random.random((n[i], p)) for i in range(K)]
    list_Y = [np.dot(list_X[i], beta[:, i]) for i in range(K)]
    data = {i: {'Y': list_Y[i], 'X': list_X[i]} for i in range(K)}
    multi_lasso = multi_task_lasso.gaussian(data,quadratic_list=none_list,ridge_term_list=none_list, randomizer_scale_list= none_list,Lambda=1.25)
    print(beta, multi_lasso.get_solution())

if __name__ == "__main__":
    main()