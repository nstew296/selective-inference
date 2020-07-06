from sklearn.metrics import mean_squared_error
from selectinf.randomized.lasso import lasso
import numpy as np

import regreg.api as rr

from selectinf.algorithms.sqrt_lasso import solve_sqrt_lasso, choose_lambda

from selectinf.randomized.query import gaussian_query

from selectinf.randomized.randomization import randomization
from selectinf.base import restricted_estimator
from selectinf.algorithms.debiased_lasso import (debiasing_matrix,
                                         pseudoinverse_debiasing_matrix)

class multi_task_lasso(gaussian_query):

    def __init__(self,
                 loglike_dict,
                 feature_weight_list,
                 ridge_term_list,
                 randomizer_list,
                 perturb_matrix=None):

        self.feature_weight_matix = feature_weight_list
        self.loglike_dict = loglike_dict
        self.nfeature_vec = p_vec = [self.loglike_dict[key].shape[0] for key in loglike_dict]

        self.K = K = len(loglike_dict)
        self.ridge_term_list = ridge_term_list
        self.penalty = rr.weighted_l1norm(feature_weight_list, lagrange=1.)

        self._initial_omega = perturb_matrix  # random perturbation

        self.randomizer = randomizer_list

    def _solve_randomized_problem(self,
                                  perturb_matrix=None,
                                  solve_args={'tol': 1.e-12, 'min_its': 50}):

        # take a new perturbation if supplied
        if perturb_matrix is not None:
            self._initial_omega = perturb_matrix
        if self._initial_omega is None:
            self._initial_omega = np.array([[self.randomizer[i].sample() for i in range(self.K)]]).reshape(2,-1)
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

def gaussian(data_dict,
    feature_weights_list,
    quadratic_list,
    ridge_term_list,
    randomizer_scale_list,
    sigma = 1.):

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
                     np.asarray(feature_weights_list) / sigma ** 2,
                     ridge_term_list,
                     randomizer_list)


def choose_lambda(data, data_cv, beta_0, num_iter=20):
    mse_list = []
    beta = beta_0
    lambda_list = [0,.001,.01,.1,1]
    for parameter in lambda_list:

        for iteration in range(num_iter):
            print(beta)
            print('chooselambda',iteration)
            sum_all_tasks = np.sum(np.absolute(beta), axis=1)
            print('sum_all_tasks',sum_all_tasks)
            penalty_weight = 1 / np.maximum(np.sqrt(sum_all_tasks), 10 ** -10)
            penalty = parameter * penalty_weight
            mse = 0
            none_list = [None for i in range(len(data))]
            beta = multi_task_lasso._solve_randomized_problem(gaussian(data,feature_weights_list=penalty,quadratic_list=none_list,ridge_term_list=none_list, randomizer_scale_list= none_list))[0]
            beta = beta.transpose()

        mse_per_task = [mean_squared_error(data_cv[i]['Y'], np.dot(data_cv[i]['X'], beta[:, i])) for i in data_cv]
        mse_list.append(np.sum(mse_per_task))

    one_se = np.std(mse_list) / np.sqrt(len(mse_list))
    min_mse = min(mse_list)
    argmin = np.argmin(abs(mse_list - min_mse - one_se))
    best_lambda = lambda_list[int(argmin)]
    return best_lambda


def get_solution(data, data_cv, beta_0, num_iter=50):
    Lambda = choose_lambda(data, data_cv, beta_0, num_iter)
    print('lambda',Lambda)
    beta = beta_0

    for iteration in range(num_iter):
        sum_all_tasks = np.sum(np.absolute(beta), axis=1)
        penalty_weight = 1 / np.maximum(np.sqrt(sum_all_tasks), 10 ** -10)
        penalty = Lambda * penalty_weight
        none_list = [None for i in range(len(data))]
        update = multi_task_lasso._solve_randomized_problem(gaussian(data,feature_weights_list=penalty,quadratic_list=none_list,ridge_term_list=none_list, randomizer_scale_list= none_list))
        beta = update[0].transpose()
        subgrad = update[1].transpose()

    return (beta,subgrad)

def main():
    K = 3
    n = (3, 3, 3)
    p = 2
    beta = np.random.random((p, K))
    #print(beta)

    list_X = [np.random.random((n[i], p)) for i in range(K)]
    list_Y = [np.dot(list_X[i], beta[:, i]) for i in range(K)]
    list_X_validate = [np.random.random((n[i], p)) for i in range(K)]
    list_Y_validate = [np.dot(list_X[i], beta[:, i]) for i in range(K)]
    data = {i: {'Y': list_Y[i], 'X': list_X[i]} for i in range(K)}
    data_cv = {i: {'Y': list_Y_validate[i], 'X': list_X_validate[i]} for i in range(K)}

    print(beta,get_solution(data, data_cv, beta))

if __name__ == "__main__":
    main()