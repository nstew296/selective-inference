import numpy as np
import selection.sampling.randomized.api as randomized
from selection.algorithms.randomized import logistic_instance
from selection.sampling.randomized.losses.glm import glm
import regreg.api as rr
from matplotlib import pyplot as plt
from scipy.stats import laplace, probplot, uniform


def test_mv(s=5, n=200, p=20, randomization_scale=1., randomization_dist="laplace"):


    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=7)
    epsilon = 1.
    lam_frac = 1.
    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))

    loss = glm.logistic(X, y)
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), lam * np.ones(p))),
                             lagrange=1.)

    if randomization_dist == "laplace":
        randomization = laplace(loc=0, scale=1.)
        random_Z = randomization.rvs(p)
    if randomization_dist == "logistic":
        random_Z = np.random.logistic(loc=0, scale=1, size=p)
    loss = glm.logistic(X, y)

    problem = rr.simple_problem(loss, penalty)
    random_term = rr.identity_quadratic(epsilon, 0, randomization_scale * random_Z, 0)
    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}

    initial_soln = problem.solve(random_term, **solve_args)
    initial_grad = loss.gradient(initial_soln)

    samplers = []
    group_lasso_sampler = randomized.group_lasso_sampler(loss, initial_soln, epsilon, penalty)
    samplers.append(group_lasso_sampler)
    data_length = samplers[0].data_length
    multiple_views = randomized.multiple_views(samplers, np.identity(data_length), np.identity(data_length))

    return

if __name__ == "__main__":

    P0, PA = [], []
    plt.figure()
    plt.ion()

    for i in range(50):
        print "iteration", i
        p0, pA = test_mv(seed=i)
        P0.extend(p0);
        PA.extend(pA)
        plt.clf()
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        probplot(P0, dist=uniform, sparams=(0, 1), plot=plt, fit=False)
        plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
        plt.pause(0.01)

    while True:
            plt.pause(0.05)

    print "done! mean: ", np.mean(P0), "std: ", np.std(P0)
    plt.figure()
    probplot(P0, dist=uniform, sparams=(0, 1), plot=plt, fit=True)
    plt.plot([0, 1], color='k', linestyle='-', linewidth=2)
    plt.show()