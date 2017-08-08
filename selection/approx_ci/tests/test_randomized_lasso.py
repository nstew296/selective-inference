from __future__ import print_function
import sys

import numpy as np
import regreg.api as rr
from selection.tests.instance import gaussian_instance
from selection.approx_ci.randomized_lasso import M_estimator_map, approximate_conditional_density_2stage

def test_approximate_inference(X,
                               y,
                               true_mean,
                               sigma,
                               seed_n = 0,
                               lam_frac = 1.,
                               loss='gaussian'):

    from selection.api import randomization

    n, p = X.shape
    np.random.seed(seed_n)
    if loss == "gaussian":
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    randomization = randomization.isotropic_gaussian((p,), scale=1.)
    M_est = M_estimator_map(loss, epsilon, penalty, randomization)

    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)
    sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")
    sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")
    sys.stderr.write("Observed target" + str(M_est.target_observed) + "\n")

    if nactive == 0:
        return None

    else:
        true_vec = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(true_mean)

        sys.stderr.write("True target to be covered" + str(true_vec) + "\n")

        ci = approximate_conditional_density_2stage(M_est)
        ci.solve_approx()

        ci_sel = np.zeros((nactive, 2))
        sel_MLE = np.zeros(nactive)
        sel_length = np.zeros(nactive)

        for j in range(nactive):
            ci_sel[j, :] = np.array(ci.approximate_ci(j))
            sel_MLE[j] = ci.approx_MLE_solver(j, step=1, nstep=150)[0]
            sel_length[j] = ci_sel[j, 1] - ci_sel[j, 0]

        sel_covered = np.zeros(nactive, np.bool)
        sel_risk = np.zeros(nactive)

        for j in range(nactive):

            sel_risk[j] = (sel_MLE[j] - true_vec[j]) ** 2.

            if (ci_sel[j, 0] <= true_vec[j]) and (ci_sel[j, 1] >= true_vec[j]):
                sel_covered[j] = 1

        print("lengths", sel_length.sum()/nactive)
        print("selective intervals", ci_sel.T)
        print("risks", sel_risk.sum() / nactive)

        return np.transpose(np.vstack((ci_sel[:, 0],
                                       ci_sel[:, 1],
                                       sel_MLE,
                                       sel_covered,
                                       sel_risk)))


X, y, beta, nonzero, sigma = gaussian_instance(n=200, p=500, s=10, rho=0., snr=3., sigma=1.)
true_mean = X.dot(beta)
test = test_approximate_inference(X,
                                  y,
                                  true_mean,
                                  sigma,
                                  seed_n = 0,
                                  lam_frac = 1.,
                                  loss='gaussian')
print(test)