import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from selection.randomized.randomization import base
from selection.randomized.M_estimator import M_estimator
from selection.randomized.multiple_views import multiple_views
from selection.randomized.glm_boot import pairs_bootstrap_glm, bootstrap_cov, glm_group_lasso

from selection.distributions.discrete_family import discrete_family
from selection.sampling.langevin import projected_langevin

def logistic_instance(n=100, p=200, s=7, rho=0.3, snr=7,
                      random_signs=False, df=np.inf,
                      scale=True, center=True):
    """
    A testing instance for the LASSO.
    Design is equi-correlated in the population,
    normalized to have columns of norm 1.

    Parameters
    ----------

    n : int
        Sample size

    p : int
        Number of features

    s : int
        True sparsity

    rho : float
        Equicorrelation value (must be in interval [0,1])

    snr : float
        Size of each coefficient

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive.

    df : int
        Degrees of freedom for noise (from T distribution).

    Returns
    -------

    X : np.float((n,p))
        Design matrix.

    y : np.float(n)
        Response vector.

    beta : np.float(p)
        True coefficients.

    active : np.int(s)
        Non-zero pattern.

    """

    X = (np.sqrt(1-rho) * np.random.standard_normal((n,p)) + 
        np.sqrt(rho) * np.random.standard_normal(n)[:,None])
    if center:
        X -= X.mean(0)[None,:]
    if scale:
        X /= X.std(0)[None,:]
    
    X /= np.sqrt(n)
    beta = np.zeros(p) 
    beta[:s] = snr
    if random_signs:
        beta[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)

    active = np.zeros(p, np.bool)
    active[:s] = True

    # noise model

    def _noise(n, df=np.inf):
        if df == np.inf:
            return np.random.standard_normal(n)
        else:
            sd_t = np.std(tdist.rvs(df,size=50000))
            return tdist.rvs(df, size=n) / sd_t

    eta = linpred = np.dot(X, beta) 
    pi = np.exp(eta) / (1 + np.exp(eta))

    Y = np.random.binomial(1, pi)
    return X, Y, beta, np.nonzero(active)[0]


def test_logistic_selected_inactive_coordinate(seed=None):
    s, n, p = 5, 200, 20 

    randomization = base.laplace((p,), scale=1.)
    if seed is not None:
        np.random.seed(seed)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=10, scale=False)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    if seed is not None:
        np.random.seed(seed)
    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    print lam
    # our randomization

    np.random.seed(seed)
    M_est1 = glm_group_lasso(loss, epsilon, penalty, randomization)

    mv = multiple_views([M_est1])
    mv.solve()

    active = M_est1.overall
    nactive = active.sum()
    if set(nonzero).issubset(np.nonzero(active)[0]):

        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]
        idx = I[0]
        boot_target, target_observed = pairs_bootstrap_glm(loss, active, inactive=M_est1.inactive)

        def null_target(indices):
            result = boot_target(indices)
            return np.hstack([result[idx], result[nactive:]])

        null_observed = np.zeros(M_est1.inactive.sum() + 1)
        null_observed[0] = target_observed[idx]

        # the null_observed[1:] is only used as a
        # starting point for chain -- could be 0
        null_observed[1:] = target_observed[nactive:]

        sampler = lambda : np.random.choice(n, size=(n,), replace=True)

        mv.setup_sampler(sampler, null_target, null_observed, target_set=[0])

        target_langevin = projected_langevin(mv.observed_state.copy(),
                                             mv.gradient,
                                             mv.projection,
                                             2. / (null_observed.shape[0] + p))


        Langevin_steps = 30000
        burning = 20000
        samples = []
        for i in range(Langevin_steps):
            if (i>=burning):
                target_langevin.next()
                samples.append(target_langevin.state[mv.target_slice].copy())

        test_stat = lambda x: x[0]
        observed = test_stat(null_observed)
        sample_test_stat = np.array([test_stat(x) for x in samples])

        family = discrete_family(sample_test_stat, np.ones_like(sample_test_stat))
        pval = np.clip(family.ccdf(0, observed), 0, 1)
        pval = 2 * min(pval, 1 - pval)
        print "pvalue", pval
        return pval

def test_logistic_selected_inactive_coordinate_burnin(seed=None):
    s, n, p = 5, 200, 20 

    randomization = base.laplace((p,), scale=1.)
    if seed is not None:
        np.random.seed(seed)
    X, y, beta, _ = logistic_instance(n=n, p=p, s=s, rho=0.1, snr=10)

    nonzero = np.where(beta)[0]
    lam_frac = 1.

    loss = rr.glm.logistic(X, y)
    epsilon = 1.

    if seed is not None:
        np.random.seed(seed)
    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.binomial(1, 1. / 2, (n, 10000)))).max(0))
    W = np.ones(p)*lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    print lam
    # our randomization

    np.random.seed(seed)
    M_est1 = glm_group_lasso(loss, epsilon, penalty, randomization)

    mv = multiple_views([M_est1])
    mv.solve()

    active = M_est1.overall
    nactive = active.sum()
    if set(nonzero).issubset(np.nonzero(active)[0]):

        active_set = np.nonzero(active)[0]
        inactive_selected = I = [i for i in np.arange(active_set.shape[0]) if active_set[i] not in nonzero]
        idx = I[0]
        boot_target, target_observed = pairs_bootstrap_glm(loss, active, inactive=M_est1.inactive)

        def null_target(indices):
            result = boot_target(indices)
            return np.hstack([result[idx], result[nactive:]])

        null_observed = np.zeros(M_est1.inactive.sum() + 1)
        null_observed[0] = target_observed[idx]

        # the null_observed[1:] is only used as a
        # starting point for chain -- could be 0
        null_observed[1:] = target_observed[nactive:]

        sampler = lambda : np.random.choice(n, size=(n,), replace=True)

        mv.setup_sampler(sampler, null_target, null_observed, target_set=[0])

        target_langevin = projected_langevin(mv.observed_state.copy(),
                                             mv.gradient,
                                             mv.projection,
                                             2. / (null_observed.shape[0] + p))


        Langevin_steps = 30000
        burning = 20000
        samples = []
        for i in range(Langevin_steps):
            target_langevin.next()
            if (i>=burning):
                samples.append(target_langevin.state[mv.target_slice].copy())

        test_stat = lambda x: x[0]
        observed = test_stat(null_observed)
        sample_test_stat = np.array([test_stat(x) for x in samples])

        family = discrete_family(sample_test_stat, np.ones_like(sample_test_stat))
        pval = np.clip(family.ccdf(0, observed), 0, 1)
        pval = 2 * min(pval, 1 - pval)
        print "pvalue", pval
        return pval
