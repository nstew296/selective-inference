import numpy as np
from scipy.stats import norm as ndist

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri

from ...tests.instance import gaussian_instance
from ...tests.decorators import rpy_test_safe

from ..screening import stepup
from ..screening import stepup, stepup_selection
from ..randomization import randomization

@rpy_test_safe()
def test_BH_procedure():

    def BHfilter(pval, q=0.2):
        numpy2ri.activate()
        rpy.r.assign('pval', pval)
        rpy.r.assign('q', q)
        rpy.r('Pval = p.adjust(pval, method="BH")')
        rpy.r('S = which((Pval < q)) - 1')
        S = rpy.r('S')
        numpy2ri.deactivate()
        return np.asarray(S, np.int)

    def BH_cutoff():
        Z = np.random.standard_normal(100)

        BH = stepup.BH(Z,
                       np.identity(100),
                       1.)

        cutoff = BH.stepup_Z / np.sqrt(2)
        return cutoff
    
    BH_cutoffs = BH_cutoff()

    for _ in range(50):
        Z = np.random.standard_normal(100)
        Z[:20] += 3

        np.testing.assert_allclose(sorted(BHfilter(2 * ndist.sf(np.fabs(Z)), q=0.2)),
                                   sorted(stepup_selection(Z, BH_cutoffs)[1]))

def test_BH(n=500, 
            p=100, 
            s=10, 
            sigma=3, 
            rho=0.65, 
            randomizer_scale=np.sqrt(1/9.),
            use_MLE=True,
            marginal=False):

    while True:

        X = gaussian_instance(n=n,
                              p=p,
                              equicorrelated=False,
                              rho=rho)[0]
        W = rho**(np.fabs(np.subtract.outer(np.arange(p), np.arange(p))))
        sqrtW = np.linalg.cholesky(W)
        sigma = 0.5
        Z = np.random.standard_normal(p).dot(sqrtW.T) * sigma
        beta = (2 * np.random.binomial(1, 0.5, size=(p,)) - 1) * np.linspace(4, 5, p) * sigma
        np.random.shuffle(beta)
        beta[s:] = 0
        print(beta)
        np.random.shuffle(beta)

        true_mean = W.dot(beta)
        score = Z + true_mean
        idx = np.arange(p)

        n, p = X.shape

        q = 0.1
        BH_select = stepup.BH(score,
                              W * sigma**2,
                              randomizer_scale * sigma,
                              q=q)

        boundary = BH_select.fit()

        if boundary is not None:
            nonzero = boundary != 0

            if marginal:
                (observed_target, 
                 cov_target, 
                 crosscov_target_score, 
                 alternatives) = BH_select.marginal_targets(nonzero)
            else:
                (observed_target, 
                 cov_target, 
                 crosscov_target_score, 
                 alternatives) = BH_select.multivariate_targets(nonzero, dispersion=sigma**2)
               
            if use_MLE:
                estimate, _, _, pval, intervals, _ = BH_select.selective_MLE(observed_target,
                                                                             cov_target,
                                                                             crosscov_target_score)
                # run summary
            else:
                _, pval, intervals = BH_select.summary(observed_target, 
                                                       cov_target, 
                                                       crosscov_target_score, 
                                                       alternatives,
                                                       compute_intervals=True)

            print(pval)
            if marginal:
                beta_target = true_mean[nonzero]
            else:
                beta_target = beta[nonzero]
            print("beta_target and intervals", beta_target, intervals)
            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])
            print("coverage for selected target", coverage.sum()/float(nonzero.sum()))
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals

def test_both():
    test_BH(marginal=True)
    test_BH(marginal=False)

def main(nsim=500, use_MLE=False):

    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    U = np.linspace(0, 1, 101)
    P0, PA, cover, length_int = [], [], [], []
    for i in range(nsim):
        p0, pA, cover_, intervals = test_BH(use_MLE=use_MLE)

        cover.extend(cover_)
        P0.extend(p0)
        PA.extend(pA)
        print(np.mean(cover),'coverage so far')

        period = 10
        if use_MLE:
            period = 50
        if i % period == 0 and i > 0:
            plt.clf()
            plt.plot(U, sm.distributions.ECDF(P0)(U), 'b', label='null')
            plt.plot(U, sm.distributions.ECDF(PA)(U), 'r', label='alt')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.legend()
            plt.savefig('BH_pvals.pdf')


