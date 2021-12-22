import numpy as np
from selectinf.randomized.lasso import lasso, selected_targets
from selectinf.tests.instance import gaussian_instance
from scipy.stats import t as tdist
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def _noise(n, df=np.inf):
    if df == np.inf:
        return np.random.standard_normal(n)
    else:
        sd_t = np.std(tdist.rvs(df, size=50000))
    return tdist.rvs(df, size=n) / sd_t

n=1000
p_list = [100,250,500,750]
rho = 0.3
sigma=1.0
signal_fac = np.array([1.0,3.0])
weight = 2.25
randomizer_scale = 0.7
covlist = {j: [] for j in range(len(p_list))}

for j in range(len(p_list)):
    p = p_list[j]
    signal = np.sqrt(signal_fac * 2 * np.log(p))
    s = int(p*0.1*0.6)
    for i in range(500):
        X, Y, beta = gaussian_instance(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=True,
                          rho=rho,
                          sigma=sigma,
                          random_signs=False)[:3]

        idx = np.arange(p)
        sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        #print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

        W = weight * np.ones(p)

        conv = lasso.gaussian(X,
                     Y,
                     W,
                     ridge_term=0.,
                     randomizer_scale=randomizer_scale * 1.0)

        omega = np.asarray([_noise(p)[j] * randomizer_scale*sigma for j in range(p)])

        signs = conv.fit(perturb=omega)
        nonzero = signs != 0

        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(conv.loglike,
                                          conv._W,
                                          nonzero,
                                          dispersion=1.0)

        result = conv.selective_MLE(observed_target,
                                    cov_target,
                                    cov_target_score)[0]

        pval = result['pvalue']
        intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        covlist[j].extend([np.mean(coverage)])

    print(np.mean(covlist[j]))

fig, ax = plt.subplots()
ax.boxplot(covlist.values())
ax.set_xticklabels([100,250,500,750])
ax.axhline(y=0.9, color='k', linestyle='--', linewidth=2)
ax.set_ylim([0.6,1.0])
ax.set_title("Average Coverage for Single Lasso", y = 1.01,fontsize=20)
plt.savefig("tst_cvrg_single_lasso.png")