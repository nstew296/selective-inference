import numpy as np
from selectinf.randomized.multitask_lasso import multi_task_lasso
from selectinf.tests.instance import gaussian_multitask_instance
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

ntask = 5
nsamples = 1000 * np.ones(ntask)
nsamples_test = 1000 * np.ones(ntask)
p_list = [100,250,500,750]
global_sparsity = 0.9
task_sparsity = 0.4
sigma = 1. * np.ones(ntask)
signal_fac = np.array([1.0,3.0])
rhos = 0.3 * np.ones(ntask)
nsamples = nsamples.astype(int)
nsamples_test = nsamples_test.astype(int)
weight = 2.25
randomizer_scale = 0.7
#covlist = {j: [] for j in range(len(p_list))}
selected_list = {j: [] for j in range(len(p_list))}

for j in range(len(p_list)):
    p = p_list[j]
    signal = np.sqrt(signal_fac * 2 * np.log(p))
    #weight = np.sqrt(2 * np.log(p))
    s = int(p*0.1*0.6)
    for i in range(100):
        response_vars_train, predictor_vars_train, response_vars_test, predictor_vars_test, beta, gaussian_noise = gaussian_multitask_instance(
            ntask,
            nsamples,
            nsamples_test,
            p,
            global_sparsity,
            task_sparsity,
            sigma,
            signal,
            rhos,
            random_signs=True,
            equicorrelated=True)[:6]

        feature_weight = weight * np.ones(p)
        randomizer_scales = randomizer_scale * np.array([sigma[i] for i in range(ntask)])
        initial_omega = np.array(
            [randomizer_scales[i] * gaussian_noise[(i * p):((i + 1) * p)] for i in range(ntask)]).T

        multi_lasso = multi_task_lasso.gaussian(predictor_vars_train,
                                                response_vars_train,
                                                feature_weight,
                                                ridge_term=None,
                                                randomizer_scales=randomizer_scales)

        active_signs = multi_lasso.fit(perturbations=initial_omega)

        selected_list[j].extend([np.sum(active_signs!=0)/5.])

    print(np.mean(selected_list[j]))

fig, ax = plt.subplots()
ax.boxplot(selected_list.values())
ax.set_xticklabels([100,250,500,750])
#ax.axhline(y=0.9, color='k', linestyle='--', linewidth=2)
#ax.set_ylim([0.6,1.0])
#ax.set_title("Average Coverage for Single Lasso", y = 1.01,fontsize=20)
ax.set_title("Mean Selected Model Size for MT Lasso", y = 1.01,fontsize=20)
plt.savefig("tst_size_mt_lasso.png")