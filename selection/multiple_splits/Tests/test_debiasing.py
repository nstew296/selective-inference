import numpy as np
from selection.randomized.lasso import lasso, full_targets, selected_targets, debiased_targets
from selection.tests.instance import gaussian_instance, fixed_gaussian_instance

def test_full_targets(n=200,
                      p=1000,
                      signal_fac=0.50,
                      s=5,
                      sigma=3.,
                      rho=0.20,
                      randomizer_scale=1.):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    while True:
        signal = np.sqrt(signal_fac * 2. * np.log(p))
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        idx = np.arange(p)
        sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

        n, p = X.shape

        sigma_ = np.std(Y)
        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = const(X,
                     Y,
                     W,
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:
            dispersion = None

            (observed_target,
             cov_target,
             cov_target_score,
             alternatives,
             M_1) = debiased_targets(conv.loglike,
                                              conv._W,
                                              nonzero,
                                              penalty=conv.penalty,
                                              dispersion=dispersion)

            estimate, _, _, pval, intervals, _, _, _ = conv.selective_MLE(observed_target,
                                                                          cov_target,
                                                                          cov_target_score,
                                                                          alternatives)

            bias_target = M_1.dot(X.dot(beta))-beta[nonzero]
            print('check ', np.mean(bias_target))
            coverage = (beta[nonzero] > intervals[:, 0]) * (beta[nonzero] < intervals[:, 1])

            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals, (estimate-beta[nonzero]), bias_target

from selection.multiple_splits.utils import sim_xy

def test_mse_theory(seedn,
                    n=100,
                    p=500,
                    nval=100,
                    alpha= 1.,
                    rho=0.35,
                    s=5,
                    beta_type=1,
                    snr=0.55,
                    randomizer_scale=1.):

    X, y, _, _, Sigma, beta, sigma, _ = sim_xy(n=n, p=p, nval=nval, seedn=seedn, alpha=alpha, rho=rho, s=s,
                                               beta_type=beta_type,
                                               snr=snr)

    true_mean = X.dot(beta)
    X -= X.mean(0)[None, :]
    y = y - y.mean()
    scaling = X.std(0)[None, :] * np.sqrt(n / (n - 1.))

    X /= scaling
    dispersion = None
    sigma_ = np.std(y)
    print("sigma ", sigma, sigma_)
    lam = np.ones(p-1) * sigma_ * 1. * np.mean(np.fabs(np.dot(X[:, 1:].T, np.random.standard_normal((n, 2000)))).max(0))

    lasso_sol = lasso.gaussian(X,
                               y,
                               #feature_weights=lam,
                               feature_weights=np.append(0.001, lam),
                               randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)
    signs = lasso_sol.fit()
    nonzero = signs != 0

    (observed_target,
     cov_target,
     cov_target_score,
     alternatives,
     M_1) = debiased_targets(lasso_sol.loglike,
                             lasso_sol._W,
                             nonzero,
                             penalty=lasso_sol.penalty,
                             dispersion=dispersion)

    estimate, _, _, pval, intervals, _, _, _ = lasso_sol.selective_MLE(observed_target,
                                                                       cov_target,
                                                                       cov_target_score,
                                                                       alternatives)

    bias_target = M_1.dot(X.dot(beta)) - beta[nonzero]
    #print('check ', M_1.dot(X), np.mean(bias_target), nonzero[0],  (beta[nonzero])[0], bias_target[0])
    #print("some more check ", M_1.dot(true_mean)[0], (beta[nonzero])[0])
    coverage = (beta[nonzero] > intervals[:, 0]) * (beta[nonzero] < intervals[:, 1])
    #print("check intervals " , intervals, beta[nonzero])

    return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, \
           intervals, (estimate - (beta[nonzero])), bias_target

def test_mse_repoinst_debiased(n=200,
                               p=1000,
                               signal_fac=0.40,
                               s=5,
                               sigma=1.,
                               rho=0.20,
                               randomizer_scale=1.):
    """
    Compare to R randomized lasso
    """

    inst, const = fixed_gaussian_instance, lasso.gaussian
    while True:
        signal = np.sqrt(signal_fac * 2. * np.log(p))
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          alpha = 1.,
                          random_signs=True)[:3]

        idx = np.arange(p)
        sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        print("snr", beta[0], beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

        n, p = X.shape

        sigma_ = np.std(Y)
        W = np.append(0.01, np.ones(X.shape[1]-1) * np.sqrt(2 * np.log(p)) * sigma_)

        conv = const(X,
                     Y,
                     W,
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:
            dispersion = None

            (observed_target,
             cov_target,
             cov_target_score,
             alternatives,
             M_1) = debiased_targets(conv.loglike,
                                              conv._W,
                                              nonzero,
                                              penalty=conv.penalty,
                                              dispersion=dispersion)

            estimate, _, _, pval, intervals, _, _, _ = conv.selective_MLE(observed_target,
                                                                          cov_target,
                                                                          cov_target_score,
                                                                          alternatives)

            bias_target = M_1.dot(X.dot(beta))-beta[nonzero]
            print('check ', nonzero[0], (beta[nonzero])[0])
            coverage = (beta[nonzero] > intervals[:, 0]) * (beta[nonzero] < intervals[:, 1])

            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, \
                   intervals, (estimate-beta[nonzero]), bias_target

def test_mse_repoinst_selected(n=200,
                               p=1000,
                               signal_fac=0.40,
                               s=5,
                               sigma=1.,
                               rho=0.20,
                               randomizer_scale=1.):
    """
    Compare to R randomized lasso
    """

    inst, const = fixed_gaussian_instance, lasso.gaussian
    while True:
        signal = np.sqrt(signal_fac * 2. * np.log(p))
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          alpha = 1.,
                          random_signs=True)[:3]

        idx = np.arange(p)
        sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        print("snr", beta[0], beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

        n, p = X.shape

        sigma_ = np.std(Y)
        W = np.append(0.01, np.ones(X.shape[1]-1) * np.sqrt(2 * np.log(p)) * sigma_)

        conv = const(X,
                     Y,
                     W,
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum(), nonzero[0])

        if nonzero[0]==1:
            dispersion = None

            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(conv.loglike,
                                              conv._W,
                                              nonzero,
                                              dispersion=dispersion)

            estimate, _, _, pval, intervals, _, _, _ = conv.selective_MLE(observed_target,
                                                                          cov_target,
                                                                          cov_target_score,
                                                                          alternatives)

            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
            bias_target = beta_target -beta[nonzero]
            print('check ', nonzero[0], (beta[nonzero])[0])
            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage[0], \
                   intervals, (estimate-beta_target)[0], bias_target[0]

def main(nsim=500, instance= "repo"):

    P0, PA, cover, length_int, nn, _nn_ = [], [], [], [], [], []
    n, p, s = 100, 500, 10

    for i in range(nsim):
        if instance== "repo":
            p0, pA, cover_, intervals, nn_, _nn = test_full_targets(n=n, p=p, s=s)
            nn.extend(nn_)
            _nn_.extend(_nn)
            cover.extend(cover_)

        elif instance== "sim_xy":
            p0, pA, cover_, intervals, nn_, _nn= test_mse_theory(seedn= (i+1),
                                                                 n=100,
                                                                 p=500,
                                                                 nval=100,
                                                                 alpha=1.,
                                                                 rho=0.35,
                                                                 s=5,
                                                                 beta_type=1,
                                                                 snr=0.55,
                                                                 randomizer_scale=1.)
            nn.extend(nn_)
            _nn_.extend(_nn)
            cover.extend(cover_)

        else:
            p0, pA, cover_, intervals, nn_, _nn = test_mse_repoinst_selected(n=n, p=p, s=s)
            nn.append(nn_)
            _nn_.append(_nn)
            cover.append(cover_)


        P0.extend(p0)
        PA.extend(pA)
        print("bias so far ", i+1, np.mean(cover), np.mean(nn), np.mean(_nn_))

main(nsim=1000, instance= "sim_xy")