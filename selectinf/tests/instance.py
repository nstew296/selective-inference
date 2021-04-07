import numpy as np
import pandas as pd

from scipy.stats import t as tdist

_cov_cache = {}

def _design(n, p, rho, equicorrelated):
    """
    Create an equicorrelated or AR(1) design.
    """
    if equicorrelated:
        X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) +
             np.sqrt(rho) * np.random.standard_normal(n)[:, None])
        def equi(rho, p):
            if ('equi', p, rho) not in _cov_cache:
                sigmaX = (1 - rho) * np.identity(p) + rho * np.ones((p, p))
                cholX = np.linalg.cholesky(sigmaX)
                _cov_cache[('equi', p, rho)] = sigmaX, cholX
            return _cov_cache[('equi', p, rho)]
        sigmaX, cholX = equi(rho=rho, p=p)
    else:
        def AR1(rho, p):
            if ('AR1', p, rho) not in _cov_cache:
                idx = np.arange(p)
                cov = rho ** np.abs(np.subtract.outer(idx, idx))
                _cov_cache[('AR1', p, rho)] = cov, np.linalg.cholesky(cov)
            cov, chol = _cov_cache[('AR1', p, rho)]
            return cov, chol
        sigmaX, cholX = AR1(rho=rho, p=p)
        X = np.random.standard_normal((n, p)).dot(cholX.T)
    return X, sigmaX, cholX

def gaussian_instance(n=100, p=200, s=7, sigma=5, rho=0., signal=7,
                      random_signs=False, df=np.inf,
                      scale=True, center=True,
                      equicorrelated=True):


    """
    A testing instance for the LASSO.
    If equicorrelated is True design is equi-correlated in the population,
    normalized to have columns of norm 1.
    If equicorrelated is False design is auto-regressive.
    For the default settings, a $\lambda$ of around 13.5
    corresponds to the theoretical $E(\|X^T\epsilon\|_{\infty})$
    with $\epsilon \sim N(0, \sigma^2 I)$.

    Parameters
    ----------

    n : int
        Sample size

    p : int
        Number of features

    s : int
        True sparsity

    sigma : float
        Noise level

    rho : float
        Equicorrelation value (must be in interval [0,1])

    signal : float or (float, float)
        Sizes for the coefficients. If a tuple -- then coefficients
        are equally spaced between these values using np.linspace.
        Note: the size of signal is for a "normalized" design, where np.diag(X.T.dot(X)) == np.ones(p).
        If scale=False, this signal is divided by np.sqrt(n), otherwise it is unchanged.

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive.

    df : int
        Degrees of freedom for noise (from T distribution).

    equicorrelated: bool
        If true, design in equi-correlated,
        Else design is AR.

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

    sigma : float
        Noise level.

    sigmaX : np.ndarray((p,p))
        Row covariance.
    """

    X, sigmaX = _design(n, p, rho, equicorrelated)[:2]

    if center:
        X -= X.mean(0)[None, :]

    beta = np.zeros(p) 
    signal = np.atleast_1d(signal)
    if signal.shape == (1,):
        beta[:s] = signal[0] 
    else:
        beta[:s] = np.linspace(signal[0], signal[1], s)
    if random_signs:
        beta[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
    np.random.shuffle(beta)
    beta /= np.sqrt(n)

    if scale:
        scaling = X.std(0) * np.sqrt(n)
        X /= scaling[None, :]
        beta *= np.sqrt(n)
        sigmaX = sigmaX / np.multiply.outer(scaling, scaling)

    active = np.zeros(p, np.bool)
    active[beta != 0] = True

    # noise model
    def _noise(n, df=np.inf):
        if df == np.inf:
            return np.random.standard_normal(n)
        else:
            sd_t = np.std(tdist.rvs(df, size=50000))
        return tdist.rvs(df, size=n) / sd_t

    Y = (X.dot(beta) + _noise(n, df)) * sigma
    return X, Y, beta * sigma, np.nonzero(active)[0], sigma, sigmaX


def logistic_instance(n=100, p=200, s=7, rho=0.3, signal=14,
                      random_signs=False, 
                      scale=True, 
                      center=True, 
                      equicorrelated=True):
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

    signal : float or (float, float)
        Sizes for the coefficients. If a tuple -- then coefficients
        are equally spaced between these values using np.linspace.
        Note: the size of signal is for a "normalized" design, where np.diag(X.T.dot(X)) == np.ones(p).
        If scale=False, this signal is divided by np.sqrt(n), otherwise it is unchanged.

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive.

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

    sigmaX : np.ndarray((p,p))
        Row covariance.

    """

    X, sigmaX = _design(n, p, rho, equicorrelated)[:2]

    if center:
        X -= X.mean(0)[None,:]

    beta = np.zeros(p) 
    signal = np.atleast_1d(signal)
    if signal.shape == (1,):
        beta[:s] = signal[0] 
    else:
        beta[:s] = np.linspace(signal[0], signal[1], s)
    if random_signs:
        beta[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
    np.random.shuffle(beta)
    beta /= np.sqrt(n)

    if scale:
        scaling = X.std(0) * np.sqrt(n)
        X /= scaling[None, :]
        beta *= np.sqrt(n)
        sigmaX = sigmaX / np.multiply.outer(scaling, scaling)

    active = np.zeros(p, np.bool)
    active[beta != 0] = True

    eta = linpred = np.dot(X, beta) 
    pi = np.exp(eta) / (1 + np.exp(eta))

    Y = np.random.binomial(1, pi)
    return X, Y, beta, np.nonzero(active)[0], sigmaX

def poisson_instance(n=100, p=200, s=7, rho=0.3, signal=4,
                     random_signs=False, 
                     scale=True, 
                     center=True, 
                     equicorrelated=True):
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

    signal : float or (float, float)
        Sizes for the coefficients. If a tuple -- then coefficients
        are equally spaced between these values using np.linspace.
        Note: the size of signal is for a "normalized" design, where np.diag(X.T.dot(X)) == np.ones(p).
        If scale=False, this signal is divided by np.sqrt(n), otherwise it is unchanged.

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive.

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

    sigmaX : np.ndarray((p,p))
        Row covariance.

    """

    X, sigmaX = _design(n, p, rho, equicorrelated)[:2]

    if center:
        X -= X.mean(0)[None,:]

    beta = np.zeros(p) 
    signal = np.atleast_1d(signal)
    if signal.shape == (1,):
        beta[:s] = signal[0] 
    else:
        beta[:s] = np.linspace(signal[0], signal[1], s)
    if random_signs:
        beta[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
    np.random.shuffle(beta)
    beta /= np.sqrt(n)

    if scale:
        scaling = X.std(0) * np.sqrt(n)
        X /= scaling[None, :]
        beta *= np.sqrt(n)
        sigmaX = sigmaX / np.multiply.outer(scaling, scaling)

    active = np.zeros(p, np.bool)
    active[beta != 0] = True

    eta = linpred = np.dot(X, beta) 
    mu = np.exp(eta)

    Y = np.random.poisson(mu)
    return X, Y, beta, np.nonzero(active)[0], sigmaX

def HIV_NRTI(drug='3TC', 
             standardize=True, 
             datafile=None,
             min_occurrences=11):
    """
    Download 

        http://hivdb.stanford.edu/pages/published_analysis/genophenoPNAS2006/DATA/NRTI_DATA.txt

    and return the data set for a given NRTI drug.

    The response is an in vitro measurement of log-fold change 
    for a given virus to that specific drug.

    Parameters
    ----------

    drug : str (optional)
        One of ['3TC', 'ABC', 'AZT', 'D4T', 'DDI', 'TDF']

    standardize : bool (optional)
        If True, center and scale design X and center response Y.

    datafile : str (optional)
        A copy of NRTI_DATA above.

    min_occurrences : int (optional)
        Only keep positions that appear
        at least a minimum number of times.
        

    """

    if datafile is None:
        datafile = "http://hivdb.stanford.edu/pages/published_analysis/genophenoPNAS2006/DATA/NRTI_DATA.txt"
    NRTI = pd.read_table(datafile, na_values="NA")

    NRTI_specific = []
    NRTI_muts = []
    mixtures = np.zeros(NRTI.shape[0])
    for i in range(1,241):
        d = NRTI['P%d' % i]
        for mut in np.unique(d):
            if mut not in ['-','.'] and len(mut) == 1:
                test = np.equal(d, mut)
                if test.sum() >= min_occurrences:
                    NRTI_specific.append(np.array(np.equal(d, mut))) 
                    NRTI_muts.append("P%d%s" % (i,mut))

    NRTI_specific = NRTI.from_records(np.array(NRTI_specific).T, columns=NRTI_muts)

    X_NRTI = np.array(NRTI_specific, np.float)
    Y = np.asarray(NRTI[drug]) # shorthand
    keep = ~np.isnan(Y).astype(np.bool)
    X_NRTI = X_NRTI[np.nonzero(keep)]; Y=Y[keep]
    Y = np.array(np.log(Y), np.float); 

    if standardize:
        Y -= Y.mean()
        X_NRTI -= X_NRTI.mean(0)[None, :]; X_NRTI /= X_NRTI.std(0)[None,:]
    return X_NRTI, Y, np.array(NRTI_muts)


def gaussian_multitask_instance(ntask,
                                nsamples,
                                p,
                                global_sparsity,
                                task_sparsity,
                                sigma,
                                signal,
                                rhos, #list of correlation parameters
                                random_signs=False,
                                df=np.inf,
                                scale=True,
                                center=True,
                                equicorrelated=False):


    predictor_vars= {i: _design(nsamples[i], p, rhos[i], equicorrelated)[0] for i in range(ntask)}

    if center:
        predictor_vars = {i: predictor_vars[i]-predictor_vars[i].mean(0)[None, :] for i in range(ntask)}

    signal = np.atleast_1d(signal)

    if signal.shape == (1,):
        beta = float(signal[0]) * np.ones((p,ntask))
        global_nulls = np.random.choice(p, int(round(global_sparsity * p)), replace=False)
        beta[global_nulls, :] = np.zeros((ntask,))
        for i in np.delete(range(p), global_nulls):
            beta[i, np.random.choice(ntask, int(round(task_sparsity * ntask)), replace=False)] = 0.

    else:
        beta = np.ones((p,ntask))
        nsignal = int(round(global_sparsity * p))
        global_nulls = np.random.choice(p, nsignal, replace=False)
        beta[global_nulls, :] = np.zeros((ntask,))

        print(np.delete(range(p), global_nulls))
        for i in np.delete(range(p), global_nulls):
            null_positions = np.random.choice(ntask, int(round(task_sparsity * ntask)), replace=False)
            beta[i, null_positions] = 0.
            non_null_positions = np.setdiff1d(np.arange(ntask), null_positions)
            beta[i, non_null_positions] = np.linspace(float(signal[0]), float(signal[1]), num=ntask-null_positions.shape[0])

    if random_signs:
        beta *= (2 * np.random.binomial(1, 0.5, size=(p,ntask)) - 1.)

    beta /= np.sqrt(nsamples)

    if scale:
        scalings = {i: predictor_vars[i].std(0) * np.sqrt(nsamples[i]) for i in range(ntask)}
        predictor_vars = {i: predictor_vars[i]/(scalings[i][None, :]) for i in range(ntask)}
        beta *= np.sqrt(nsamples)

    active = np.zeros((p, ntask), np.bool)
    active[beta != 0] = True

    # noise model
    def _noise(n, df=np.inf):
        if df == np.inf:
            return np.random.standard_normal(n)
        else:
            sd_t = np.std(tdist.rvs(df, size=50000))
        return tdist.rvs(df, size=n) / sd_t

    gaussian_noise = _noise(nsamples.sum() + p*ntask, df)
    response_vars = {}
    nsamples_cumsum = np.cumsum(nsamples)
    for i in range(ntask):
        if i == 0:
            response_vars[i] = (predictor_vars[i].dot(beta[:, i]) + gaussian_noise[:nsamples_cumsum[i]]) * sigma[i]
        else:
            response_vars[i] = (predictor_vars[i].dot(beta[:, i]) + gaussian_noise[nsamples_cumsum[i-1]:nsamples_cumsum[i]]) * sigma[i]

    return response_vars, predictor_vars, beta * sigma, gaussian_noise[nsamples_cumsum[-1]:], np.nonzero(active), sigma


def logistic_multitask_instance(ntask,
                                nsamples,
                                p,
                                global_sparsity,
                                task_sparsity,
                                sigma,
                                signal,
                                rhos, #list of correlation parameters
                                random_signs=False,
                                df=np.inf,
                                scale=True,
                                center=True,
                                equicorrelated=False):


    predictor_vars= {i: _design(nsamples[i], p, rhos[i], equicorrelated)[0] for i in range(ntask)}

    if center:
        predictor_vars = {i: predictor_vars[i]-predictor_vars[i].mean(0)[None, :] for i in range(ntask)}

    signal = np.atleast_1d(signal)

    if signal.shape == (1,):
        beta = float(signal[0]) * np.ones((p,ntask))
        global_nulls = np.random.choice(p, int(round(global_sparsity * p)), replace=False)
        beta[global_nulls, :] = np.zeros((ntask,))
        for i in np.delete(range(p), global_nulls):
            beta[i, np.random.choice(ntask, int(round(task_sparsity * ntask)), replace=False)] = 0.

    else:
        beta = np.ones((p,ntask))
        nsignal = int(round(global_sparsity * p))
        global_nulls = np.random.choice(p, nsignal, replace=False)
        beta[global_nulls, :] = np.zeros((ntask,))

        for i in np.delete(range(p), global_nulls):
            null_positions = np.random.choice(ntask, int(round(task_sparsity * ntask)), replace=False)
            beta[i, null_positions] = 0.
            non_null_positions = np.setdiff1d(np.arange(ntask), null_positions)
            beta[i, non_null_positions] = np.linspace(float(signal[0]), float(signal[1]), num=ntask-null_positions.shape[0])

    if random_signs:
        beta *= (2 * np.random.binomial(1, 0.5, size=(p,ntask)) - 1.)

    beta /= np.sqrt(nsamples)

    if scale:
        scalings = {i: predictor_vars[i].std(0) * np.sqrt(nsamples[i]) for i in range(ntask)}
        predictor_vars = {i: predictor_vars[i]/(scalings[i][None, :]) for i in range(ntask)}
        beta *= np.sqrt(nsamples)

    active = np.zeros((p, ntask), np.bool)
    active[beta != 0] = True

    # noise model
    def _noise(n, df=np.inf):
        if df == np.inf:
            return np.random.standard_normal(n)
        else:
            sd_t = np.std(tdist.rvs(df, size=50000))
        return tdist.rvs(df, size=n) / sd_t

    gaussian_noise = _noise(p*ntask, df)
    response_vars = {}
    pis = {}
    for i in range(ntask):
        pis[i] = predictor_vars[i].dot(beta[:, i])*sigma[i]
        response_vars[i] = np.random.binomial(1,np.exp(pis[i])/(1.0+np.exp(pis[i]))).astype(float)
        print(response_vars[i])

    return response_vars, predictor_vars, beta * sigma, gaussian_noise, np.nonzero(active), sigma


def poisson_multitask_instance(ntask,
                                nsamples,
                                p,
                                global_sparsity,
                                task_sparsity,
                                sigma,
                                signal,
                                rhos, #list of correlation parameters
                                random_signs=False,
                                df=np.inf,
                                scale=True,
                                center=True,
                                equicorrelated=False):


    predictor_vars= {i: _design(nsamples[i], p, rhos[i], equicorrelated)[0] for i in range(ntask)}

    if center:
        predictor_vars = {i: predictor_vars[i]-predictor_vars[i].mean(0)[None, :] for i in range(ntask)}

    signal = np.atleast_1d(signal)

    if signal.shape == (1,):
        beta = float(signal[0]) * np.ones((p,ntask))
        global_nulls = np.random.choice(p, int(round(global_sparsity * p)), replace=False)
        beta[global_nulls, :] = np.zeros((ntask,))
        for i in np.delete(range(p), global_nulls):
            beta[i, np.random.choice(ntask, int(round(task_sparsity * ntask)), replace=False)] = 0.

    else:
        beta = np.ones((p,ntask))
        nsignal = int(round(global_sparsity * p))
        global_nulls = np.random.choice(p, nsignal, replace=False)
        beta[global_nulls, :] = np.zeros((ntask,))

        for i in np.delete(range(p), global_nulls):
            null_positions = np.random.choice(ntask, int(round(task_sparsity * ntask)), replace=False)
            beta[i, null_positions] = 0.
            non_null_positions = np.setdiff1d(np.arange(ntask), null_positions)
            beta[i, non_null_positions] = np.linspace(float(signal[0]), float(signal[1]), num=ntask-null_positions.shape[0])

    if random_signs:
        beta *= (2 * np.random.binomial(1, 0.5, size=(p,ntask)) - 1.)

    beta /= np.sqrt(nsamples)

    if scale:
        scalings = {i: predictor_vars[i].std(0) * np.sqrt(nsamples[i]) for i in range(ntask)}
        predictor_vars = {i: predictor_vars[i]/(scalings[i][None, :]) for i in range(ntask)}
        beta *= np.sqrt(nsamples)

    active = np.zeros((p, ntask), np.bool)
    active[beta != 0] = True

    # noise model
    def _noise(n, df=np.inf):
        if df == np.inf:
            return np.random.standard_normal(n)
        else:
            sd_t = np.std(tdist.rvs(df, size=50000))
        return tdist.rvs(df, size=n) / sd_t

    gaussian_noise = _noise(p*ntask, df)
    response_vars = {}
    pis = {}
    for i in range(ntask):
        pis[i] = predictor_vars[i].dot(beta[:, i])*sigma[i]
        response_vars[i] = np.random.poisson(np.exp(pis[i])).astype(float)
        print(response_vars[i])

    return response_vars, predictor_vars, beta * sigma, gaussian_noise, np.nonzero(active), sigma