import matplotlib.pyplot as plt
import numpy as np
from selection.scaled_randomization.logistic_intervals import logistic_intervals
from selection.scaled_randomization.gaussian_alternatives import gaussian_intervals

def test_logistic(n, grid_n, scale):

    data_grid_obs = np.linspace(-6./np.sqrt(n),4./np.sqrt(n), num = grid_n)
    intervals = np.zeros((grid_n, 2))
    naive_intervals = np.zeros((grid_n, 2))

    for i in range(data_grid_obs.shape[0]):

        compute = logistic_intervals(data_grid_obs[i],
                                       n,
                                       0.,
                                       scale= scale)

        intervals[i,:] = compute.confidence_intervals()
        naive_intervals[i,0] = data_grid_obs[i]-1.65*(1./np.sqrt(n))
        naive_intervals[i, 1] = data_grid_obs[i] + 1.65 * (1. / np.sqrt(n))
        print("iteration completed", i, intervals[i, :])

    return intervals, naive_intervals

def plot_intervals(n, grid_n):

    compute_1 = test_logistic(n, grid_n, True)
    selective_intervals_1 = compute_1[0]
    nominal_intervals = compute_1[1]

    compute = test_logistic(n, grid_n, False)
    selective_intervals = compute[0]

    data_grid_obs = np.linspace(-6. / np.sqrt(n), 4. / np.sqrt(n), num= grid_n)

    fig = plt.figure()
    fig.suptitle('Univariate intervals')

    #ax = fig.gca()
    #ax.set_xbound(lower= -6. / np.sqrt(n), upper= 4. / np.sqrt(n))
    plt.xlim(-6. / np.sqrt(n), 4. / np.sqrt(n))
    plt.ylim(-13. / np.sqrt(n), 6. / np.sqrt(n))
    plt.plot(data_grid_obs, selective_intervals_1[:,0], '-', c='b', lw=2, label='scaled')
    plt.plot(data_grid_obs, selective_intervals_1[:,1], '-', c='b', lw=2)
    plt.plot(data_grid_obs, selective_intervals[:, 0], '-.', c='g', lw=1, label= 'usual')
    plt.plot(data_grid_obs, selective_intervals[:, 1], '-.', c='g', lw=1)
    plt.plot(data_grid_obs, nominal_intervals[:,0], 'r--', c ='k', lw=1, label = 'nominal')
    plt.plot(data_grid_obs, nominal_intervals[:,1], 'r--', c = 'k', lw=1)
    legend = plt.legend(loc='upper left', shadow=False)
    #frame = legend.get_frame()

    plt.savefig('/Users/snigdhapanigrahi/Documents/Research/Python_plots/logistic_intervals.pdf', bbox_inches='tight')

#plot_intervals(10000, 200)
#test(100, 200)

####Gaussian test

def test_gaussian(n, grid_n, scale):

    data_grid_obs = np.linspace(-2./np.sqrt(n), 4./np.sqrt(n), num = grid_n)
    intervals = np.zeros((grid_n, 2))
    naive_intervals = np.zeros((grid_n, 2))

    for i in range(data_grid_obs.shape[0]):

        compute = gaussian_intervals(data_grid_obs[i],
                                     n,
                                     0.,
                                     scale= scale)

        intervals[i,:] = compute.confidence_intervals()
        naive_intervals[i,0] = data_grid_obs[i]-1.65*(1./np.sqrt(n))
        naive_intervals[i, 1] = data_grid_obs[i] + 1.65 * (1. / np.sqrt(n))
        print("iteration completed", i, intervals[i, :])

    return intervals, naive_intervals


def plot_intervals(n, grid_n):

    compute_1 = test_gaussian(n, grid_n, True)
    selective_intervals_1 = compute_1[0]
    nominal_intervals = compute_1[1]

    compute = test_gaussian(n, grid_n, False)
    selective_intervals = compute[0]

    data_grid_obs = np.linspace(-2. / np.sqrt(n), 4. / np.sqrt(n), num= grid_n)

    fig = plt.figure()
    fig.suptitle('Univariate intervals')


    plt.xlim(-2. / np.sqrt(n), 4. / np.sqrt(n))
    plt.ylim(-15. / np.sqrt(n), 6. / np.sqrt(n))
    plt.plot(data_grid_obs, selective_intervals_1[:,0], '-', c='b', lw=2, label='scaled')
    plt.plot(data_grid_obs, selective_intervals_1[:,1], '-', c='b', lw=2)
    plt.plot(data_grid_obs, selective_intervals[:, 0], '-.', c='g', lw=1, label= 'usual')
    plt.plot(data_grid_obs, selective_intervals[:, 1], '-.', c='g', lw=1)
    plt.plot(data_grid_obs, nominal_intervals[:,0], 'r--', c ='k', lw=1, label = 'nominal')
    plt.plot(data_grid_obs, nominal_intervals[:,1], 'r--', c = 'k', lw=1)
    legend = plt.legend(loc='upper left', shadow=False)

    plt.savefig('/Users/snigdhapanigrahi/Documents/Research/Python_plots/gaussian_intervals.pdf', bbox_inches='tight')


plot_intervals(10000, 200)




