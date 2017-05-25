import numpy as np


class logistic_intervals():

    def __init__(self,
                 X_bar_obs,
                 n,
                 threshold,
                 data_variance):

        self.X_bar_obs = X_bar_obs
        self.n = n
        self.threshold = threshold
        self.data_variance = data_variance

        grid_length = 400
        bar_X_grid = np.linspace(-10./np.sqrt(self.n),10./np.sqrt(self.n), num=grid_length)
        self.index_grid = np.argmin(np.abs(bar_X_grid-self.X_bar_obs))

        delta = 0.001
        self.randomization_scale = n^{-1./(6+delta)}

        weights = []
        for i in range(bar_X_grid.shape[0]):
            weights.append(self.log_logistic_weights(bar_X_grid[i]))

        self.weights = np.array(weights)

    def log_logistic_weights(self, X_bar):

        x = np.true_divide(self.threshold - np.sqrt(self.n)* X_bar, self.randomization_scale)
        G_bar = 1.-(1./(1.+ np.exp(-x)))

        return G_bar

    def area_normalized_density(self, mu):

        normalizer = 0.
        nonnormalized = []

        for i in range(self.bar_X_grid.shape[0]):
            density = np.exp(-self.n* (np.true_divide((self.grid[i] - mu) ** 2), 2* self.data_variance)+ (self.weights[i]))
            normalizer += density
            nonnormalized.append(density)

        return np.cumsum(np.array(nonnormalized / normalizer))


    def confidence_intervals(self):

        param_grid = np.linspace(-10./np.sqrt(self.n),10./np.sqrt(self.n), num= 400)
        area = np.zeros(param_grid.shape[0])

        for k in range(param_grid.shape[0]):
            area_vec = self.area_normalized_density(param_grid[k])
            area[k] = area_vec[self.index_grid]

        region = param_grid[(area >= 0.05) & (area <= 0.95)]
        if region.size > 0:
            return np.nanmin(region), np.nanmax(region)
        else:
            return 0, 0














