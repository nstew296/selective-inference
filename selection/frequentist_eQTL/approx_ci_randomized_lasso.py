import numpy as np
import sys
import regreg.api as rr
from selection.bayesian.selection_probability_rr import nonnegative_softmax_scaled
from scipy.stats import norm

#from selection.frequentist_eQTL.approx_confidence_intervals import neg_log_cube_probability

class neg_log_cube_probability(rr.smooth_atom):
    def __init__(self,
                 q, #equals p - E in our case
                 lagrange,
                 randomization_scale = 1., #equals the randomization variance in our case
                 coef = 1.,
                 offset=None,
                 quadratic=None):

        self.randomization_scale = randomization_scale
        self.lagrange = lagrange
        self.q = q

        rr.smooth_atom.__init__(self,
                                (self.q,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=None,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        arg = self.apply_offset(arg)

        arg_u = (arg + self.lagrange)/self.randomization_scale
        arg_l = (arg - self.lagrange)/self.randomization_scale
        prod_arg = np.exp(-(2. * self.lagrange * arg)/(self.randomization_scale**2))
        neg_prod_arg = np.exp((2. * self.lagrange * arg)/(self.randomization_scale**2))
        cube_prob = norm.cdf(arg_u) - norm.cdf(arg_l)

        threshold = 10 ** -10
        indicator = np.zeros(self.q, bool)
        indicator[(cube_prob > threshold)] = 1
        positive_arg = np.zeros(self.q, bool)
        positive_arg[(arg>0)] = 1
        pos_index = np.logical_and(positive_arg, ~indicator)
        neg_index = np.logical_and(~positive_arg, ~indicator)

        # log_cube_prob = np.zeros(self.q)
        # log_cube_prob[indicator] = -np.log(cube_prob[indicator])
        #
        # log_cube_prob[pos_index] = ((((arg[pos_index]**2.)/2.) + ((self.lagrange[pos_index]**2./2.))
        #                                 -(arg[pos_index]*self.lagrange[pos_index]))/(self.randomization_scale**2))-\
        #                               np.log(-prod_arg[pos_index]/np.abs(arg_u[pos_index]) + 1./np.abs(arg_l[pos_index]))
        #
        # log_cube_prob[neg_index] = ((((arg[neg_index] ** 2.) / 2.) + ((self.lagrange[neg_index] ** 2. / 2.))
        #                                 + (arg[neg_index] * self.lagrange[neg_index])) / (self.randomization_scale ** 2)) - \
        #                                   np.log(1./np.abs(arg_u[neg_index]) - neg_prod_arg[neg_index]/np.abs(arg_l[neg_index]))
        #
        # neg_log_cube_prob = log_cube_prob.sum()
        neg_log_cube_prob = -np.log(cube_prob).sum()

        log_cube_grad = np.zeros(self.q)
        log_cube_grad[indicator] = (np.true_divide(-norm.pdf(arg_u[indicator]) + norm.pdf(arg_l[indicator]),
                                        cube_prob[indicator]))/self.randomization_scale

        log_cube_grad[pos_index] = ((-1. + prod_arg[pos_index])/
                                     ((prod_arg[pos_index]/arg_u[pos_index])-
                                      (1./arg_l[pos_index])))/self.randomization_scale

        log_cube_grad[neg_index] = ((arg_u[neg_index] -(arg_l[neg_index]*neg_prod_arg[neg_index]))
                                    /self.randomization_scale)/(1.- neg_prod_arg[neg_index])


        if mode == 'func':
            return self.scale(neg_log_cube_prob)
        elif mode == 'grad':
            return self.scale(log_cube_grad)
        elif mode == 'both':
            return self.scale(neg_log_cube_prob), self.scale(log_cube_grad)
        else:
            raise ValueError("mode incorrectly specified")


class approximate_conditional_prob(rr.smooth_atom):

    def __init__(self,
                 t, #point at which density is to computed
                 map,
                 coef = 1.,
                 offset= None,
                 quadratic= None):

        self.t = t
        self.map = map
        self.q = map.p - map.nactive
        self.inactive_conjugate = self.active_conjugate = map.randomization.CGF_conjugate

        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        self.inactive_lagrange = self.map.inactive_lagrange

        rr.smooth_atom.__init__(self,
                                (map.nactive,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=self.map.feasible_point,
                                coef=coef)

        self.coefs[:] = map.feasible_point

        self.nonnegative_barrier = nonnegative_softmax_scaled(self.map.nactive)


    def sel_prob_smooth_objective(self, param, mode='both', check_feasibility=False):

        param = self.apply_offset(param)

        data = np.squeeze(self.t *  self.map.A)

        offset_active = self.map.offset_active + data[:self.map.nactive]
        offset_inactive = self.map.offset_inactive + data[self.map.nactive:]

        active_conj_loss = rr.affine_smooth(self.active_conjugate,
                                            rr.affine_transform(self.map.B_active, offset_active))


        cube_obj = neg_log_cube_probability(self.q, self.inactive_lagrange, randomization_scale = 1.)

        cube_loss = rr.affine_smooth(cube_obj, rr.affine_transform(self.map.B_inactive, offset_inactive))

        total_loss = rr.smooth_sum([active_conj_loss,
                                    cube_loss,
                                    self.nonnegative_barrier])

        if mode == 'func':
            f = total_loss.smooth_objective(param, 'func')
            return self.scale(f)
        elif mode == 'grad':
            g = total_loss.smooth_objective(param, 'grad')
            #print(active_conj_loss.smooth_objective(param, 'grad'), cube_loss.smooth_objective(param, 'grad'),
            #      self.nonnegative_barrier.smooth_objective(param, 'grad'))
            return self.scale(g)
        elif mode == 'both':
            f, g = total_loss.smooth_objective(param, 'both')
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def minimize2(self, step=1., nstep=30, tol=1.e-6):

        current = self.coefs
        current_value = np.inf

        objective = lambda u: self.sel_prob_smooth_objective(u, 'func')
        grad = lambda u: self.sel_prob_smooth_objective(u, 'grad')
        inv_hessian = np.identity(self.coefs.shape[0])

        for itercount in xrange(nstep):
            newton_step = inv_hessian.dot(grad(current))

            # make sure proposal is feasible

            count = 0
            while True:
                count += 1
                proposal = current - step * newton_step
                #print("proposal, current", proposal-current, step * newton_step)
                #print("current proposal and grad", proposal, newton_step)
                if np.all(proposal > 0):
                    break
                step *= 0.5
                if count >= 40:
                    #print(proposal)
                    raise ValueError('not finding a feasible point')


            # make sure proposal is a descent

            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)
                #print(current_value, proposed_value, 'minimize')
                if proposed_value <= current_value:
                    break
                step *= 0.5

                #print("proposal, current", proposal, current)
            # stop if relative decrease is small

            if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value):
                current = proposal
                current_value = proposed_value
                break

            diff_gradient = grad(proposal) - grad(current)
            diff_step = proposal-current
            diff_mat_step = diff_step.dot(diff_gradient.T)
            diff_mat_grad = diff_gradient.dot(diff_step.T)
            diff_scalar = diff_gradient.T.dot(diff_step)
            #print("proposal, current", proposal, current)
            #print("diff_scalar", diff_scalar, diff_gradient, diff_step)
            inv_hessian = (np.identity(self.coefs.shape[0])- (1./diff_scalar)*diff_mat_step).dot(inv_hessian)\
                .dot(np.identity(self.coefs.shape[0])- (1./diff_scalar)*diff_mat_grad) \
                          + (1./diff_scalar)*(diff_step.dot(diff_step.T))

            current = proposal
            current_value = proposed_value

            if itercount % 4 == 0:
                step *= 2

        # print('iter', itercount)
        value = objective(current)

        return current, value

class approximate_conditional_density(rr.smooth_atom):

    def __init__(self, sel_alg,
                       coef=1.,
                       offset=None,
                       quadratic=None,
                       nstep=10):

        self.sel_alg = sel_alg

        rr.smooth_atom.__init__(self,
                                (1,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

        self.target_observed = self.sel_alg.target_observed
        self.nactive = self.target_observed.shape[0]
        self.target_cov = self.sel_alg.target_cov

    def solve_approx(self):

        #defining the grid on which marginal conditional densities will be evaluated
        grid_length = 361

        #print("observed values", self.target_observed)
        self.ind_obs = np.zeros(self.nactive, int)
        self.norm = np.zeros(self.nactive)
        self.h_approx = np.zeros((self.nactive, grid_length))
        self.grid = np.zeros((self.nactive, grid_length))

        for j in xrange(self.nactive):
            obs = self.target_observed[j]

            self.grid[j,:] = np.linspace(self.target_observed[j]-18., self.target_observed[j]+18.,num= grid_length)
            grid_j = self.grid[j,:]

            self.norm[j] = self.target_cov[j,j]
            if obs < grid_j[0]:
                self.ind_obs[j] = 0
            elif obs > np.max(grid_j):
                self.ind_obs[j] = grid_length-1
            else:
                self.ind_obs[j] = np.argmin(np.abs(grid_j-obs))

            sys.stderr.write("number of variable being computed: " + str(j) + "\n")
            self.h_approx[j, :] = self.approx_conditional_prob(j)


    def approx_conditional_prob(self, j):
        h_hat = []

        self.sel_alg.setup_map(j)

        for i in xrange(self.grid[j,:].shape[0]):

            approx = approximate_conditional_prob((self.grid[j,:])[i], self.sel_alg)
            val = -(approx.minimize2(step=1., nstep=100)[::-1])[0]
            h_hat.append(val)
            sys.stderr.write("point on grid: " + str(i) + "\n")
            sys.stderr.write("value on grid: " + str(val) + "\n")
        return np.array(h_hat)

    def area_normalized_density(self, j, mean):

        normalizer = 0.
        approx_nonnormalized = []

        for i in xrange(self.grid[j:,].shape[1]):
            approx_density = np.exp(-np.true_divide(((self.grid[j,:])[i] - mean) ** 2, 2 * self.norm[j])
                                    + (self.h_approx[j,:])[i])
            normalizer += approx_density
            approx_nonnormalized.append(approx_density)

        return np.cumsum(np.array(approx_nonnormalized / normalizer))

    def approximate_ci(self, j):

        grid_length = 401
        param_grid = np.linspace(-20,20, num=grid_length)
        area = np.zeros(param_grid.shape[0])

        for k in xrange(param_grid.shape[0]):
            area_vec = self.area_normalized_density(j, param_grid[k])
            area[k] = area_vec[self.ind_obs[j]]

        region = param_grid[(area >= 0.05) & (area <= 0.95)]
        if region.size > 0:
            return np.nanmin(region), np.nanmax(region)
        else:
            return 0, 0

    def approximate_pvalue(self, j, param):

        area_vec = self.area_normalized_density(j, param)
        area = area_vec[self.ind_obs[j]]

        return 2*min(area, 1-area)
