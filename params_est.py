#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import sys, nlopt, warnings
import numpy as np
import scipy.special as sp

class PARAMS_EST(object):
    '''
    parameter baseline estimation
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def bayesian(self, method='', verbose=False, **kwargs):
        bayesian_method = BAYESIAN(self.x, self.y)
        if method == 'plus':
            try:
                return bayesian_method._plus(verbose=verbose, **kwargs)

            except:
                raise ValueError("Invalid args input for BAYESIAN method")
                sys.exit()
        else:
            try:
                return bayesian_method._default(verbose=verbose, **kwargs)
            except:
                raise ValueError("Invalid args input for BAYESIAN method")
                sys.exit()

class ESTFUNC(object):
    '''
    background function to be estimate
    '''
    def __init__(self, x):
        self.x = x

    def linear(self, params=[1e-1, 5e-2]):
        return params[0] + self.x * params[1]

    def quadratic(self, params=[1e-1, 5e-2, 2e-3]):
        return params[0] + self.x * params[1] + self.x**2 * params[2]

    def cubic(self, params=[1e-1, 5e-2, 2e-3, 3e-4]):
        return params[0] + self.x * params[1] + self.x**2 * params[2] + self.x**3 * params[3]

    def lorentzian(self, params=[10, 1e0, 1e-1, 4e-2]):
        return params[1] + params[2] / (1 + (self.x - params[0])**2 * params[3])

    def landaugauss(self, params=[15, 15, .1, 4.]):
        '''require pylandau library'''
        import pylandau
        return pylandau.langau(self.x, mpv=params[0], eta=params[1], sigma=params[2], A=params[3])

    def custom_func(self, params=[]):
        '''user-defined function'''
        return
    


class BAYESIAN(object):
    '''
    Bayesian approach for background estimation
    @ R. Fischer, et. al., Phys. Rev. E., 2000, 61(2).
    @ W. von der Linden, et. al., Phys. Rev. E., 1999, 59(6).
    ---
    this sample is for linear, quadratic, cubic, Lorentzian and landau-gaussian (require pylandau library) background
    user can write their own following this structure
    '''
    def __init__(self, x, y):
        '''
        initial
        ---
        x:         range of x-coordinate values
        y:         range of y-coordinate values (values >=0)
        '''
        self.x = x
        self.y = y
        self.est_func = ESTFUNC(x)

    def _default(self, func='linear', params=[], params_range=[], sigma=1e-2, mu=None, beta=0.5, verbose=False, opt_algorithm=nlopt.LN_COBYLA):
        '''
        background probability user-defined version.
        finding the parameters that maximize the posterior
        ---
        func:       background function to estimate,
                    'linear', 'quadratic', 'cubic', 'lorentzian', 'landaugauss' (require pylandau library), 
                    any other custom function: set the same name of function in class ESTFUNC as a defined function
        params:     parameters of background function to be estimated, initial values
                    [param_0, param_1, ...]
        params_range:
                    lower and upper bounds of the parameters
                    [[param_0_lower, param_0_upper], [param_1_lower, param_1_upper], ...]
        sigma:      noise scale
        mu:         scale of datum, nuisance parameter can be modified with partical, default average of data list 
        beta:       chance that a datum contains a signal, nuisance parameter can be modified with partical, default 0.5 for noisy signal
        verbose:    True for print fitting result, including result code of nlopt, estimated parameter values of background function, and sigma
        ---
        return
                    estimated parameters of function
        '''
        est_func = getattr(self.est_func, func)
        mu = np.mean(self.y) if mu == None else mu
        initial = params.copy()
        initial.append(sigma)
        lower_bound = [values[0] for values in params_range]
        lower_bound.append(sigma*1e-3)
        upper_bound = [values[1] for values in params_range]
        upper_bound.append(sigma*1e3)
        warnings.filterwarnings("ignore")
        opt = nlopt.opt(opt_algorithm, len(params)+1)
        def opt_f(x, beta, mu): 
            params, sigma, beta, mu = x[:-1], x[-1], beta, mu
            diff_yb = self.y - est_func(params)
            return np.sum(np.log((1 - beta) / sigma / np.sqrt(2 * np.pi) * np.exp(-diff_yb**2 / sigma**2 / 2) + beta / mu / 2 * (1 + sp.erf((diff_yb - sigma**2 / mu) / sigma / np.sqrt(2))) * np.exp(- diff_yb / mu + sigma**2 / mu**2 / 2)))
        opt.set_max_objective(lambda x, grad, beta=beta, mu=mu: opt_f(x, beta, mu))
        opt.set_lower_bounds(lower_bound)
        opt.set_upper_bounds(upper_bound)
        opt.set_xtol_rel(1e-32)
        opt.set_ftol_rel(1e-32)
        x = opt.optimize(initial)
        if verbose: print("result code = {:}\noptimum at params = {:}, sigma= {:}".format(opt.last_optimize_result(), x[:-1], x[-1]))
        return est_func(x[:-1])

    def _plus(self, func='linear', params=[], params_range=[], sigma=1e-2, mu=None, verbose=False, opt_algorithm=nlopt.LN_COBYLA):
        '''
        background probability user-defined version.
        finding the parameters that maximize the posterior
        ---
        func:       background function to estimate,
                    'linear', 'quadratic', 'cubic', 'lorentzian', 'landaugauss' (require pylandau library)
        params:     parameters of background function to be estimated, initial values
                    [param_0, param_1, ...]
        params_range:
                    lower and upper bounds of the parameters
                    [[param_0_lower, param_0_upper], [param_1_lower, param_1_upper], ...]
        sigma:      noise scale
        mu:         scale of datum, nuisance parameter can be modified with partical, default average of data list 
        verbose:    True for print fitting result, including result code of nlopt, estimated parameter values of background function, and sigma
        ---
        return
                    estimated parameters of function
        '''
        est_func = getattr(self.est_func, func)
        mu = np.mean(self.y) if mu == None else mu
        initial = params.copy()
        initial.append(sigma)
        initial.append(0.5)
        lower_bound = [values[0] for values in params_range]
        lower_bound.append(sigma*1e-3)
        lower_bound.append(.01)
        upper_bound = [values[1] for values in params_range]
        upper_bound.append(sigma*1e3)
        upper_bound.append(.99)
        warnings.filterwarnings("ignore")
        opt = nlopt.opt(opt_algorithm, len(params)+2)
        def opt_f(x, mu): 
            params, sigma, beta, mu = x[:-2], x[-2], x[-1], mu
            diff_yb = self.y - est_func(params)
            return np.sum(np.log((1 - beta) / sigma / np.sqrt(2 * np.pi) * np.exp(-diff_yb**2 / sigma**2 / 2) + beta / mu / 2 * (1 + sp.erf((diff_yb - sigma**2 / mu) / sigma / np.sqrt(2))) * np.exp(- diff_yb / mu + sigma**2 / mu**2 / 2)))
        def opt_constraint(x, mu):
            params, sigma, beta, mu = x[:-2], x[-2], x[-1], mu
            diff_yb = self.y - est_func(params)
            return np.abs(np.mean(1 / (1 + beta / (1 - beta) * np.sqrt(np.pi / 2) * np.abs(sigma / mu * (1 + sp.erf((diff_yb / sigma - sigma / mu) / np.sqrt(2))) * np.exp((diff_yb / sigma - sigma / mu)**2 / 2)))) +  beta - 1.)
        opt.set_max_objective(lambda x, grad, mu=mu: opt_f(x, mu))
        opt.add_equality_constraint(lambda x, grad, mu=mu: opt_constraint(x, mu), 1e-6)
        opt.set_lower_bounds(lower_bound)
        opt.set_upper_bounds(upper_bound)
        opt.set_xtol_rel(1e-32)
        opt.set_ftol_rel(1e-32)
        x = opt.optimize(initial)
        if verbose: print("result code = {:}\noptimum at params = {:}, sigma= {:}, beta={:}".format(opt.last_optimize_result(), x[:-2], x[-2], x[-1]))
        return est_func(x[:-2])

