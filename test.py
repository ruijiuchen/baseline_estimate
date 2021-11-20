#!/usr/bin/env python3
# −*− coding:utf-8 −*−

import numpy as np
import matplotlib.pyplot as plt
import time

from nonparams_est import NONPARAMS_EST
from params_est import PARAMS_EST

bg_linear = lambda x, B_params: B_params[0]*x + B_params[1] 
bg_lorentz = lambda x, fc, B_params: B_params[0] + B_params[1] / (1 + B_params[2] * (x - fc)**2) 
peak_gen = lambda x, P_info: np.sum([P_params[0] * np.exp(-(x - P_params[1])**2 / P_params[2]**2 / 2) for P_params in P_info], axis=0)
r_sqr_non_params = lambda y, y_hat: 1 - np.sum((y - y_hat)**2) / np.std(y, ddof=len(y)-1) # R^2 goodness of fit test
r_sqr_params = lambda y, y_hat, p: 1 - np.sum((y - y_hat)**2) / (len(y) - p - 1) / np.std(y, ddof=1) # R^2 goodness of fit test

'''data for simulation'''
n_sig, P_info = .01, [[10, -1, 0.5], [2, 10, 0.3], [5, 12, 0.4],[15, 30, .1], [3, 39, 0.2], [2, 42, 0.1], [20, 50, 0.6], [14, 60, 0.5]]
sim_x = np.arange(-10, 100, 0.01)
sim_bg_linear = bg_linear(sim_x, [0.2, 0.4])
sim_linear = peak_gen(sim_x, P_info=P_info) + sim_bg_linear + np.random.normal(scale=n_sig, size=len(sim_x))
sim_bg_lorentz = bg_lorentz(sim_x, 40., [0.1, 4., 0.01])
sim_lorentz = peak_gen(sim_x, P_info=P_info) + sim_bg_lorentz + np.random.normal(scale=n_sig, size=len(sim_x))

# est initial
nonparams_base_est_linear = NONPARAMS_EST(sim_linear)
nonparams_base_est_lorentz = NONPARAMS_EST(sim_lorentz)
params_base_est_linear = PARAMS_EST(sim_x, sim_linear)
params_base_est_lorentz = PARAMS_EST(sim_x, sim_lorentz)

'''test for SNIP method'''
## est for lorentz
#linear_snip = nonparams_base_est_lorentz.snip(m=80)
#linear_snip_threshold = nonparams_base_est_lorentz.snip(method='threshold', m=80, sig=1e-3)
#linear_snip_sort = nonparams_base_est_lorentz.snip(method='sort', m=80, sort=8e-1)
#linear_snip_auto, _ = nonparams_base_est_lorentz.snip(method='auto', scale=2.6e-3)
#
#fig, ax = plt.subplots(2, 1)
#ax[0].plot(sim_x, sim_lorentz, color='lightgray')
#ax[0].plot(sim_x, linear_snip, label='SNIP')
#ax[0].plot(sim_x, linear_snip_threshold, label='SNIP-threshold')
#ax[0].plot(sim_x, linear_snip_sort, label='SNIP-sort')
#ax[0].plot(sim_x, linear_snip_auto, label='SNIP-auto')
#ax[0].set_xlim([sim_x.min(),sim_x.max()])
#ax[0].legend()
#ax[1].plot(sim_x, np.zeros(len(sim_x)), color='lightgray')
#ax[1].fill_between(sim_x, -n_sig*np.ones(len(sim_x)), n_sig*np.ones(len(sim_x)), alpha=.2, color='lightgray')
#ax[1].plot(sim_x, linear_snip - sim_bg_lorentz, label='SNIP')
#ax[1].plot(sim_x, linear_snip_threshold - sim_bg_lorentz, label='SNIP-threshold')
#ax[1].plot(sim_x, linear_snip_sort - sim_bg_lorentz, label='SNIP-sort')
#ax[1].plot(sim_x, linear_snip_auto - sim_bg_lorentz, label='SNIP-auto')
#ax[1].set_xlim([sim_x.min(),sim_x.max()])
#ax[1].legend()
#plt.show()


'''test for PLS method'''
# est for linear
linear_AsLS = nonparams_base_est_linear.pls(method='AsLS', l=1e7, p=0.001, niter=10)
linear_airPLS = nonparams_base_est_linear.pls(method='airPLS', l=1e7, nitermax=10)
linear_arPLS = nonparams_base_est_linear.pls(method='arPLS', l=4e9, ratio=1e-6)
linear_BrPLS = nonparams_base_est_linear.pls(method='BrPLS', l=4e9, ratio=1e-6)
linear_AsLS_R_sqr = r_sqr_non_params(sim_bg_linear, linear_AsLS)
linear_airPLS_R_sqr = r_sqr_non_params(sim_bg_linear, linear_airPLS)
linear_arPLS_R_sqr = r_sqr_non_params(sim_bg_linear, linear_arPLS)
linear_BrPLS_R_sqr = r_sqr_non_params(sim_bg_linear, linear_BrPLS)

# est for lorentz
lorentz_AsLS = nonparams_base_est_lorentz.pls(method='AsLS', l=1e7, p=0.001, niter=10)
lorentz_airPLS = nonparams_base_est_lorentz.pls(method='airPLS', l=1e7, nitermax=10)
lorentz_arPLS = nonparams_base_est_lorentz.pls(method='arPLS', l=1e9, ratio=1e-6)
lorentz_BrPLS = nonparams_base_est_lorentz.pls(method='BrPLS', l=1e9, ratio=1e-6)
lorentz_AsLS_R_sqr = r_sqr_non_params(sim_bg_lorentz, lorentz_AsLS)
lorentz_airPLS_R_sqr = r_sqr_non_params(sim_bg_lorentz, lorentz_airPLS)
lorentz_arPLS_R_sqr = r_sqr_non_params(sim_bg_lorentz, lorentz_arPLS)
lorentz_BrPLS_R_sqr = r_sqr_non_params(sim_bg_lorentz, lorentz_BrPLS)

fig, ax = plt.subplots(2, 2)
ax[0,0].plot(sim_x, sim_linear, color='lightgray')
ax[0,0].plot(sim_x, sim_bg_linear, linestyle=(0,(5,1)), color='dimgray', label='baseline')
ax[0,0].plot(sim_x, linear_AsLS, linestyle=(0,(1,1)), color='green', label="{:}, $R^2$={:.4f}".format('AsLS', linear_AsLS_R_sqr))
ax[0,0].plot(sim_x, linear_airPLS, linestyle='dashed', color='crimson', label="{:}, $R^2$={:.4f}".format('airPLS', linear_airPLS_R_sqr))
ax[0,0].plot(sim_x, linear_arPLS, linestyle='dashdot', color='limegreen', label="{:}, $R^2$={:.4f}".format('arPLS', linear_arPLS_R_sqr))
ax[0,0].plot(sim_x, linear_BrPLS, linestyle=(0,(3,1,1,1)), color='darkorange', label="{:}, $R^2={:.4f}$".format('BrPLS', linear_BrPLS_R_sqr))
ax[0,0].set_xlim([sim_x.min(),sim_x.max()])
ax[0,0].legend()
ax[0,1].plot(sim_x, sim_lorentz, color='lightgray')
ax[0,1].plot(sim_x, sim_bg_lorentz, linestyle=(0,(5,1)), color='dimgray', label='baseline')
ax[0,1].plot(sim_x, lorentz_AsLS, linestyle=(0,(1,1)), color='green', label="{:}, $R^2={:.4f}$".format('AsLS', lorentz_AsLS_R_sqr))
ax[0,1].plot(sim_x, lorentz_airPLS, linestyle='dashed', color='crimson', label="{:}, $R^2={:.4f}$".format('airPLS', lorentz_airPLS_R_sqr))
ax[0,1].plot(sim_x, lorentz_arPLS, linestyle='dashdot', color='limegreen', label="{:}, $R^2={:.4f}$".format('arPLS', lorentz_arPLS_R_sqr))
ax[0,1].plot(sim_x, lorentz_BrPLS, linestyle=(0,(3,1,1,1)), color='darkorange', label="{:}, $R^2={:.4f}$".format('BrPLS', lorentz_BrPLS_R_sqr))
ax[0,1].set_xlim([sim_x.min(),sim_x.max()])
ax[0,1].legend()

ax[1,0].plot(sim_x, np.zeros_like(sim_x), color='lightgray')
ax[1,0].fill_between(sim_x, -n_sig*np.ones(len(sim_x)), n_sig*np.ones(len(sim_x)), alpha=.2, color='lightgray')
ax[1,0].plot(sim_x, linear_AsLS - sim_bg_linear, linestyle=(0,(1,1)), color='green', label="{:}, $R^2_{{bas}}$={:.4f}".format('AsLS', np.std(sim_bg_linear - linear_AsLS)))
ax[1,0].plot(sim_x, linear_airPLS - sim_bg_linear, linestyle='dashed', color='crimson', label="{:}, $R^2_{{bas}}$={:.4f}".format('airPLS', np.std(sim_bg_linear - linear_airPLS)))
ax[1,0].plot(sim_x, linear_arPLS - sim_bg_linear, linestyle='dashdot', color='limegreen', label="{:}, $R^2_{{bas}}$={:.4f}".format('arPLS', np.std(sim_bg_linear - linear_arPLS)))
ax[1,0].plot(sim_x, linear_BrPLS - sim_bg_linear, linestyle=(0,(3,1,1,1)), color='darkorange', label="{:}, $R^2_{{bas}}={:.4f}$".format('BrPLS', np.std(sim_bg_linear - linear_BrPLS)))
ax[1,0].set_xlim([sim_x.min(),sim_x.max()])
ax[1,0].legend()
ax[1,1].plot(sim_x, np.zeros_like(sim_x), color='lightgray')
ax[1,1].fill_between(sim_x, -n_sig*np.ones(len(sim_x)), n_sig*np.ones(len(sim_x)), alpha=.2, color='lightgray')
ax[1,1].plot(sim_x, lorentz_AsLS - sim_bg_lorentz, linestyle=(0,(1,1)), color='green', label="{:}, $R^2_{{bas}}={:.4f}$".format('AsLS', np.std(sim_bg_lorentz - lorentz_AsLS)))
ax[1,1].plot(sim_x, lorentz_airPLS - sim_bg_lorentz, linestyle='dashed', color='crimson', label="{:}, $R^2_{{bas}}={:.4f}$".format('airPLS', np.std(sim_bg_lorentz - lorentz_airPLS)))
ax[1,1].plot(sim_x, lorentz_arPLS - sim_bg_lorentz, linestyle='dashdot', color='limegreen', label="{:}, $R^2_{{bas}}={:.4f}$".format('arPLS', np.std(sim_bg_lorentz - lorentz_arPLS)))
ax[1,1].plot(sim_x, lorentz_BrPLS - sim_bg_lorentz, linestyle=(0,(3,1,1,1)), color='darkorange', label="{:}, $R^2_{{bas}}={:.4f}$".format('BrPLS', np.std(sim_bg_lorentz - lorentz_BrPLS)))
ax[1,1].set_xlim([sim_x.min(),sim_x.max()])
ax[1,1].legend()

plt.show()

'''test for bayesian params method'''
#params, sigma = [40., 0.1, 4., 0.01], .1
#params_range = [[-160, 240], [-0.1, 0.2], [1., 10.], [1e-6, 1e2]]
#import time
#time0 = time.time()
#lorentz_bayesian = params_base_est_lorentz.bayesian(method='plus', func='lorentzian', params=params, params_range=params_range, sigma=sigma, verbose=True)
#print("time: {:} s".format(time.time() - time0))
##lorentz_bayesian = params_base_est_lorentz.bayesian(func='lorentzian', params=params, params_range=params_range, sigma=sigma, verbose=True)
#
#fig, ax = plt.subplots(2, 1)
#ax[0].plot(sim_x, sim_lorentz, label='original data', color='lightgray')
#ax[0].plot(sim_x, lorentz_bayesian, label='estimated background')
#ax[0].legend()
#ax[0].set_xlim([sim_x.min(),sim_x.max()])
#ax[1].plot(sim_x, np.zeros(len(sim_x)), label='baseline', color='lightgray')
#ax[1].fill_between(sim_x, -n_sig*np.ones(len(sim_x)), n_sig*np.ones(len(sim_x)), alpha=.2, color='lightgray')
#ax[1].plot(sim_x, lorentz_bayesian - sim_bg_lorentz, label='removed background')
#ax[1].legend()
#ax[1].set_xlim([sim_x.min(),sim_x.max()])
#plt.show()
