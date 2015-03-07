from VSSGP_opt import VSSGP_opt
from scipy.optimize import minimize
import numpy as np
from numpy.random import randn, rand
np.set_printoptions(precision=2, suppress=True)
import pylab; pylab.ion() # turn interactive mode on

N, Q, D, K = 250, 1, 1, 25
components, init_period, init_lengthscales, sf2s, tau = 2, 1e32, 1, np.array([1, 5]), 1

# Some synthetic data to play with
X = rand(N,Q) * 5*np.pi 
X = np.sort(X, axis=0)
Z = rand(Q,K,components) * 5*np.pi 
#a, b, c, d, e, f = randn(), randn(), randn(), randn(), randn(), randn()
#a, b, c, d, e, f = 0.6, 0.7, -0.6, 0.5, -0.1, -0.8
#a, b, c, d, e, f = -0.6, -0.3, -0.6, 0.6, 0.7, 0.6
#a, b, c, d, e, f = -0.5, -0.3, -0.6, 0.1, 1.1, 0.1
a, b, c, d, e, f = 0.6, -1.8, -0.5, -0.5, 1.7, 0
Y = a*np.sin(b*X+c) + d*np.sin(e*X+f)

# Initialise near the posterior:
mu = randn(Q,K,components)
# TODO: Currently tuned by hand to smallest value that doesn't diverge; we break symmetry to allow for some to get very small while others very large
feature_lengthscale = 5 # features are non-diminishing up to feature_lengthscale / lengthscale from z / lengthscale
lSigma = np.log(randn(Q,K,components)**2 / feature_lengthscale**2) # feature weights are np.exp(-0.5 * (x-z)**2 * Sigma / lengthscale**2)
lalpha = np.log(rand(K,components)*2*np.pi)
lalpha_delta = np.log(rand(K,components) * (2*np.pi - lalpha))
m = randn(components*K,D)
ls = np.zeros((components*K,D)) - 4
lhyp = np.log(1 + 1e-2*randn(2*Q+1, components)) # break symmetry
lhyp[0,:] += np.log(sf2s) # sf2
lhyp[1:Q+1,:] += np.log(init_lengthscales) # length-scales
lhyp[Q+1:,:] += np.log(init_period) # period
ltau = np.log(tau) # precision
lstsq = np.linalg.lstsq(np.hstack([X, np.ones((N,1))]), Y)[0]
a = 0*np.atleast_2d(lstsq[0]) # mean function slope
b = 0*lstsq[1] # mean function intercept

opt_params = {'Z': Z, 'm': m, 'ls': ls, 'mu': mu, 'lSigma': lSigma, 'lhyp': lhyp, 'ltau': ltau}
fixed_params = {'lalpha': lalpha, 'lalpha_delta': lalpha_delta, 'a': a, 'b': b}
inputs = {'X': X, 'Y': Y}
vssgp_opt = VSSGP_opt(N, Q, D, K, inputs, opt_params, fixed_params, use_exact_A=True, print_interval=1)

# LBFGS
x0 = np.concatenate([np.atleast_2d(opt_params[n]).flatten() for n in vssgp_opt.opt_param_names])
pylab.figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='w')
vssgp_opt.callback(x0)
res = minimize(vssgp_opt.func, x0, method='L-BFGS-B', jac=vssgp_opt.fprime,
	options={'ftol': 0, 'disp': False, 'maxiter': 500}, tol=0, callback=vssgp_opt.callback)

raw_input("PRESS ENTER TO CONTINUE.")