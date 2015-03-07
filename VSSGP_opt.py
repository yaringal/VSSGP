import numpy as np
from VSSGP_model import VSSGP
import pylab
import multiprocessing
def extend(x, y, z = {}):
    return dict(x.items() + y.items() + z.items())
pool, global_f, global_g = None, None, None
def eval_f_LL((X, Y, params)):
    return global_f['LL'](**extend({'X': X, 'Y': Y}, params))
def eval_g_LL((name, X, Y, params)):
    return global_g[name]['LL'](**extend({'X': X, 'Y': Y}, params))

class VSSGP_opt():
    def __init__(self, N, Q, D, K, inputs, opt_params, fixed_params, use_exact_A = False, test_set = {},
                 parallel = False, batch_size = None, components = None, print_interval = None):
        self.vssgp, self.N, self.Q, self.K, self.fixed_params = VSSGP(use_exact_A), N, Q, K, fixed_params
        self.use_exact_A, self.parallel, self.batch_size = use_exact_A, parallel, batch_size
        self.inputs, self.test_set = inputs, test_set
        self.print_interval = 10 if print_interval is None else print_interval
        self.opt_param_names = [n for n,_ in opt_params.iteritems()]
        opt_param_values = [np.atleast_2d(opt_params[n]) for n in self.opt_param_names]
        self.shapes = [v.shape for v in opt_param_values]
        self.sizes = [sum([np.prod(x) for x in self.shapes[:i]]) for i in xrange(len(self.shapes)+1)]
        self.components = opt_params['lSigma'].shape[2] if components is None else components
        self.colours = [np.random.rand(3,1) for c in xrange(self.components)]
        self.callback_counter = [0]
        if 'train_ind' not in test_set:
            print 'train_ind not found!'
            self.test_set['train_ind'] = np.arange(inputs['X'].shape[0]).astype(int)
            self.test_set['test_ind'] = np.arange(0).astype(int)
        if batch_size is not None:
            if parallel:
                global pool, global_f, global_g
                global_f, global_g = self.vssgp.f, self.vssgp.g
                pool = multiprocessing.Pool(int(self.N / self.batch_size))
            else:
                self.params = np.concatenate([v.flatten() for v in opt_param_values])
                self.param_updates = np.zeros_like(self.params)
                self.moving_mean_squared = np.zeros_like(self.params)
                self.learning_rates = 1e-2*np.ones_like(self.params)


    def unpack(self, x):
        x_param_values = [x[self.sizes[i-1]:self.sizes[i]].reshape(self.shapes[i-1]) for i in xrange(1,len(self.shapes)+1)]
        params = {n:v for (n,v) in zip(self.opt_param_names, x_param_values)}
        if 'ltau' in params:
            params['ltau'] = params['ltau'].squeeze()
        return params

    def func(self, x):
        params = extend(self.fixed_params, self.unpack(x))
        if self.batch_size is not None:
            X, Y, splits = self.inputs['X'], self.inputs['Y'], int(self.N / self.batch_size)
            if self.parallel:
                arguments = [(X[i::splits], Y[i::splits], params) for i in xrange(splits)]
                LL = sum(pool.map_async(eval_f_LL, arguments).get(9999999))
                KL = self.vssgp.f['KL'](**extend({'X': [[0]], 'Y': [[0]]}, params))
            else:
                split = np.random.randint(splits)
                LL = self.N / self.batch_size * self.vssgp.f['LL'](**extend({'X': X[split::splits], 'Y': Y[split::splits]}, params))
                print LL
                KL = self.vssgp.f['KL'](**extend({'X': [[0]], 'Y': [[0]]}, params))
        else:
            params = extend(self.inputs, params)
            LL, KL = self.vssgp.f['LL'](**params), self.vssgp.f['KL'](**params)
        return -(LL - KL)

    def fprime(self, x):
        grads, params = [], extend(self.fixed_params, self.unpack(x))
        for n in self.opt_param_names:
            if self.batch_size is not None:
                X, Y, splits = self.inputs['X'], self.inputs['Y'], int(self.N / self.batch_size)
                if self.parallel:
                    arguments = [(n, X[i::splits], Y[i::splits], params) for i in xrange(splits)]
                    dLL = sum(pool.map_async(eval_g_LL, arguments).get(9999999))
                    dKL = self.vssgp.g[n]['KL'](**extend({'X': [[0]], 'Y': [[0]]}, params))
                else:
                    split = np.random.randint(splits)
                    dLL = self.N / self.batch_size * self.vssgp.g[n]['LL'](**extend({'X': X[split::splits], 'Y': Y[split::splits]}, params))
                    dKL = self.vssgp.g[n]['KL'](**extend({'X': [[0]], 'Y': [[0]]}, params))
            else:
                params = extend(self.inputs, params)
                dLL, dKL = self.vssgp.g[n]['LL'](**params), self.vssgp.g[n]['KL'](**params)
            grads += [-(dLL - dKL)]
        return np.concatenate([grad.flatten() for grad in grads])

    def plot_func(self, X, Y, plot_test):
        vis_ind = self.test_set['test_ind'] if plot_test else self.test_set['train_ind']
        N = self.test_set['train_ind'].shape[0] + self.test_set['test_ind'].shape[0]
        invis_ind = np.setdiff1d(np.arange(N), vis_ind)
        x, y = np.empty(N), np.empty(N)
        x[invis_ind], y[invis_ind] = np.nan, np.nan
        x[vis_ind], y[vis_ind] = X[:,0], Y[:,0]
        pylab.plot(x, y, c="#a40000" if not plot_test else "#4e9a06")

    def plot_predict(self, X, params, plot_test):
        inputs = {'X': X, 'Y': [[0]]}
        params = extend(inputs, self.fixed_params, params)
        mean = self.vssgp.f['Y_pred_mean'](**params)[:,0]
        std = self.vssgp.f['Y_pred_var'](**params)[:,0,0]**0.5
        lower_bound, upper_bound = mean - 2*std, mean + 2*std
        vis_ind = self.test_set['test_ind'] if plot_test else self.test_set['train_ind']
        N = self.test_set['train_ind'].shape[0] + self.test_set['test_ind'].shape[0]
        invis_ind = np.setdiff1d(np.arange(N), vis_ind)
        x, y, y1, y2 = np.empty(N), np.empty(N), np.empty(N), np.empty(N)
        x[invis_ind], y[invis_ind], y1[invis_ind], y2[invis_ind] = np.nan, np.nan, np.nan, np.nan
        x[vis_ind], y[vis_ind], y1[vis_ind], y2[vis_ind] = X[:,0], mean, lower_bound, upper_bound
        pylab.plot(x, y, c="#204a87")
        pylab.fill_between(x, y1, y2, facecolor="#3465a4", color='w', alpha=0.25)

    def callback(self, x):
        if self.callback_counter[0]%self.print_interval == 0:
            opt_params = self.unpack(x)
            params = extend(self.inputs, self.fixed_params, opt_params)

            if self.use_exact_A:
                opt_A_mean = self.vssgp.f['opt_A_mean'](**params)
                opt_A_cov = self.vssgp.f['opt_A_cov'](**params)
                if 'm' in self.fixed_params:
                    self.fixed_params['m'] = opt_A_mean
                    self.fixed_params['ls'] = opt_A_cov
                else:
                    opt_params['m'] = opt_A_mean
                    opt_params['ls'] = opt_A_cov

            pylab.clf()
            pylab.subplot(3, 1, 1)
            self.plot_func(params['X'], params['Y'], False)
            self.plot_predict(self.inputs['X'], opt_params, False)
            if 'X' in self.test_set:
                self.plot_func(self.test_set['X'], self.test_set['Y'], True)
                self.plot_predict(self.test_set['X'], opt_params, True)
            for c in xrange(self.components):
                pylab.scatter(params['Z'][0,:,c], 0*params['Z'][0,:,c], c=self.colours[c], zorder=3, edgecolors='none')

            hyp = np.exp(params['lhyp'].copy())
            sf2s = hyp[0]
            lss = hyp[1:1+self.Q]
            ps = hyp[1+self.Q:]
            mean_p, std_p = ps**-1, (2*np.pi*lss)**-1 # Q x comp
            mu, Sigma = params['mu'].copy(), np.exp(params['lSigma'].copy())
            min_mean = (std_p[None, :] * mu[0, :, :] + mean_p[None, :]).min()
            max_mean = (std_p[None, :] * mu[0, :, :] + mean_p[None, :]).max()
            min_std = (std_p[None, :] * Sigma[0, :, :]).max()**0.5
            max_std = (std_p[None, :] * Sigma[0, :, :]).max()**0.5
            linspace = np.linspace(min_mean-2*min_std, max_mean+2*max_std, 1000)

            pylab.subplot(3, 1, 2)
            for c in xrange(self.components):
                pdf = pylab.normpdf(linspace,mean_p[:,c],np.min(std_p[:,c],1e-5))
                pylab.plot(linspace,pdf,c=self.colours[c], linewidth=1.0)
            pylab.ylim(0,100)

            pylab.subplot(3, 1, 3)
            for c in xrange(self.components):
                for (mean, std) in zip(mu[0,:,c], Sigma[0,:,c]**0.5):
                    pdf = pylab.normpdf(linspace,std_p[:,c]*mean+mean_p[:,c],np.min(std_p[:,c]*std,1e-5))
                    pylab.plot(linspace,pdf,c=self.colours[c], linewidth=1.0)
            pylab.ylim(0,100)
            pylab.draw()

            print 'sf2 = ' + str(sf2s.squeeze())
            print 'l = ' + str(lss.squeeze())
            print 'p = ' + str(ps.squeeze())
            print 'tau = ' + str(np.exp(params['ltau']))
            print 'mu = ' 
            print params['mu'][:,:5,:]
            print 'Sigma = ' 
            print np.exp(params['lSigma'][:,:5,:])
            print 'm = ' 
            print params['m'][:5,:].T
            print 's = ' 
            print np.exp(params['ls'][:5,:].T)
            print 'a = ' + str(params['a']) + ', b = ' + str(params['b'])
            print 'EPhi = '
            EPhi = self.vssgp.f['EPhi'](**params)
            print EPhi[:5,:5]
            LL = self.vssgp.f['LL'](**params)
            KL = self.vssgp.f['KL'](**params)
            print LL - KL
        self.callback_counter[0] += 1


    def rmsprop_one_step(self, mask, decay = 0.9, momentum = 0, learning_rate_adapt = 0.05,
            learning_rate_min = 1e-6, learning_rate_max = 10):
        # RMSPROP: Tieleman, T. and Hinton, G. (2012), Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning
        # Implementation based on https://github.com/BRML/climin/blob/master/climin/rmsprop.py

        # We use Nesterov momentum: first, we make a step according to the momentum and then we calculate the gradient.
        step1 = self.param_updates * momentum
        self.params[mask] += step1[mask]
        grad = -self.fprime(self.params)

        self.moving_mean_squared[mask] = (decay * self.moving_mean_squared[mask] + (1 - decay) * grad[mask] ** 2)
        step2 = self.learning_rates * grad / (self.moving_mean_squared + 1e-8)**0.5
        self.params[mask] += step2[mask]

        step = step1 + step2

        # Step rate adaption. If the current step and the momentum agree, we slightly increase the step rate for that dimension.
        if learning_rate_adapt:
            # This code might look weird, but it makes it work with both numpy and gnumpy.
            step_non_negative = step > 0
            step_before_non_negative = self.param_updates > 0
            agree = (step_non_negative == step_before_non_negative) * 1.
            adapt = 1 + agree * learning_rate_adapt * 2 - learning_rate_adapt
            self.learning_rates[mask] *= adapt[mask]
            self.learning_rates[mask] = np.clip(self.learning_rates[mask], learning_rate_min, learning_rate_max)

        self.param_updates[mask] = step[mask]
