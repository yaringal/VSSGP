# To speed Theano up, create ram disk: mount -t tmpfs -o size=512m tmpfs /mnt/ramdisk
# Then use flag THEANO_FLAGS='base_compiledir=/mnt/ramdisk' python script.py
import sys; sys.path.insert(0, "../Theano"); sys.path.insert(0, "../../Theano")
import theano; import theano.tensor as T; import theano.sandbox.linalg as sT
import numpy as np
import cPickle

print 'Theano version: ' + theano.__version__ + ', base compile dir: ' + theano.config.base_compiledir
theano.config.mode = 'FAST_RUN'
theano.config.optimizer = 'fast_run'
theano.config.reoptimize_unpickled_function = False

class VSSGP:
    def __init__(self, use_exact_A = False):
        try:
            print 'Trying to load model...'
            with open('model_exact_A.save' if use_exact_A else 'model.save', 'rb') as file_handle:
                self.f, self.g = cPickle.load(file_handle)
                print 'Loaded!'
            return
        except:
            print 'Failed. Creating a new model...'

        print 'Setting up variables...'
        Z, mu, lSigma = T.dtensor3s('Z', 'mu', 'lSigma')
        X, Y, m, ls, lhyp, lalpha, lalpha_delta, a = T.dmatrices('X', 'Y', 'm', 'ls', 'lhyp', 'lalpha', 'lalpha_delta', 'a')
        b = T.dvector('b')
        ltau = T.dscalar('ltau')
        Sigma, alpha, alpha_delta, tau = T.exp(lSigma), T.exp(lalpha), T.exp(lalpha_delta), T.exp(ltau)
        alpha = alpha % 2*np.pi
        beta = T.minimum(alpha + alpha_delta, 2*np.pi)
        (N, Q), D, K = X.shape, Y.shape[1], mu.shape[1]
        sf2s, lss, ps = T.exp(lhyp[0]), T.exp(lhyp[1:1+Q]), T.exp(lhyp[1+Q:]) # length-scales abd periods

        print 'Setting up model...'
        if not use_exact_A:
            LL, KL, Y_pred_mean, Y_pred_var, EPhi, EPhiTPhi, opt_A_mean, opt_A_cov = self.get_model(Y, X, Z, alpha, beta,
                mu, Sigma, m, ls, sf2s, lss, ps, tau, a, b, N, Q, D, K)
        else:
            LL, KL, Y_pred_mean, Y_pred_var, EPhi, EPhiTPhi, opt_A_mean, opt_A_cov = self.get_model_exact_A(Y, X, Z, alpha, beta,
                mu, Sigma, m, ls, sf2s, lss, ps, tau, a, b, N, Q, D, K)

        print 'Compiling model...'
        inputs = {'X': X, 'Y': Y, 'Z': Z, 'mu': mu, 'lSigma': lSigma, 'm': m, 'ls': ls, 'lalpha': lalpha,
            'lalpha_delta': lalpha_delta, 'lhyp': lhyp, 'ltau': ltau, 'a': a, 'b': b}
        z = 0.0*sum([T.sum(v) for v in inputs.values()]) # solve a bug with derivative wrt inputs not in the graph
        f = zip(['opt_A_mean', 'opt_A_cov', 'EPhi', 'EPhiTPhi', 'Y_pred_mean', 'Y_pred_var', 'LL', 'KL'],
                [opt_A_mean, opt_A_cov, EPhi, EPhiTPhi, Y_pred_mean, Y_pred_var, LL, KL])
        self.f = {n: theano.function(inputs.values(), f+z, name=n, on_unused_input='ignore') for n,f in f}
        g = zip(['LL', 'KL'], [LL, KL])
        wrt = {'Z': Z, 'mu': mu, 'lSigma': lSigma, 'm': m, 'ls': ls, 'lalpha': lalpha,
            'lalpha_delta': lalpha_delta, 'lhyp': lhyp, 'ltau': ltau, 'a': a, 'b': b}
        self.g = {vn: {gn: theano.function(inputs.values(), T.grad(gv+z, vv), name='d'+gn+'_d'+vn,
            on_unused_input='ignore') for gn,gv in g} for vn, vv in wrt.iteritems()}

        with open('model_exact_A.save' if use_exact_A else 'model.save', 'wb') as file_handle:
            print 'Saving model...'
            sys.setrecursionlimit(2000)
            cPickle.dump([self.f, self.g], file_handle, protocol=cPickle.HIGHEST_PROTOCOL)

    def get_EPhi(self, X, Z, alpha, beta, mu, Sigma, sf2s, lss, ps, K):
        two_over_K = 2.*sf2s[None, None, :]/K # N x K x comp
        mean_p, std_p = ps**-1, (2*np.pi*lss)**-1 # Q x comp
        Ew = std_p[:, None, :] * mu + mean_p[:, None, :] # Q x K x comp
        XBAR = 2 * np.pi * (X[:, :, None, None] - Z[None, :, :, :]) # N x Q x K x comp
        decay = T.exp(-0.5 * ((std_p[None, :, None, :] * XBAR)**2 * Sigma[None, :, :, :]).sum(1)) # N x K x comp

        cos_w = T.cos(alpha + (XBAR * Ew[None, :, :, :]).sum(1)) # N x K x comp
        EPhi = two_over_K**0.5 * decay * cos_w
        EPhi = EPhi.flatten(2) # N x K*comp

        cos_2w = T.cos(2 * alpha + 2 * (XBAR * Ew[None, :, :, :]).sum(1)) # N x K x comp
        E_cos_sq = two_over_K * (0.5 + 0.5*decay**4 * cos_2w) # N x K x comp
        EPhiTPhi = (EPhi.T).dot(EPhi)
        EPhiTPhi = EPhiTPhi - T.diag(T.diag(EPhiTPhi)) + T.diag(E_cos_sq.sum(0).flatten(1))
        return EPhi, EPhiTPhi, E_cos_sq

    def get_opt_A(self, tau, EPhiTPhi, YT_EPhi):
        SigInv = EPhiTPhi + (tau**-1 + 1e-4) * T.identity_like(EPhiTPhi)
        cholTauSigInv = tau**0.5 * sT.cholesky(SigInv)
        invCholTauSigInv = sT.matrix_inverse(cholTauSigInv)
        tauInvSig = invCholTauSigInv.T.dot(invCholTauSigInv)
        Sig_EPhiT_Y = tau * tauInvSig.dot(YT_EPhi.T)
        return Sig_EPhiT_Y, tauInvSig, cholTauSigInv

    def get_model(self, Y, X, Z, alpha, beta, mu, Sigma, m, ls, sf2s, lss, ps, tau, a, b, N, Q, D, K):
        s = T.exp(ls)
        Y = Y - (X.dot(a) + b[None,:])
        EPhi, EPhiTPhi, E_cos_sq = self.get_EPhi(X, Z, alpha, beta, mu, Sigma, sf2s, lss, ps, K)
        YT_EPhi = Y.T.dot(EPhi)

        LL = (-0.5*N*D * np.log(2 * np.pi) + 0.5*N*D * T.log(tau) - 0.5*tau*T.sum(Y**2)
              - 0.5*tau * T.sum(EPhiTPhi * (T.diag(s.sum(1)) + T.sum(m[:,None,:]*m[None,:,:], axis=2)))
              + tau * T.sum((Y.T.dot(EPhi)) * m.T))

        KL_A = 0.5 * (s + m**2 - ls - 1).sum()
        KL_w = 0.5 * (Sigma + mu**2 - T.log(Sigma) - 1).sum()
        KL = KL_A + KL_w

        Y_pred_mean = EPhi.dot(m) + (X.dot(a) + b[None,:])
        Psi = T.sum(E_cos_sq.flatten(2)[:, :, None] * s[None, :, :], 1) # N x K*comp
        flat_diag_n = E_cos_sq.flatten(2) - EPhi**2 # N x K*comp
        Y_pred_var = tau**-1 * T.eye(D) + np.transpose(m.T.dot(flat_diag_n[:, :, None] * m),(1,0,2)) \
                     + T.eye(D)[None, :, :] * Psi[:, :, None]

        opt_A_mean, opt_A_cov, _ = self.get_opt_A(tau, EPhiTPhi, YT_EPhi)
        return LL, KL, Y_pred_mean, Y_pred_var, EPhi, EPhiTPhi, opt_A_mean, opt_A_cov

    def get_model_exact_A(self, Y, X, Z, alpha, beta, mu, Sigma, m, ls, sf2s, lss, ps, tau, a, b, N, Q, D, K):
        Y = Y - (X.dot(a) + b[None,:])
        EPhi, EPhiTPhi, E_cos_sq = self.get_EPhi(X, Z, alpha, beta, mu, Sigma, sf2s, lss, ps, K)
        YT_EPhi = Y.T.dot(EPhi)

        opt_A_mean, opt_A_cov, cholSigInv = self.get_opt_A(tau, EPhiTPhi, YT_EPhi)
        LL = (-0.5*N*D * np.log(2 * np.pi) + 0.5*N*D * T.log(tau) - 0.5*tau*T.sum(Y**2)
               - 0.5*D * T.sum(2*T.log(T.diag(cholSigInv)))
               + 0.5*tau * T.sum(opt_A_mean.T * YT_EPhi))

        KL_w = 0.5 * (Sigma + mu**2 - T.log(Sigma) - 1).sum()

        ''' For prediction, m is assumed to be [m_1, ..., m_d] with m_i = opt_a_i, and and ls = opt_A_cov  '''
        Y_pred_mean = EPhi.dot(m) + (X.dot(a) + b[None,:])
        EphiTphi = EPhi[:, :, None] * EPhi[:, None, :] # N x K*comp x K*comp
        comp = sf2s.shape[0]
        EphiTphi = EphiTphi - T.eye(K*comp)[None, :, :] * EphiTphi + T.eye(K*comp)[None, :, :] * E_cos_sq.flatten(2)[:, :, None]
        Psi = T.sum(T.sum(EphiTphi * ls[None, :, :], 2), 1) # N
        flat_diag_n = E_cos_sq.flatten(2) - EPhi**2 # N x K*comp
        Y_pred_var = tau**-1 * T.eye(D) + np.transpose(m.T.dot(flat_diag_n[:, :, None] * m),(1,0,2)) \
                     + T.eye(D)[None, :, :] * Psi[:, None, None]

        return LL, KL_w, Y_pred_mean, Y_pred_var, EPhi, EPhiTPhi, opt_A_mean, opt_A_cov
