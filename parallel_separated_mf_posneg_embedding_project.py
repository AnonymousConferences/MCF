import os
import sys
import time
import threading
import numpy as np
from numpy import linalg as LA
import ParallelSolver as ps
import MultiProcessParallelSolverSeparate as mpps
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin

import rec_eval


class SeparatedParallelMFPositiveNegativeProjectEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, mu_p_p = 1, mu_p_n = 1, n_components=100, max_iter=10, batch_size=1000,
                 init_std=0.01, dtype='float32', n_jobs=8, random_state=None,
                 save_params=False, save_dir='.', early_stopping=False,
                 verbose=False, **kwargs):
        '''
        Parameters
        ---------
        n_components : int
            Number of latent factors
        max_iter : int
            Maximal number of iterations to perform
        batch_size: int
            Batch size to perform parallel update
        init_std: float
            The latent factors will be initialized as Normal(0, init_std**2)
        dtype: str or type
            Data-type for the parameters, default 'float32' (np.float32)
        n_jobs: int
            Number of parallel jobs to update latent factors
        random_state : int or RandomState
            Pseudo random number generator used for sampling
        save_params: bool
            Whether to save parameters after each iteration
        save_dir: str
            The directory to save the parameters
        early_stopping: bool
            Whether to early stop the training by monitoring performance on
            validation set
        verbose : bool
            Whether to show progress during model fitting
        **kwargs: dict
            Model hyperparameters
        '''
        self.mu_p_p = mu_p_p
        self.mu_p_n = mu_p_n
        self.n_components = n_components
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.init_std = init_std
        self.dtype = dtype
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.save_params = save_params
        self.save_dir = save_dir
        self.early_stopping = early_stopping
        self.verbose = verbose

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self, **kwargs):
        ''' Model hyperparameters
        Parameters
        ---------
        lambda_alpha, lambda_beta, lambda_gamma: float
            Regularization parameter for user (lambda_alpha), item factors (
            lambda_beta), and context factors (lambda_gamma).
        c0, c1: float
            Confidence for 0 and 1 in Hu et al., c0 must be less than c1
        '''
        self.lam_alpha = float(kwargs.get('lambda_alpha', 1e-5))
        self.lam_beta = float(kwargs.get('lambda_beta', 1e-5))
        self.lam_gamma_p = float(kwargs.get('lambda_gamma_p', 1e-5))
        self.lam_gamma_n = float(kwargs.get('lambda_gamma_n', 1e+5))
        self.c0 = float(kwargs.get('c0', 0.1))
        self.c1 = float(kwargs.get('c1', 2.0))
        assert self.c0 < self.c1, "c0 must be smaller than c1"

    def _init_params(self, n_users, n_projects):
        ''' Initialize all the latent factors and biases '''
        self.alpha = self.init_std * np.random.randn(n_users, self.n_components).astype(self.dtype)
        self.beta = self.init_std *  np.random.randn(n_projects, self.n_components).astype(self.dtype)
        self.gamma_p = self.init_std * np.random.randn(n_projects, self.n_components).astype(self.dtype)
        # bias for beta and gamma
        self.bias_d_p = np.zeros(n_projects, dtype=self.dtype)
        self.bias_e_p = np.zeros(n_projects, dtype=self.dtype)
        # global bias
        self.global_x_p = 0.0

        self.gamma_n = self.init_std * np.random.randn(n_projects, self.n_components).astype(self.dtype)
        # bias for alpha and theta
        self.bias_d_n = np.zeros(n_projects, dtype=self.dtype)
        self.bias_e_n = np.zeros(n_projects, dtype=self.dtype)
        # global bias
        self.global_x_n = 0.0  # intercept of second factorization for user-user

        # Init got equivalent res
        # print 'Initial alpha: \n', self.alpha
        # print 'Initial beta: \n', self.beta
        # print 'Initial gamma: \n', self.gamma
        #
        # print 'Initial bias_d: \n', self.bias_d
        # print 'Initial bias_e: \n', self.bias_e
        # print 'Initial global_x: \n', self.global_x

    def fit(self, M, X = None, Y = None, FX=None, FY = None, vad_data=None, **kwargs):
        '''Fit the model to the data in M.
        Parameters
        ----------
        M : scipy.sparse.csr_matrix, shape (n_users, n_projects)
            Training click matrix.
        X : scipy.sparse.csr_matrix, shape (n_projects, n_projects)
            Training co-occurrence matrix.
        F : scipy.sparse.csr_matrix, shape (n_projects, n_projects)
            The weight for the co-occurrence matrix. If not provided,
            weight by default is 1.
        vad_data: scipy.sparse.csr_matrix, shape (n_users, n_projects)
            Validation click data.
        **kwargs: dict
            Additional keywords to evaluation function call on validation data
        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        n_users, n_projects = M.shape
        assert X.shape == (n_projects, n_projects)

        self._init_params(n_users, n_projects)
        self._update(M, X, Y, FX, FY, vad_data, **kwargs)
        return self

    def transform(self, M):
        pass

    def _update(self, M, XP, XN, FXP, FXN, vad_data, **kwargs):
        '''Model training and evaluation on validation set'''
        MT = M.T.tocsr()  # pre-compute this
        XPT, XNT, FXPT, FXNT = None, None, None, None
        if XP != None:
            # XT = X.T.tocsr()
            XPT = XP.T
        if XN != None:
            # YT = Y.T.tocsr()
            XNT = XN.T
        if FXP != None:
            # FXT = FX.T
            FXPT = FXP.T
        if FXN != None:
            # FYT = FY.T
            FXNT = FXN.T
        self.vad_ndcg = -np.inf

        #learn item first:
        for i in xrange(self.max_iter):
            if self.verbose:
                print('ITERATION #%d' % i)
            self._update_separate_item_factors(XP, XPT, XN, XNT, FXP, FXPT, FXN, FXNT)
            self._update_biases(XP, XPT, XN, XNT, FXP, FXPT, FXN, FXNT)

        #now we update user's latent representation.
        self._update_separate_user_factors(M)
        if self.save_params:
            self._save_params()


        #print 'alpha:\n', self.alpha
        # print '\n'
        #print 'beta:\n', self.beta
        # print '\n'
        #print 'gamma:\n', self.gamma
        #print 'theta:\n', self.theta

        pass

    def _update_separate_item_factors(self, XP, XPT, XN, XNT, FXP, FXPT, FXN, FXNT):
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating project factors...')
            self.beta = update_beta(None, self.gamma_p, self.gamma_n,
                                self.bias_d_p, self.bias_e_p, self.global_x_p,
                                self.bias_d_n, self.bias_e_n, self.global_x_n,
                                None, XP, FXP, XN, FXN,
                                self.c0, self.c1, self.lam_beta,
                                self.n_jobs,
                                batch_size=self.batch_size,
                                mu_p_p = self.mu_p_p, mu_p_n = self.mu_p_n, f=self.n_components)
        if self.verbose:
            print('\r\tUpdating project factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating positive project embedding factors...')
        # here it really should be X^T and FX^T, but both are symmetric
        self.gamma_p = update_embedding_factor(self.beta,
                                             self.bias_d_p, self.bias_e_p, self.global_x_p,
                                             XPT, FXPT, self.lam_gamma_p,
                                             self.n_jobs,
                                             batch_size=self.batch_size,
                                             mu_p=self.mu_p_p)
        # print('checking gamma_p isnan : %d' % (np.sum(np.isnan(self.gamma_p))))
        if self.verbose:
            print('\r\tUpdating positive project factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating negative project embedding factors...')
        self.gamma_n = update_embedding_factor(self.beta,
                                             self.bias_d_n, self.bias_e_n, self.global_x_n,
                                             XNT, FXNT, self.lam_gamma_n,
                                             self.n_jobs,
                                             batch_size=self.batch_size,
                                             mu_p=self.mu_p_n)
        if self.verbose:
            print('\r\tUpdating negative project embedding factors: time=%.2f'
                  % (time.time() - start_t))

    def _update_biases(self, XP, XPT, XN, XNT, FXP, FXPT, FXN, FXNT):
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating bias terms...')

        self.bias_d_p = update_bias(self.beta, self.gamma_p,
                                  self.bias_e_p, self.global_x_p, XP, FXP,
                                  self.n_jobs, batch_size=self.batch_size,
                                  mu = self.mu_p_p)
        # print('checking isnan bias_d_p : %d' % (np.sum(np.isnan(self.bias_d_p))))
        if np.sum(np.isnan(self.bias_d_p)) > 0 :
            np.argwhere(np.isnan(self.bias_d_p))
        # here it really should be X^T and FX^T, but both are symmetric
        self.bias_e_p = update_bias(self.gamma_p, self.beta,
                                  self.bias_d_p, self.global_x_p, XP, FXP,
                                  self.n_jobs, batch_size=self.batch_size,
                                  mu=self.mu_p_p)
        # print('checking isnan bias_e_p: %d' % (np.sum(np.isnan(self.bias_e_p))))
        if np.sum(np.isnan(self.bias_e_p)) > 0 :
            np.argwhere(np.isnan(self.bias_e_p))
        self.global_x_p = update_global(self.beta, self.gamma_p,
                                  self.bias_d_p, self.bias_e_p, XP, FXP,
                                  self.n_jobs, batch_size=self.batch_size,
                                  mu=self.mu_p_p)
        # print('checking isnan global_x_p: %d' % (np.sum(np.isnan(self.global_x_p))))
        if np.sum(np.isnan(self.global_x_p)) > 0 :
            np.argwhere(np.isnan(self.global_x_p))
        self.bias_d_n = update_bias(self.beta, self.gamma_n,
                                    self.bias_e_n, self.global_x_n, XN, FXN,
                                    self.n_jobs, batch_size=self.batch_size,
                                    mu=self.mu_p_n)
        # print('checking isnan : %d' % (np.sum(np.isnan(self.bias_d_n))))
        # here it really should be X^T and FX^T, but both are symmetric
        self.bias_e_n = update_bias(self.gamma_n, self.beta,
                                    self.bias_d_n, self.global_x_n, XN, FXN,
                                    self.n_jobs, batch_size=self.batch_size,
                                    mu=self.mu_p_n)
        # print('checking isnan : %d' % (np.sum(np.isnan(self.bias_e_n))))
        self.global_x_n = update_global(self.beta, self.gamma_n,
                                        self.bias_d_n, self.bias_e_n, XN, FXN,
                                        self.n_jobs, batch_size=self.batch_size,
                                        mu=self.mu_p_n)
        # print('checking isnan : %d' % (np.sum(np.isnan(self.global_x_n))))
        if self.verbose:
            print('\r\tUpdating bias terms: time=%.2f'
                  % (time.time() - start_t))
        pass


    def _update_separate_user_factors(self, M):
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating user factors...')
        self.alpha = update_alpha(self.beta, M, self.c0, self.c1,
                                  self.lam_alpha,
                                  self.n_jobs,
                                  batch_size=self.batch_size)
        # print('checking user factor isnan : %d'%(np.sum(np.isnan(self.alpha))))
        if self.verbose:
            print('\r\tUpdating user factors: time=%.2f'
                  % (time.time() - start_t))

    # def _update_factors(self, M, MT, XP, XPT, XN, XNT, FXP, FXPT, FXN, FXNT):
    #     if self.verbose:
    #         start_t = _writeline_and_time('\tUpdating user factors...')
    #     self.alpha = update_alpha(self.beta, M, self.c0, self.c1,
    #                               self.lam_alpha,
    #                               self.n_jobs,
    #                               batch_size=self.batch_size)
    #     # print('checking user factor isnan : %d'%(np.sum(np.isnan(self.alpha))))
    #     if self.verbose:
    #         print('\r\tUpdating user factors: time=%.2f'
    #               % (time.time() - start_t))
    #         start_t = _writeline_and_time('\tUpdating project factors...')
    #     self.beta = update_beta(self.alpha, self.gamma_p, self.gamma_n,
    #                             self.bias_d_p, self.bias_e_p, self.global_x_p,
    #                             self.bias_d_n, self.bias_e_n, self.global_x_n,
    #                             MT, XP, FXP, XN, FXN,
    #                             self.c0, self.c1, self.lam_beta,
    #                             self.n_jobs,
    #                             batch_size=self.batch_size,
    #                             mu_p_p = self.mu_p_p, mu_p_n = self.mu_p_n)
    #     # print('checking project factor isnan : %d' % (np.sum(np.isnan(self.beta))))
    #
    #     if self.verbose:
    #         print('\r\tUpdating project factors: time=%.2f'
    #               % (time.time() - start_t))
    #         start_t = _writeline_and_time('\tUpdating positive project embedding factors...')
    #     # here it really should be X^T and FX^T, but both are symmetric
    #     self.gamma_p = update_embedding_factor(self.beta,
    #                                          self.bias_d_p, self.bias_e_p, self.global_x_p,
    #                                          XPT, FXPT, self.lam_gamma_p,
    #                                          self.n_jobs,
    #                                          batch_size=self.batch_size,
    #                                          mu_p=self.mu_p_p)
    #     # print('checking gamma_p isnan : %d' % (np.sum(np.isnan(self.gamma_p))))
    #     if self.verbose:
    #         print('\r\tUpdating positive project factors: time=%.2f'
    #               % (time.time() - start_t))
    #         start_t = _writeline_and_time('\tUpdating negative project embedding factors...')
    #     self.gamma_n = update_embedding_factor(self.beta,
    #                                          self.bias_d_n, self.bias_e_n, self.global_x_n,
    #                                          XNT, FXNT, self.lam_gamma_n,
    #                                          self.n_jobs,
    #                                          batch_size=self.batch_size,
    #                                          mu_p=self.mu_p_n)
    #     # print('checking gamma_n isnan : %d' % (np.sum(np.isnan(self.gamma_n))))
    #
    #     if self.verbose:
    #         print('\r\tUpdating negative project embedding factors: time=%.2f'
    #               % (time.time() - start_t))
    #
    #
    #
    #     pass
    #
    # def _update_biases(self, XP, XPT, XN, XNT, FXP, FXPT, FXN, FXNT):
    #     if self.verbose:
    #         start_t = _writeline_and_time('\tUpdating bias terms...')
    #
    #     self.bias_d_p = update_bias(self.beta, self.gamma_p,
    #                               self.bias_e_p, self.global_x_p, XP, FXP,
    #                               self.n_jobs, batch_size=self.batch_size,
    #                               mu = self.mu_p_p)
    #     # print('checking isnan bias_d_p : %d' % (np.sum(np.isnan(self.bias_d_p))))
    #     if np.sum(np.isnan(self.bias_d_p)) > 0 :
    #         np.argwhere(np.isnan(self.bias_d_p))
    #     # here it really should be X^T and FX^T, but both are symmetric
    #     self.bias_e_p = update_bias(self.gamma_p, self.beta,
    #                               self.bias_d_p, self.global_x_p, XP, FXP,
    #                               self.n_jobs, batch_size=self.batch_size,
    #                               mu=self.mu_p_p)
    #     # print('checking isnan bias_e_p: %d' % (np.sum(np.isnan(self.bias_e_p))))
    #     if np.sum(np.isnan(self.bias_e_p)) > 0 :
    #         np.argwhere(np.isnan(self.bias_e_p))
    #     self.global_x_p = update_global(self.beta, self.gamma_p,
    #                               self.bias_d_p, self.bias_e_p, XP, FXP,
    #                               self.n_jobs, batch_size=self.batch_size,
    #                               mu=self.mu_p_p)
    #     # print('checking isnan global_x_p: %d' % (np.sum(np.isnan(self.global_x_p))))
    #     if np.sum(np.isnan(self.global_x_p)) > 0 :
    #         np.argwhere(np.isnan(self.global_x_p))
    #     self.bias_d_n = update_bias(self.beta, self.gamma_n,
    #                                 self.bias_e_n, self.global_x_n, XN, FXN,
    #                                 self.n_jobs, batch_size=self.batch_size,
    #                                 mu=self.mu_p_n)
    #     # print('checking isnan : %d' % (np.sum(np.isnan(self.bias_d_n))))
    #     # here it really should be X^T and FX^T, but both are symmetric
    #     self.bias_e_n = update_bias(self.gamma_n, self.beta,
    #                                 self.bias_d_n, self.global_x_n, XN, FXN,
    #                                 self.n_jobs, batch_size=self.batch_size,
    #                                 mu=self.mu_p_n)
    #     # print('checking isnan : %d' % (np.sum(np.isnan(self.bias_e_n))))
    #     self.global_x_n = update_global(self.beta, self.gamma_n,
    #                                     self.bias_d_n, self.bias_e_n, XN, FXN,
    #                                     self.n_jobs, batch_size=self.batch_size,
    #                                     mu=self.mu_p_n)
    #     # print('checking isnan : %d' % (np.sum(np.isnan(self.global_x_n))))
    #     if self.verbose:
    #         print('\r\tUpdating bias terms: time=%.2f'
    #               % (time.time() - start_t))
    #     pass


    def _save_params(self):
        '''Save the parameters'''
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        filename = 'CoFacto_K%d_separated.npz' % (self.n_components)
        np.savez(os.path.join(self.save_dir, filename), U=self.alpha,
                 V=self.beta)


# Utility functions #
def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()


def get_row(Y, i):
    '''Given a scipy.sparse.csr_matrix Y, get the values and indices of the
    non-zero values in i_th row'''
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return Y.data[lo:hi], Y.indices[lo:hi]


def update_alpha(beta, M, c0, c1, lam_alpha,
                 n_jobs = 8, batch_size=1000):
    '''Update user latent factors'''
    m, n = M.shape  # m: number of users, n: number of items
    f = beta.shape[1]  # f: number of factors

    BTB = c0 * np.dot(beta.T, beta)  # precompute this
    BTBpR = BTB + lam_alpha * np.eye(f, dtype=beta.dtype)


    return mpps.UpdateUserFactorParallel(
        beta = beta, M = M, BTBpR=BTBpR, c0 = c0, c1 = c1,
        f = f, n_jobs= n_jobs, mode = None
    ).run()

def update_beta(alpha, gamma_p, gamma_n,
                bias_d_p, bias_e_p, global_x_p,
                bias_d_n, bias_e_n, global_x_n,
                MT, XP, FXP, XN, FXN,
                c0, c1, lam_beta,
                n_jobs, batch_size=1000, mu_p_p = 1, mu_p_n = 1, f = 100):
    '''Update item latent factors/embeddings'''


    TTTpR = lam_beta * np.eye(f, dtype=gamma_p.dtype)


    return mpps.UpdateProjectFactorParallel(
        alpha = alpha, gamma_p = gamma_p, gamma_n=gamma_n,
        bias_d_p=bias_d_p, bias_e_p=bias_e_p, global_x_p = global_x_p,
        bias_d_n=bias_d_n, bias_e_n=bias_e_n, global_x_n = global_x_n,
        MT = MT, XP = XP, FXP = FXP, XN = XN, FXN = FXN,
        TTTpR = TTTpR, c0=c0, c1=c1, f=f, mu_p_p=mu_p_p, mu_p_n=mu_p_n,
        n_jobs=n_jobs, mode='hybrid'
    ).run()



def update_embedding_factor(beta, bias_d, bias_e, global_x, XT, FXT, lam_gamma,
                 n_jobs, batch_size=1000, mu_p = 1):
    '''Update context latent factors'''
    n, f = beta.shape  # n: number of items, f: number of factors

    return mpps.UpdateEmbeddingFactorParallel(
        main_factor = beta, bias_main=bias_d, bias_embedding=bias_e,
        intercept=global_x, XT=XT, FXT=FXT, f=f, lam_embedding=lam_gamma,
        mu=mu_p, n_jobs=n_jobs
    ).run()


def update_bias(beta, gamma, bias_e, global_x, X, FX, n_jobs = 8, batch_size=1000,
                        mu = 1):
    ''' Update the per-item (or context) bias term.
    '''
    n = beta.shape[0]


    return mpps.UpdateBiasParallel(
        main_factor=beta, embedding_factor=gamma,
        bias=bias_e, intercept=global_x, X=X, FX=FX, mu=mu,
        n_jobs=n_jobs
    ).run()



def update_global(beta, gamma, bias_d, bias_e, X, FX, n_jobs, batch_size=1000,
                  mu = 1):
    ''' Update the global bias term
    '''
    n = beta.shape[0]
    assert beta.shape == gamma.shape
    assert bias_d.shape == bias_e.shape

    return mpps.UpdateInterceptParallel(
        main_factor = beta, embedding_factor = gamma,
        bias_main = bias_d, bias_embedding = bias_e, X = X, FX = FX, mu=mu,
        n_jobs=n_jobs
    ).run()
