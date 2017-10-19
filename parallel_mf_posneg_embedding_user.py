import os
import sys
import time
import threading
import numpy as np
from numpy import linalg as LA
import ParallelSolver as ps
import MultiProcessParallelSolver as mpps
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin

import rec_eval


class ParallelMFPositiveNegativeUserEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, mu_u_p = 1, mu_u_n = 1, n_components=100, max_iter=10, batch_size=1000,
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
        self.mu_u_p = mu_u_p
        self.mu_u_n = mu_u_n
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
        self.lam_theta_p = float(kwargs.get('lambda_theta_p', 1e-5))
        self.lam_theta_n = float(kwargs.get('lambda_theta_n', 1e-5))
        self.c0 = float(kwargs.get('c0', 0.1))
        self.c1 = float(kwargs.get('c1', 2.0))
        assert self.c0 < self.c1, "c0 must be smaller than c1"

    def _init_params(self, n_users, n_projects):
        ''' Initialize all the latent factors and biases '''
        self.alpha = self.init_std * np.random.randn(n_users, self.n_components).astype(self.dtype)
        self.beta = self.init_std *  np.random.randn(n_projects, self.n_components).astype(self.dtype)
        self.theta_p = self.init_std * np.random.randn(n_users, self.n_components).astype(self.dtype)
        # bias for beta and gamma
        self.bias_b_p = np.zeros(n_users, dtype=self.dtype)
        self.bias_c_p = np.zeros(n_users, dtype=self.dtype)
        # global bias
        self.global_y_p = 0.0

        self.theta_n = self.init_std * np.random.randn(n_users, self.n_components).astype(self.dtype)
        # bias for alpha and theta
        self.bias_b_n = np.zeros(n_users, dtype=self.dtype)
        self.bias_c_n = np.zeros(n_users, dtype=self.dtype)
        # global bias
        self.global_y_n = 0.0  # intercept of second factorization for user-user

        # Init got equivalent res
        # print 'Initial alpha: \n', self.alpha
        # print 'Initial beta: \n', self.beta
        # print 'Initial gamma: \n', self.gamma
        #
        # print 'Initial bias_d: \n', self.bias_d
        # print 'Initial bias_e: \n', self.bias_e
        # print 'Initial global_x: \n', self.global_x

    def fit(self, M, YP = None, YN = None, FYP=None, FYN = None, vad_data=None, **kwargs):
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
        assert YP.shape == (n_users, n_users)
        assert YN.shape == (n_users, n_users)

        self._init_params(n_users, n_projects)
        self._update(M, YP, YN, FYP, FYN, vad_data, **kwargs)
        return self

    def transform(self, M):
        pass

    def _update(self, M, YP, YN, FYP, FYN, vad_data, **kwargs):
        '''Model training and evaluation on validation set'''
        MT = M.T.tocsr()  # pre-compute this
        YPT, YNT, FYPT, FYNT = None, None, None, None
        if YP != None:
            YPT = YP.T
        if YN != None:
            YNT = YN.T
        if FYP != None:
            FYPT = FYP.T
        if FYN != None:
            FYNT = FYN.T
        self.vad_ndcg = -np.inf
        for i in xrange(self.max_iter):
            if self.verbose:
                print('ITERATION #%d' % i)
            self._update_factors(M, MT, YP, YPT, YN, YNT, FYP, FYPT, FYN, FYNT)
            self._update_biases(YP, YPT, YN, YNT, FYP, FYPT, FYN, FYNT)
            if vad_data is not None:
                vad_ndcg = self._validate(M, vad_data, **kwargs)
                if self.early_stopping and self.vad_ndcg > vad_ndcg:
                    break  # we will not save the parameter for this iteration
                self.vad_ndcg = vad_ndcg
            if self.save_params:
                self._save_params(i)
        #print 'alpha:\n', self.alpha
        # print '\n'
        #print 'beta:\n', self.beta
        # print '\n'
        #print 'gamma:\n', self.gamma
        #print 'theta:\n', self.theta

        pass

    def _update_factors(self, M, MT, YP, YPT, YN, YNT, FYP, FYPT, FYN, FYNT):
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating user factors...')
        self.alpha = update_alpha(self.beta, self.theta_p, self.theta_n,
                                  self.bias_b_p, self.bias_c_p, self.global_y_p,
                                  self.bias_b_n, self.bias_c_n, self.global_y_n,
                                  M, YP, FYP, YN, FYN, self.c0, self.c1, self.lam_alpha,
                                  n_jobs=self.n_jobs, batch_size=self.batch_size,
                                  mu_u_p=self.mu_u_p, mu_u_n=self.mu_u_n)
        # print('checking user factor isnan : %d'%(np.sum(np.isnan(self.alpha))))
        if self.verbose:
            print('\r\tUpdating user factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating project factors...')
        self.beta = update_beta(self.alpha, MT,
                                self.c0, self.c1, self.lam_beta,
                                self.n_jobs, batch_size=self.batch_size)
        # print('checking project factor isnan : %d' % (np.sum(np.isnan(self.beta))))

        if self.verbose:
            print('\r\tUpdating user embedding factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating positive user embedding factors...')
        # here it really should be X^T and FX^T, but both are symmetric
        self.theta_p = update_embedding_factor(self.alpha,
                                             self.bias_b_p, self.bias_c_p, self.global_y_p,
                                             YPT, FYPT, self.lam_theta_p,
                                             self.n_jobs,
                                             batch_size=self.batch_size,
                                             mu_p=self.mu_u_p)
        # print('checking theta_p isnan : %d' % (np.sum(np.isnan(self.theta_p))))
        if self.verbose:
            print('\r\tUpdating positive user factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating negative user embedding factors...')
        self.theta_n = update_embedding_factor(self.alpha,
                                             self.bias_b_n, self.bias_c_n, self.global_y_n,
                                             YNT, FYNT, self.lam_theta_n,
                                             self.n_jobs,
                                             batch_size=self.batch_size,
                                             mu_p=self.mu_u_n)
        # print('checking theta_n isnan : %d' % (np.sum(np.isnan(self.theta_n))))

        if self.verbose:
            print('\r\tUpdating negative user embedding factors: time=%.2f'
                  % (time.time() - start_t))



        pass

    def _update_biases(self, YP, YPT, YN, YNT, FYP, FYPT, FYN, FYNT):
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating bias terms...')

        self.bias_b_p = update_bias(self.alpha, self.theta_p,
                                  self.bias_c_p, self.global_y_p, YP, FYP,
                                  self.n_jobs, batch_size=self.batch_size,
                                  mu = self.mu_u_p)
        # print('checking isnan bias_b_p : %d' % (np.sum(np.isnan(self.bias_b_p))))
        if np.sum(np.isnan(self.bias_b_p)) > 0 :
            np.argwhere(np.isnan(self.bias_b_p))
        # here it really should be X^T and FX^T, but both are symmetric
        self.bias_c_p = update_bias(self.theta_p, self.alpha,
                                  self.bias_b_p, self.global_y_p, YP, FYP,
                                  self.n_jobs, batch_size=self.batch_size,
                                  mu=self.mu_u_p)
        # print('checking isnan bias_c_p: %d' % (np.sum(np.isnan(self.bias_c_p))))
        if np.sum(np.isnan(self.bias_c_p)) > 0 :
            np.argwhere(np.isnan(self.bias_c_p))
        self.global_y_p = update_global(self.alpha, self.theta_p,
                                  self.bias_b_p, self.bias_c_p, YP, FYP,
                                  self.n_jobs, batch_size=self.batch_size,
                                  mu=self.mu_u_p)
        # print('checking isnan global_y_p: %d' % (np.sum(np.isnan(self.global_y_p))))
        if np.sum(np.isnan(self.global_y_p)) > 0 :
            np.argwhere(np.isnan(self.global_y_p))
        self.bias_b_n = update_bias(self.alpha, self.theta_n,
                                    self.bias_c_n, self.global_y_n, YN, FYN,
                                    self.n_jobs, batch_size=self.batch_size,
                                    mu=self.mu_u_n)
        # print('checking isnan : %d' % (np.sum(np.isnan(self.bias_b_n))))
        # here it really should be X^T and FX^T, but both are symmetric
        self.bias_c_n = update_bias(self.theta_n, self.alpha,
                                    self.bias_b_n, self.global_y_n, YN, FYN,
                                    self.n_jobs, batch_size=self.batch_size,
                                    mu=self.mu_u_n)
        # print('checking isnan : %d' % (np.sum(np.isnan(self.bias_c_n))))
        self.global_y_n = update_global(self.alpha, self.theta_n,
                                        self.bias_b_n, self.bias_c_n, YN, FYN,
                                        self.n_jobs, batch_size=self.batch_size,
                                        mu=self.mu_u_n)
        # print('checking isnan : %d' % (np.sum(np.isnan(self.global_y_n))))
        if self.verbose:
            print('\r\tUpdating bias terms: time=%.2f'
                  % (time.time() - start_t))
        pass

    def _validate(self, M, vad_data, **kwargs):
        vad_ndcg = rec_eval.parallel_normalized_dcg_at_k(M, vad_data,
                                                self.alpha,
                                                self.beta,
                                                **kwargs)
        if self.verbose:
            print('\tValidation NDCG@k: %.5f' % vad_ndcg)
        return vad_ndcg

    def _save_params(self, iter):
        '''Save the parameters'''
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        filename = 'CoFacto_K%d_iter%d.npz' % (self.n_components, iter)
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

def update_alpha(beta, theta_p, theta_n,
                 bias_b_p, bias_c_p, global_y_p,
                 bias_b_n, bias_c_n, global_y_n,
                 M, YP, FYP, YN, FYN, c0, c1, lam_alpha,
                 n_jobs = 8, batch_size=1000, mu_u_p=1, mu_u_n=1):
    '''Update user latent factors'''
    m, n = M.shape  # m: number of users, n: number of items
    f = beta.shape[1]  # f: number of factors

    BTB = c0 * np.dot(beta.T, beta)  # precompute this
    BTBpR = BTB + lam_alpha * np.eye(f, dtype=beta.dtype)

    return mpps.UpdateUserFactorParallel(
        beta, theta_p=theta_p, theta_n=theta_n,
        bias_b_p=bias_b_p, bias_c_p=bias_c_p, global_y_p=global_y_p,
        bias_b_n=bias_b_n, bias_c_n=bias_c_n, global_y_n=global_y_n,
        M=M, YP=YP, FYP=FYP, YN=YN, FYN=FYN, BTBpR=BTBpR,
        c0=c0, c1=c1, f=f, mu_u_p=mu_u_p, mu_u_n=mu_u_n,
        n_jobs=n_jobs, mode='hybrid'
    ).run()

def update_beta(alpha, MT, c0, c1, lam_beta,
                n_jobs, batch_size=1000):
    '''Update item latent factors/embeddings'''
    n, m = MT.shape  # m: number of users, n: number of projects
    f = alpha.shape[1]
    assert alpha.shape[0] == m

    TTT = c0 * np.dot(alpha.T, alpha)  # precompute this
    TTTpR = TTT + lam_beta * np.eye(f, dtype=alpha.dtype)

    return mpps.UpdateProjectFactorParallel(
        alpha = alpha, MT = MT, TTTpR = TTTpR, c0=c0, c1=c1,
        f=f, n_jobs=n_jobs, mode=None
    ).run()

def update_embedding_factor(alpha, bias_b, bias_c, global_y, YT, FYT, lam_theta,
                 n_jobs, batch_size=1000, mu_p = 1):
    '''Update context latent factors'''
    m, f = alpha.shape  # m: number of users, f: number of factors

    return mpps.UpdateEmbeddingFactorParallel(
        main_factor = alpha, bias_main=bias_b, bias_embedding=bias_c,
        intercept=global_y, XT=YT, FXT=FYT, f=f, lam_embedding=lam_theta,
        mu=mu_p, n_jobs=n_jobs
    ).run()

def update_bias(alpha, theta, bias_c, global_y, Y, FY,
                n_jobs = 8, batch_size=1000, mu = 1):
    return mpps.UpdateBiasParallel(
        main_factor=alpha, embedding_factor=theta,
        bias=bias_c, intercept=global_y, X=Y, FX=FY, mu=mu,
        n_jobs=n_jobs
    ).run()


def update_global(alpha, theta, bias_b, bias_c, Y, FY,
                  n_jobs = 8, batch_size=1000, mu = 1):

    assert alpha.shape == theta.shape
    assert bias_b.shape == bias_c.shape
    return mpps.UpdateInterceptParallel(
        main_factor = alpha, embedding_factor = theta,
        bias_main = bias_b, bias_embedding = bias_c, X = Y, FX = FY, mu=mu,
        n_jobs=n_jobs
    ).run()

