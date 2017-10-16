import os
import sys
import time

import numpy as np
from numpy import linalg as LA

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin

import rec_eval


class MFPositiveUserProjectEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, mu_u = 1, mu_p = 1, n_components=100, max_iter=10, batch_size=1000,
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
        self.mu_u = mu_u
        self.mu_p = mu_p
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
        self.lam_theta = float(kwargs.get('lambda_theta', 1e-5))
        self.lam_beta = float(kwargs.get('lambda_beta', 1e-5))
        self.lam_gamma = float(kwargs.get('lambda_gamma', 1e+0))
        self.c0 = float(kwargs.get('c0', 0.01))
        self.c1 = float(kwargs.get('c1', 1.0))
        assert self.c0 < self.c1, "c0 must be smaller than c1"

    def _init_params(self, n_users, n_projects):
        ''' Initialize all the latent factors and biases '''
        self.alpha = self.init_std * np.random.randn(n_users, self.n_components).astype(self.dtype)
        self.beta = self.init_std *  np.random.randn(n_projects, self.n_components).astype(self.dtype)
        self.gamma = self.init_std * np.random.randn(n_projects, self.n_components).astype(self.dtype)
        # bias for beta and gamma
        self.bias_d = np.zeros(n_projects, dtype=self.dtype)
        self.bias_e = np.zeros(n_projects, dtype=self.dtype)
        # global bias
        self.global_x = 0.0

        self.theta = self.init_std * np.random.randn(n_users, self.n_components).astype(self.dtype)
        # bias for alpha and theta
        self.bias_b = np.zeros(n_users, dtype=self.dtype)
        self.bias_c = np.zeros(n_users, dtype=self.dtype)
        # global bias
        self.global_y = 0.0  # intercept of second factorization for user-user

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

    def _update(self, M, X, Y, FX, FY, vad_data, **kwargs):
        '''Model training and evaluation on validation set'''
        MT = M.T.tocsr()  # pre-compute this
        XT, YT, FXT, FYT = None, None, None, None
        if X != None:
            # XT = X.T.tocsr()
            XT = X.T
        if Y != None:
            # YT = Y.T.tocsr()
            YT = Y.T
        if FX != None:
            # FXT = FX.T
            FXT = FX.T
        if FY != None:
            # FYT = FY.T
            FYT = FY.T
        self.vad_ndcg = -np.inf
        for i in xrange(self.max_iter):
            if self.verbose:
                print('ITERATION #%d' % i)
            self._update_factors(M, MT, X, XT, Y,YT, FX, FXT, FY, FYT)
            self._update_biases(X, XT, Y, YT, FX, FXT, FY, FYT)
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

    def _update_factors(self, M, MT, X, XT, Y,YT, FX, FXT, FY, FYT):
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating user factors...')
        self.alpha = update_alpha(self.beta, self.theta,
                                  self.bias_b, self.bias_c, self.global_y,
                                  M, Y, FY, self.c0, self.c1, self.lam_alpha,
                                  self.n_jobs,
                                  batch_size=self.batch_size,
                                  mu_u=self.mu_u)
        if self.verbose:
            print('\r\tUpdating user factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating project factors...')
        self.beta = update_beta(self.alpha, self.gamma,
                                self.bias_d, self.bias_e, self.global_x,
                                MT, X, FX, self.c0, self.c1, self.lam_beta,
                                self.n_jobs,
                                batch_size=self.batch_size,
                                mu_p = self.mu_p)

        if self.verbose:
            print('\r\tUpdating user embedding factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating project embedding factors...')
        # here it really should be X^T and FX^T, but both are symmetric
        self.gamma = update_embedding_factor(self.beta, self.bias_d, self.bias_e, self.global_x,
                                  XT, FXT, self.lam_gamma,
                                  self.n_jobs,
                                  batch_size=self.batch_size,
                                  mu_p=self.mu_p)

        if self.verbose:
            print('\r\tUpdating project factors: time=%.2f'
                  % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating user embedding factors...')
        self.theta = update_embedding_factor(self.alpha, self.bias_b, self.bias_c, self.global_y,
                                  YT, FYT, self.lam_theta,
                                  self.n_jobs,
                                  batch_size=self.batch_size,
                                  mu_p=self.mu_u)

        if self.verbose:
            print('\r\tUpdating project embedding factors: time=%.2f'
                  % (time.time() - start_t))



        pass

    def _update_biases(self, X, XT, Y, YT, FX, FXT, FY, FYT):
        if self.verbose:
            start_t = _writeline_and_time('\tUpdating bias terms...')

        self.bias_d = update_bias(self.beta, self.gamma,
                                  self.bias_e, self.global_x, X, FX,
                                  self.n_jobs, batch_size=self.batch_size,
                                  mu = self.mu_p)
        # here it really should be X^T and FX^T, but both are symmetric
        self.bias_e = update_bias(self.gamma, self.beta,
                                  self.bias_d, self.global_x, X, FX,
                                  self.n_jobs, batch_size=self.batch_size,
                                  mu=self.mu_p)
        self.global_x = update_global(self.beta, self.gamma,
                                  self.bias_d, self.bias_e, X, FX,
                                  self.n_jobs, batch_size=self.batch_size,
                                  mu=self.mu_p)

        self.bias_b = update_bias(self.alpha, self.theta,
                                  self.bias_c, self.global_y, Y, FY,
                                  self.n_jobs, batch_size=self.batch_size,
                                  mu=self.mu_u)
        self.bias_c = update_bias(self.theta, self.alpha,
                                  self.bias_b, self.global_y, Y, FY,
                                  self.n_jobs, batch_size=self.batch_size,
                                  mu=self.mu_u)
        self.global_y = update_global(self.alpha, self.theta,
                                      self.bias_b, self.bias_c, Y, FY,
                                      self.n_jobs, batch_size=self.batch_size,
                                      mu=self.mu_u)

        if self.verbose:
            print('\r\tUpdating bias terms: time=%.2f'
                  % (time.time() - start_t))
        pass

    def _validate(self, M, vad_data, **kwargs):
        vad_ndcg = rec_eval.normalized_dcg_at_k(M, vad_data,
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


def update_alpha(beta, theta,
                 bias_b, bias_c, global_y,
                 M, Y, FY, c0, c1, lam_alpha,
                 n_jobs = 8, batch_size=1000, mu_u = 1):
    '''Update user latent factors'''
    m, n = M.shape  # m: number of users, n: number of items
    f = beta.shape[1]  # f: number of factors

    BTB = c0 * np.dot(beta.T, beta)  # precompute this
    BTBpR = BTB + lam_alpha * np.eye(f, dtype=beta.dtype)

    start_idx = range(0, m, batch_size)
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_solve_weighted_user_factor)(
            lo, hi, beta, theta, bias_b, bias_c, global_y,
            M, Y, FY, BTBpR, c0, c1, f, mu_u)
        for lo, hi in zip(start_idx, end_idx))
    alpha = np.vstack(res)
    return alpha


def _solve_weighted_user_factor(lo, hi, beta, theta, bias_b, bias_c, global_y,
                                M, Y, FY, BTBpR, c0, c1, f, mu_u):
    alpha_batch = np.empty((hi - lo, f), dtype=beta.dtype)
    for ui, u in enumerate(xrange(lo, hi)):
        m_u, idx_m_p = get_row(M, u)
        B_p = beta[idx_m_p]

        y_u, idx_y_u = get_row(Y, u)
        T_j = theta[idx_y_u]

        rsd = y_u - bias_b[u] - bias_c[idx_y_u] - global_y

        if FY is not None: #FY is weighted matrix of Y
            f_u, _ = get_row(FY, u)
            TTT = T_j.T.dot(T_j * f_u[:, np.newaxis])
            rsd *= f_u
        else:
            TTT = T_j.T.dot(T_j)
        TTT = mu_u*TTT
        # a = m_u.dot(c1 * B_p) + np.dot(rsd, T_j)

        a = m_u.dot(c1 * B_p) + mu_u * np.dot(rsd, T_j)
        A = BTBpR + B_p.T.dot((c1 - c0) * B_p) + TTT
        alpha_batch[ui] = LA.solve(A, a)
    return alpha_batch


def update_beta(alpha, gamma, bias_d, bias_e, global_x, MT, X, FX, c0, c1,
                lam_beta, n_jobs, batch_size=1000, mu_p = 1):
    '''Update item latent factors/embeddings'''
    n, m = MT.shape  # m: number of users, n: number of items
    f = alpha.shape[1]
    assert alpha.shape[0] == m
    assert gamma.shape == (n, f)

    TTT = c0 * np.dot(alpha.T, alpha)  # precompute this
    TTTpR = TTT + lam_beta * np.eye(f, dtype=alpha.dtype)
    # print 'PRE: \n', TTTpR

    start_idx = range(0, n, batch_size)
    end_idx = start_idx[1:] + [n]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_solve_weighted_project_cofactor)(
            lo, hi, alpha, gamma, bias_d, bias_e, global_x,
            MT, X, FX, TTTpR, c0, c1, f, mu_p)
        for lo, hi in zip(start_idx, end_idx))
    beta = np.vstack(res)
    # print 'Final beta: \n', beta
    return beta


def _solve_weighted_project_cofactor(lo, hi, alpha, gamma, bias_d, bias_e, global_x, MT,
                             X, FX, TTTpR, c0, c1, f, mu_p):
    beta_batch = np.empty((hi - lo, f), dtype=alpha.dtype)
    for pi, p in enumerate(xrange(lo, hi)):
        m_u, idx_u = get_row(MT, p)
        A_u = alpha[idx_u]

        x_pj, idx_x_j = get_row(X, p)
        G_i = gamma[idx_x_j]

        rsd = x_pj - bias_d[p] - bias_e[idx_x_j] - global_x

        if FX is not None:
            f_i, _ = get_row(FX, p)
            GTG = G_i.T.dot(G_i * f_i[:, np.newaxis])
            rsd *= f_i
        else:
            GTG = G_i.T.dot(G_i)
        GTG = mu_p * GTG


        B = TTTpR + A_u.T.dot((c1 - c0) * A_u) + GTG
        a = m_u.dot(c1 * A_u) + mu_p*np.dot(rsd, G_i)

        beta_batch[pi] = LA.solve(B, a)
        # if lo == 0:
            # print 'a is \n', a
            # print 'B is \n', B
            # print 'RES*****:,', B * beta_batch[ib] - a
    return beta_batch


def update_embedding_factor(beta, bias_d, bias_e, global_x, XT, FXT, lam_gamma,
                 n_jobs, batch_size=1000, mu_p = 1):
    '''Update context latent factors'''
    n, f = beta.shape  # n: number of items, f: number of factors

    start_idx = range(0, n, batch_size)
    end_idx = start_idx[1:] + [n]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_solve_embedding_factor)(
            lo, hi, beta, bias_d, bias_e, global_x, XT, FXT, f, lam_gamma, mu_p)
        for lo, hi in zip(start_idx, end_idx))
    gamma = np.vstack(res)
    return gamma


def _solve_embedding_factor(lo, hi, beta, bias_d, bias_e, global_x, XT, FXT, f, lam_gamma,
                  mu_p):
    gamma_batch = np.empty((hi - lo, f), dtype=beta.dtype)
    for jb, j in enumerate(xrange(lo, hi)):
        x_jp, idx_p = get_row(XT, j)
        rsd = x_jp - bias_d[idx_p] - bias_e[j] - global_x
        B_j = beta[idx_p]
        if FXT is not None:
            f_j, _ = get_row(FXT, j)
            BTB = B_j.T.dot(B_j * f_j[:, np.newaxis])
            rsd *= f_j
        else:
            BTB = B_j.T.dot(B_j)

        B = BTB + lam_gamma * np.eye(f, dtype=beta.dtype)
        a = mu_p*np.dot(rsd, B_j)
        gamma_batch[jb] = LA.solve(B, a)
    return gamma_batch


def update_bias(beta, gamma, bias_e, global_x, X, FX, n_jobs = 8, batch_size=1000,
                        mu = 1):
    ''' Update the per-item (or context) bias term.
    '''
    n = beta.shape[0]

    start_idx = range(0, n, batch_size)
    end_idx = start_idx[1:] + [n]

    res = Parallel(n_jobs=n_jobs)(
        delayed(_solve_bias)(lo, hi, beta, gamma, bias_e, global_x, X, FX, mu)
        for lo, hi in zip(start_idx, end_idx))
    bias_d = np.hstack(res)
    return bias_d


def _solve_bias(lo, hi, beta, gamma, bias_e, global_x, X, FX, mu):
    bias_d_batch = np.empty(hi - lo, dtype=beta.dtype)
    if mu != 0:
        for ib, i in enumerate(xrange(lo, hi)):
            m_i, idx_i = get_row(X, i)
            m_i_hat = gamma[idx_i].dot(beta[i]) + bias_e[idx_i] + global_x
            rsd = m_i - m_i_hat

            if FX is not None:
                f_i, _ = get_row(FX, i)
                rsd *= f_i

            if rsd.size > 0:
                bias_d_batch[ib] = mu*rsd.mean()
            else:
                bias_d_batch[ib] = 0.
    return bias_d_batch


def update_global(beta, gamma, bias_d, bias_e, X, FX, n_jobs, batch_size=1000,
                  mu = 1):
    ''' Update the global bias term
    '''
    n = beta.shape[0]
    assert beta.shape == gamma.shape
    assert bias_d.shape == bias_e.shape

    start_idx = range(0, n, batch_size)
    end_idx = start_idx[1:] + [n]

    res = Parallel(n_jobs=n_jobs)(
        delayed(_solve_global)(lo, hi, beta, gamma, bias_d, bias_e, X, FX, mu)
        for lo, hi in zip(start_idx, end_idx))

    return np.sum(res) / X.data.size


def _solve_global(lo, hi, beta, gamma, bias_d, bias_e, X, FX, mu):
    res = 0.
    if mu != 0:
        for ib, i in enumerate(xrange(lo, hi)):
            m_i, idx_i = get_row(X, i)
            m_i_hat = gamma[idx_i].dot(beta[i]) + bias_d[i] + bias_e[idx_i]
            rsd = m_i - m_i_hat

            if FX is not None:
                f_i, _ = get_row(FX, i)
                rsd *= f_i
            res += rsd.sum()
    return mu*res
