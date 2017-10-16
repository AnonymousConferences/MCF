
import os
import sys
import time

import numpy as np
from numpy import linalg as LA

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin

import rec_eval

class MFEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, mu_u = 1, mu_p = 1, n_components=100, max_iter=10, batch_size=1000,
                 init_std=0.01, dtype='float32', n_jobs=8, random_state=None,
                 save_params=False, save_dir='data/rec_data/all/params-save', early_stopping=False,
                 verbose=False, **kwargs):
        '''
        MFEmbedding
        Parameters
        ---------
        mu_u : a hyper parameter to control the affect of user-user co-occurrence
        mu_p : a hyper parameter to control the affect of project-project co-occurrence
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
               lambda_theta, lambda_beta, lambda_gamma: float
                   Regularization parameter for user (lambda_theta), item factors (
                   lambda_beta), and context factors (lambda_gamma).
               c0, c1: float
                   Confidence for 0 and 1 in Hu et al., c0 must be less than c1
               '''
        self.lam_alpha = float(kwargs.get('lambda_alpha', 1e-5))
        self.lam_theta = float(kwargs.get('lambda_theta', 1e-5))
        self.lam_beta  = float(kwargs.get('lambda_beta', 1e-5))
        self.lam_gamma = float(kwargs.get('lambda_gamma', 1e-5))

        self.c0 = float(kwargs.get('c0', 0.001))
        self.c1 = float(kwargs.get('c1', 2.0))
        assert self.c0 < self.c1, "c0 must be smaller than c1"

    def _init_params(self, n_users, n_projects):
        ''' Initialize all the latent factors and biases '''
        #latent matrix
        self.alpha = self.init_std * np.random.randn(n_users, self.n_components).astype(self.dtype)
        self.theta = self.init_std * np.random.randn(n_users, self.n_components).astype(self.dtype)
        self.beta = self.init_std  * np.random.randn(n_projects, self.n_components).astype(self.dtype)
        self.gamma = self.init_std * np.random.randn(n_projects, self.n_components).astype(self.dtype)

        # bias for beta and gamma
        self.bias_d = np.zeros(n_projects, dtype=self.dtype)
        self.bias_e = np.zeros(n_projects, dtype=self.dtype)
        # global bias
        self.global_x = 0.0  # intercept of second factorization for project-project


        # bias for alpha and theta
        self.bias_b = np.zeros(n_users, dtype=self.dtype)
        self.bias_c = np.zeros(n_users, dtype=self.dtype)
        # global bias
        self.global_y = 0.0 #intercept of second factorization for user-user




    def fit(self, M, X, Y, FX=None, FY=None, vad_data=None, **kwargs):
        '''Fit the model to the data in X.
        Parameters
        ----------
        M : scipy.sparse.csr_matrix, shape (n_users, n_items)
            backing training data
        X : scipy.sparse.csr_matrix, shape (n_projects, n_projects)
            Training co-occurrence matrix of projects
        Y : scipy.sparse.csr_matrix, shape (n_users, n_users)
            Training co-occurrence matrix of users
        FX : scipy.sparse.csr_matrix, shape (n_projects, n_projects)
            The weight for matrix X. If not provided, weight by default is 1.
        FY : scipy.sparse.csr_matrix, shape (n_users, n_users)
        The weight for the matrix Y. If not provided, weight by default is 1.
        vad_data: scipy.sparse.csr_matrix, shape (n_users, n_items)
            Validation backing data.
        **kwargs: dict
            Additional keywords to evaluation function call on validation data
        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        n_users, n_projects = M.shape
        assert X.shape == (n_projects, n_projects)
        assert Y.shape == (n_users, n_users)

        self._init_params(n_users, n_projects)
        self._update(M, X, Y,  FX, FY, vad_data, **kwargs)
        return self

    def _update(self, M, X, Y, FX, FY, vad_data, **kwargs):
        '''Model training and evaluation on validation set'''
        MT = M.T.tocsr()  # pre-compute this
        XT,YT, FXT, FYT = None, None, None, None
        if X != None:
            XT = X.T.tocsr()
        if Y != None:
            YT = Y.T.tocsr()
        if FX != None:
            FXT = FX.T
        if FY != None:
            FYT = FY.T
        self.vad_ndcg = -np.inf
        for i in xrange(self.max_iter):
            if self.verbose:
                print('ITERATION #%d' % i)

            self._update_factors(M, MT, X, XT, Y, YT, FX, FXT, FY, FYT)
            self._update_biases(X, XT, Y, YT, FX, FXT, FY, FYT)



            if vad_data is not None:
                vad_ndcg = self._validate(M, vad_data, **kwargs)
                if self.early_stopping and self.vad_ndcg > vad_ndcg:
                    break  # we will not save the parameter for this iteration
                self.vad_ndcg = vad_ndcg
            if self.save_params:
                self._save_params(i)
        pass

    def _update_factors(self, M, MT, X, XT, Y, YT, FX, FXT, FY, FYT):


        # print 'Optimizing gamma'
        # self.gamma = update_gamma(self.beta, self.bias_d, self.bias_e, self.global_x,
        #                           XT, FXT, self.lam_gamma,
        #                           self.n_jobs, batch_size=self.batch_size)

        if self.verbose:
            start_t = _writeline_and_time('\tUpdating user factors...')
        self.alpha = update_alpha(self.beta, self.theta,
                                  self.bias_b, self.bias_c, self.global_y,
                                  M, Y, FY, self.c0, self.c1, self.lam_alpha,
                                  self.n_jobs,
                                  batch_size=self.batch_size,
                                  mu_u = self.mu_u)
        if self.verbose:
            print('\r\tUpdating user factors: time=%.2f' % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating project factors...')
        self.beta = update_beta(self.alpha, self.gamma,
                                self.bias_d, self.bias_e, self.global_x,
                                MT, X, FX, self.c0, self.c1, self.lam_beta,
                                self.n_jobs,
                                batch_size=self.batch_size,
                                mu_p = self.mu_p)
        if self.verbose:
            print('\r\tUpdating item factors: time=%.2f' % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating user context factors...')

        self.theta = update_theta(self.alpha, self.bias_b, self.bias_c, self.global_y,
                                  YT, FYT, self.lam_theta,
                                  self.n_jobs, batch_size = self.batch_size,
                                  mu_u=self.mu_u)
        if self.verbose:
            print('\r\tUpdating user context factors: time=%.2f' % (time.time() - start_t))
            start_t = _writeline_and_time('\tUpdating project context factors...')
        # here it really should be M^T and F^T, but both are symmetric
        self.gamma = update_gamma(self.beta, self.bias_d, self.bias_e, self.global_x,
                                  XT, FXT, self.lam_gamma,
                                  self.n_jobs, batch_size=self.batch_size,
                                  mu_p=self.mu_p)
        if self.verbose:
            print('\r\tUpdating project context factors: time=%.2f' % (time.time() - start_t))

        # print 'alpha:\n',self.alpha
        # print '\n'
        # print 'beta:\n', self.beta
        # print '\n'
        # print 'gamma:\n', self.gamma
        # print '\n'
        # print 'theta:\n', self.theta
        pass

    def _update_biases(self, X, XT, Y, YT, FX, FXT, FY, FYT):

        if self.verbose:
            start_t = _writeline_and_time('\tUpdating bias terms...')

        self.bias_b = update_user_biases(self.alpha, self.theta,
                                  self.bias_c, self.global_y, Y, FY,
                                  self.n_jobs, batch_size=self.batch_size, mu_u = self.mu_u)
        self.bias_c = update_user_biases(self.theta, self.alpha,
                                  self.bias_b, self.global_y, YT, FYT,
                                  self.n_jobs, batch_size=self.batch_size, mu_u = self.mu_u)
        self.global_y = update_global(self.alpha, self.theta,
                                      self.bias_b, self.bias_c, Y, FY,
                                      self.n_jobs, batch_size=self.batch_size, mu = self.mu_u)

        self.bias_d = update_project_bias(self.beta, self.gamma,
                                  self.bias_e, self.global_x, X, FX,
                                  self.n_jobs, batch_size=self.batch_size, mu_p = self.mu_p)
        self.bias_e = update_project_bias(self.gamma, self.beta,
                                  self.bias_d, self.global_x, XT, FXT,
                                  self.n_jobs, batch_size=self.batch_size, mu_p = self.mu_p)
        self.global_x = update_global(self.beta, self.gamma,
                                  self.bias_d, self.bias_e, X, FX,
                                  self.n_jobs, batch_size=self.batch_size, mu = self.mu_p)
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
        np.savez(os.path.join(self.save_dir, filename), U=self.theta,
                 V=self.beta)

def _writeline_and_time(s):
    sys.stdout.write(s)
    sys.stdout.flush()
    return time.time()

def get_row(Y, i):
    '''Given a scipy.sparse.csr_matrix Y, get the values and indices of the
    non-zero values in i_th row'''
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return Y.data[lo:hi], Y.indices[lo:hi]

def update_alpha(beta, theta, bias_b, bias_c, global_y, M, Y, FY, c0, c1, lam_alpha, n_jobs, batch_size=1000, mu_u = 1):
    '''Update user latent factors'''
    m, n = M.shape  # m: number of users, n: number of projects
    k = beta.shape[1]  # f: number of factors
    assert beta.shape[0] == n
    assert theta.shape == (m, k)
    assert Y.shape == (m, m) #user-user sppmi matrix
    PRE = c0 * np.dot(beta.T, beta)  # precompute this
    PRE = PRE + lam_alpha * np.eye(k, dtype=beta.dtype)
    start_idx = range(0, m, batch_size)
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_solve_weighted_user_factor)(
            lo, hi, beta, theta, bias_b, bias_c, global_y, M, Y, FY, PRE, c0, c1, k, mu_u)
        for lo, hi in zip(start_idx, end_idx))
    alpha = np.vstack(res)
    return alpha


def _solve_weighted_user_factor(lo, hi, beta, theta, bias_b, bias_c, global_y, M, Y, FY, PRE, c0, c1, k, mu_u):
    alpha_batch = np.empty((hi - lo, k), dtype=beta.dtype)
    for ui, u in enumerate(xrange(lo, hi)): #ub come from 0 to (hi-lo) while u come from [lo, hi)
        m_p, idx_m_p = get_row(M, u) # get the columns and data that not equal to 0,
                                   # m_p is data, idx_p is the column indices of projects where user u backed
        B_p = beta[idx_m_p]

        y_u, idx_y_u = get_row(Y, u)
        T_j = theta[idx_y_u]

        rsd = y_u - bias_b[u] - bias_c[idx_y_u] - global_y

        rsd = mu_u*rsd

        if FY is not None: #FY is weighted matrix of Y
            f_u, _ = get_row(FY, u)
            TTT = T_j.T.dot(T_j * f_u[:, np.newaxis])
            rsd *= f_u
        else:
            TTT = T_j.T.dot(T_j)

        TTT = mu_u*TTT

        A = PRE + \
            B_p.T.dot((c1 - c0)*B_p) + \
            TTT
            ##(c1 - c0)*T_i.T.dot.T_i: put more weight on positive samples.

        a = np.dot(rsd, T_j) + m_p.dot(c1*B_p)

        alpha_batch[ui] = LA.solve(A, a)
    return alpha_batch

def update_beta(alpha, gamma, bias_d, bias_e, global_x,
                MT, X, FX, c0, c1, lam_beta,
                n_jobs, batch_size=1000,
                mu_p = 1):
    '''Update item latent factors/embeddings'''
    n, m = MT.shape  # m: number of users, n: number of items
    k = alpha.shape[1]
    assert alpha.shape == (m, k)
    assert gamma.shape == (n, k)
    assert bias_d.shape == (n,)
    assert bias_e.shape == (n,)
    assert X.shape == (n, n)

    PRE = c0 * np.dot(alpha.T, alpha) + lam_beta*np.eye(k, dtype=alpha.dtype)
    # print 'PRE: \n', PRE
    start_idx = range(0, n, batch_size)
    end_idx = start_idx[1:] + [n]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_solve_weighted_project_factor)(
            lo, hi, alpha, gamma, bias_d, bias_e, global_x,
            MT, X, FX, PRE, c0, c1, k, mu_p)
        for lo, hi in zip(start_idx, end_idx))
    beta = np.vstack(res)
    # print 'Final beta: \n', beta
    return beta


def _solve_weighted_project_factor(lo, hi, alpha, gamma, bias_d, bias_e, global_x,
                                   MT, X, FX, PRE, c0, c1, k, mu_p):
    beta_batch = np.empty((hi - lo, k), dtype=alpha.dtype)
    for pi, p in enumerate(xrange(lo, hi)):
        m_u, idx_m_u = get_row(MT, p)
        A_u = alpha[idx_m_u]

        x_p, idx_x_p = get_row(X, p)
        G_p = gamma[idx_x_p]

        rsd = x_p - bias_d[p] - bias_e[idx_x_p] - global_x
        rsd = mu_p * rsd

        if FX is not None:
            f_p,_ = get_row(FX, p)
            GTG = G_p.T.dot(G_p*f_p[:, np.newaxis])
        else:
            GTG = G_p.T.dot(G_p)

        GTG = mu_p*GTG

        B = PRE + (c1 - c0)*A_u.T.dot(A_u) + GTG

        a = rsd.dot(G_p) + c1*m_u.dot(A_u)
        beta_batch[pi] = LA.solve(B, a)
        # if lo == 0:
        #     print 'a is \n', a
        #     print 'B is \n', B
        #     print 'RES*****:,',B*beta_batch[pi] - a

    return beta_batch

def update_theta(alpha, bias_b, bias_c, global_y, YT, FYT, lam_theta,
                 n_jobs, batch_size=1000, mu_u = 1):
    '''Update user context latent factors'''
    m, k = alpha.shape  # m: number of users, k: number of factors

    start_idx = range(0, m, batch_size)
    end_idx = start_idx[1:] + [m]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_solve_user_factor)(
            lo, hi, alpha, bias_b, bias_c, global_y, YT, FYT, k, lam_theta, mu_u)
        for lo, hi in zip(start_idx, end_idx))
    theta = np.vstack(res)
    return theta


def _solve_user_factor(lo, hi, alpha, bias_b, bias_c, global_y, YT, FYT, k, lam_theta, mu_u):
    theta_batch = np.empty((hi - lo, k), dtype=alpha.dtype)
    for ib, i in enumerate(xrange(lo, hi)):
        y_i, idx_y_i = get_row(YT, i)
        rsd = y_i - bias_b[idx_y_i] - bias_c[i] - global_y
        A_i = alpha[idx_y_i]

        if FYT is not None:
            f_i, _ = get_row(FYT, i)
            ATA = A_i.T.dot(A_i * f_i[:, np.newaxis])
            rsd *= f_i
        else:
            ATA = A_i.T.dot(A_i)

        B = ATA + lam_theta*np.eye(k, dtype = alpha.dtype)

        a = rsd.dot(A_i)
        a = mu_u * a

        theta_batch[ib] = LA.solve(B, a)
    return theta_batch

def update_gamma(beta, bias_d, bias_e, global_x, XT, FXT, lam_gamma,
                 n_jobs, batch_size=1000, mu_p = 1):
    '''Update user context latent factors'''
    n, k = beta.shape  # n: number of projects, k: number of factors

    start_idx = range(0, n, batch_size)
    end_idx = start_idx[1:] + [n]
    res = Parallel(n_jobs=n_jobs)(
        delayed(_solve_project_factor)(
            lo, hi, beta, bias_d, bias_e, global_x, XT, FXT, k, lam_gamma, mu_p)
        for lo, hi in zip(start_idx, end_idx))
    gamma = np.vstack(res)
    return gamma


def _solve_project_factor(lo, hi, beta, bias_d, bias_e, global_x, XT, FXT, k, lam_gamma, mu_p):
    #here I used k as the column index so I need to use different one
    gamma_batch = np.empty((hi - lo, k), dtype=beta.dtype)
    for jb, j in enumerate(xrange(lo, hi)):
        x_j, idx_x_j = get_row(XT, j)
        rsd = x_j - bias_d[idx_x_j] - bias_e[j] - global_x
        B_j = beta[idx_x_j]

        if FXT is not None:
            f_j, _ = get_row(FXT, j)
            BTB = B_j.T.dot(B_j * f_j[:, np.newaxis])
            rsd *= f_j
        else:
            BTB = B_j.T.dot(B_j)


        B = BTB + lam_gamma*np.eye(k, dtype = beta.dtype)

        a = rsd.dot(B_j)
        a = mu_p*a
        gamma_batch[jb] = LA.solve(B, a)
    return gamma_batch



def update_user_biases(alpha, theta, bias, global_y, Y, FY,
                n_jobs = 8, batch_size=1000, mu_u = 1):
    ''' Update the bias term.
    '''
    m = alpha.shape[0] #m is the number of users

    start_idx = range(0, m, batch_size)
    end_idx = start_idx[1:] + [m]

    res = Parallel(n_jobs=n_jobs)(
        delayed(_solve_user_bias)(lo, hi, alpha, theta, bias, global_y, Y, FY, mu_u)
        for lo, hi in zip(start_idx, end_idx))
    user_bias = np.hstack(res)
    return user_bias


def _solve_user_bias(lo, hi, alpha, theta, bias, global_y, Y, FY, mu_u):
    bias_batch = np.empty(hi - lo, dtype=alpha.dtype)
    for ub, u in enumerate(xrange(lo, hi)):
        y_u, idx_y_u = get_row(Y, u)
        A_u = alpha[u]
        T_j = theta[idx_y_u]
        bias_j = bias[idx_y_u]
        # rsd = y_u - T_j.dot(A_u) - bias_j - global_y #github
        rsd = y_u - A_u.dot(T_j.T) - bias_j - global_y   #mine
        rsd = mu_u*rsd
        if FY is not None:
            f_u, _ = get_row(FY, u)
            rsd *= f_u

        if rsd.size > 0:
            bias_batch[ub] = rsd.mean()
        else:
            bias_batch[ub] = 0.
    return bias_batch

def update_project_bias(beta, gamma, bias, global_x, X, FX,
                n_jobs = 8, batch_size=1000, mu_p = 1):
    ''' Update the bias term.
    '''
    n = beta.shape[0] #n is the number of users

    start_idx = range(0, n, batch_size)
    end_idx = start_idx[1:] + [n]

    res = Parallel(n_jobs=n_jobs)(
        delayed(_solve_project_bias)(lo, hi, beta, gamma, bias, global_x, X, FX, mu_p)
        for lo, hi in zip(start_idx, end_idx))
    project_bias = np.hstack(res)
    return project_bias


def _solve_project_bias(lo, hi, beta, gamma, bias, global_x, X, FX, mu_p = 1):
    bias_batch = np.empty(hi - lo, dtype=beta.dtype)
    for pb, p in enumerate(xrange(lo, hi)):
        x_p, idx_x_p = get_row(X, p)
        B_u = beta[p]
        G_j = gamma[idx_x_p]
        bias_k = bias[idx_x_p]
        # rsd = x_p - G_j.dot(B_u) - bias_k - global_x  #github
        rsd = x_p - B_u.dot(G_j.T) - bias_k - global_x    #mine
        rsd = mu_p*rsd
        if FX is not None:
            f_p, _ = get_row(FX, p)
            rsd *= f_p

        if rsd.size > 0:
            bias_batch[pb] = rsd.mean()
        else:
            bias_batch[pb] = 0.
    return bias_batch

#######UPDATE For two intercepts: global_x for X matrix and global_y for Y matrix
def update_global(beta, gamma, bias_d, bias_e, X, FX, n_jobs, batch_size=1000, mu = 1):
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
    for ib, i in enumerate(xrange(lo, hi)):
        m_i, idx_i = get_row(X, i)
        m_i_hat = gamma[idx_i].dot(beta[i]) + bias_d[i] + bias_e[idx_i]
        rsd = m_i - m_i_hat
        rsd = mu*rsd
        if FX is not None:
            f_i, _ = get_row(FX, i)
            rsd *= f_i
        res += rsd.sum()
    return res