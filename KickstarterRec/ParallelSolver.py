import threading
import os
import sys
import time
import numpy as np
from numpy import linalg as LA

def get_row(Y, i):
    '''Given a scipy.sparse.csr_matrix Y, get the values and indices of the
    non-zero values in i_th row'''
    lo, hi = Y.indptr[i], Y.indptr[i + 1]
    return Y.data[lo:hi], Y.indices[lo:hi]
class UpdateUserFactorParallel:
    def __init__(self,
                 beta, theta_p = None, theta_n = None,
                 bias_b_p = None, bias_c_p = None, global_y_p = None,
                 bias_b_n = None, bias_c_n = None, global_y_n = None,
                 M = None, XP = None, FXP = None, XN = None, FXN = None, BTBpR = None,
                 c0 = 1.0, c1 = 20.0, f = 100, mu_u_p = 1, mu_u_n = 1,
                 n_jobs = 8,  mode=None):

        # self.para_type = para_type  #update user factor, project factor,
        #                             #cofactor, bias, or global term?
        #                             # value will be : user, project, embedding,
        #                             # bias, and global
        self.beta = beta
        self.theta_p = theta_p
        self.theta_n = theta_n
        self.bias_b_p = bias_b_p
        self.bias_c_p = bias_c_p
        self.global_y_p = global_y_p
        self.bias_b_n = bias_b_n
        self.bias_c_n = bias_c_n
        self.global_y_n = global_y_n
        self.M = M
        self.XP = XP
        self.Y = XP
        self.FXP = FXP
        self.XN = XN
        self.FXN = FXN
        self.BTBpR = BTBpR
        self.c0 = c0
        self.c1 = c1
        self.f = f #number of latent factors
        self.n_jobs = n_jobs
        self.mu_u_p = mu_u_p
        self.mu_u_n = mu_u_n
        self.mode = mode            # value for this param is : None, positive, negative, hybrid
                                    # None means no embedding for this para_type
                                    # positive means applying positive embedding for this para_type
                                    # negative means applying negative embedding for this para_type
                                    # hybrid: combine positive + negative embeddding
        self.thread_lock = threading.Lock()
        self.m = M.shape[0]  # m: number of users


    def run(self):
        self.alpha = np.zeros((self.m, self.f), dtype=self.beta.dtype)
        step = int(self.m/self.n_jobs)
        self.threads = []
        for i in range(0, self.n_jobs):
            lo = i*step
            hi = (i+1)*step
            if i == (self.n_jobs - 1):
                hi = self.m
            thread = UserFactorUpdateWorker('Thread_%d' % i, self, lo, hi)
            self.threads.append(thread)
            # self.threads = [UserFactorUpdateWorker('Thread_%d' % i, self) for i in range(self.n_jobs)]
        for t in self.threads:
            t.start()
        for t in self.threads:
            t.join()
        return self.alpha

class UserFactorUpdateWorker(threading.Thread):
    def __init__(self, name, parent, lo, hi):
        threading.Thread.__init__(self)
        self.name = name
        self.parent = parent
        self.lo = lo
        self.hi = hi
    def run(self):
        alpha_batch = np.zeros((self.hi - self.lo, self.parent.f), dtype=self.parent.beta.dtype)
        if self.parent.mode == None:
            #update user factor without embedding
            for ui, u in enumerate(xrange(self.lo, self.hi)):
                m_u, idx_m_p = get_row(self.parent.M, u)
                B_p = self.parent.beta[idx_m_p]
                a = m_u.dot(self.parent.c1 * B_p)
                A = self.parent.BTBpR + B_p.T.dot((self.parent.c1 - self.parent.c0) * B_p)
                alpha_batch[ui] = LA.solve(A, a)

        elif self.parent.mode == "positive":
            for ui, u in enumerate(xrange(self.lo, self.hi)):
                m_u, idx_m_p = get_row(self.parent.M, u)
                B_p = self.parent.beta[idx_m_p]

                y_u, idx_y_u = get_row(self.parent.XP, u)
                T_j = self.parent.theta_p[idx_y_u]

                rsd = y_u - self.parent.bias_b_p[u] - self.parent.bias_c_p[idx_y_u] - self.parent.global_y_p

                if self.parent.FXP is not None:  # FY is weighted matrix of Y
                    f_u, _ = get_row(self.parent.FXP, u)
                    TTT = T_j.T.dot(T_j * f_u[:, np.newaxis])
                    rsd *= f_u
                else:
                    TTT = T_j.T.dot(T_j)
                # TTT = self.parent.mu_u_p * TTT

                a = m_u.dot(self.parent.c1 * B_p) + self.parent.mu_u_p * np.dot(rsd, T_j)
                A = self.parent.BTBpR + B_p.T.dot((self.parent.c1 - self.parent.c0) * B_p) + TTT
                alpha_batch[ui] = LA.solve(A, a)

        elif self.parent.mode == "negative":
            for ui, u in enumerate(xrange(self.lo, self.hi)):
                m_u, idx_m_p = get_row(self.parent.M, u)
                B_p = self.parent.beta[idx_m_p]

                y_u, idx_y_u = get_row(self.parent.XN, u)
                T_j = self.parent.theta_n[idx_y_u]

                rsd = y_u - self.parent.bias_b_n[u] - self.parent.bias_c_n[idx_y_u] - self.parent.global_y_n

                if self.parent.FXN is not None:  # FY is weighted matrix of Y
                    f_u, _ = get_row(self.parent.FXN, u)
                    TTT = T_j.T.dot(T_j * f_u[:, np.newaxis])
                    rsd *= f_u
                else:
                    TTT = T_j.T.dot(T_j)
                # TTT = self.parent.mu_u_n * TTT

                a = m_u.dot(self.parent.c1 * B_p) + self.parent.mu_u_n * np.dot(rsd, T_j)
                A = self.parent.BTBpR + B_p.T.dot((self.parent.c1 - self.parent.c0) * B_p) + TTT
                alpha_batch[ui] = LA.solve(A, a)
            return alpha_batch
        elif self.parent.mode == "hybird":
            print 'Hybrid mode'
            for ui, u in enumerate(xrange(self.lo, self.hi)):
                m_u, idx_m_p = get_row(self.parent.M, u)
                B_p = self.parent.beta[idx_m_p]

                y_u_p, idx_y_u_p = get_row(self.parent.XP, u)
                T_j_p = self.parent.theta_p[idx_y_u_p]
                rsd_p = y_u_p - self.parent.bias_b_p[u] - self.parent.bias_c_p[idx_y_u_p] - self.parent.global_y_p
                if self.parent.FXP is not None:  # FY is weighted matrix of Y
                    f_u, _ = get_row(self.parent.FXP, u)
                    TTT_p = T_j_p.T.dot(T_j_p * f_u[:, np.newaxis])
                    rsd_p *= f_u
                else:
                    TTT_p = T_j_p.T.dot(T_j_p)
                # TTT_p = self.parent.mu_u_p * TTT_p

                y_u_n, idx_y_u_n = get_row(self.parent.XN, u)
                T_j_n = self.parent.theta_n[idx_y_u_n]
                rsd_n = y_u_n - self.parent.bias_b_n[u] - self.parent.bias_c_n[idx_y_u_n] - self.parent.global_y_n
                if self.parent.FXN is not None:  # FY is weighted matrix of Y
                    f_u, _ = get_row(self.parent.FXN, u)
                    TTT_n = T_j_n.T.dot(T_j_n * f_u[:, np.newaxis])
                    rsd_n *= f_u
                else:
                    TTT_n = T_j_n.T.dot(T_j_n)
                # TTT_n = self.parent.mu_u_n * TTT_n

                a = m_u.dot(self.parent.c1 * B_p) + self.parent.mu_u_p * np.dot(rsd_p, T_j_p) + \
                                                    self.parent.mu_u_n * np.dot(rsd_n, T_j_n)
                A = self.parent.BTBpR + B_p.T.dot((self.parent.c1 - self.parent.c0) * B_p) + TTT_p + TTT_n
                alpha_batch[ui] = LA.solve(A, a)
        self.parent.thread_lock.acquire()
        self.parent.alpha[self.lo:self.hi] = alpha_batch
        self.parent.thread_lock.release()


class UpdateProjectFactorParallel:
    def __init__(self,
                 alpha, gamma_p, gamma_n,
                 bias_d_p, bias_e_p, global_x_p,
                 bias_d_n, bias_e_n, global_x_n,
                 MT, XP, FXP, XN, FXN, TTTpR,
                 c0, c1, f, mu_p_p = 1, mu_p_n = 1,
                 n_jobs = 8, mode=None):

        # self.para_type = para_type  #update user factor, project factor,
        #                             #cofactor, bias, or global term?
        #                             # value will be : user, project, embedding,
        #                             # bias, and global
        self.alpha = alpha
        self.gamma_p = gamma_p
        self.gamma_n = gamma_n
        self.bias_d_p = bias_d_p
        self.bias_e_p = bias_e_p
        self.global_x_p = global_x_p
        self.bias_d_n = bias_d_n
        self.bias_e_n = bias_e_n
        self.global_x_n = global_x_n
        self.MT = MT
        self.XP = XP
        self.FXP = FXP
        self.XN = XN
        self.FXN = FXN
        self.TTTpR = TTTpR
        self.c0 = c0
        self.c1 = c1
        self.f = f
        self.mu_p_p = mu_p_p
        self.mu_p_n = mu_p_n
        self.n_jobs = n_jobs
        self.mode = mode            # value for this param is : None, positive, negative, hybrid
                                    # None means no embedding for this para_type
                                    # positive means applying positive embedding for this para_type
                                    # negative means applying negative embedding for this para_type
                                    # hybrid: combine positive + negative embeddding
        self.n = MT.shape[0]  # n: number of projects, m: number of users
        self.thread_lock = threading.Lock()


    def run(self):
        self.beta = np.zeros((self.n, self.f), dtype=self.alpha.dtype)
        step = int(self.n/self.n_jobs)
        self.threads = []
        for i in range(0, self.n_jobs):
            lo = i*step
            hi = (i+1)*step
            if i == (self.n_jobs - 1):
                hi = self.n
            thread = ProjectFactorUpdateWorker('Thread_%d' % i, self, lo, hi)
            self.threads.append(thread)
            # self.threads = [UserFactorUpdateWorker('Thread_%d' % i, self) for i in range(self.n_jobs)]
        for t in self.threads:
            t.start()
        for t in self.threads:
            t.join()
        return self.beta

class ProjectFactorUpdateWorker(threading.Thread):
    def __init__(self, name, parent, lo, hi):
        threading.Thread.__init__(self)
        self.name = name
        self.parent = parent
        self.lo = lo
        self.hi = hi
    def run(self):
        beta_batch = np.zeros((self.hi - self.lo, self.parent.f), dtype=self.parent.alpha.dtype)
        if self.parent.mode == None:
            print 'None mode'
            for pi, p in enumerate(xrange(self.lo, self.hi)):
                m_u, idx_u = get_row(self.parent.MT, p)
                A_u = self.parent.alpha[idx_u]

                a = m_u.dot(self.parent.c1*A_u)
                B = self.parent.TTTpR + A_u.T.dot((self.parent.c1 - self.parent.c0) * A_u)
                beta_batch[pi] = LA.solve(B, a)

        elif self.parent.mode == 'positive':
            print 'positive mode'
            for pi, p in enumerate(xrange(self.lo, self.hi)):
                m_u, idx_u = get_row(self.parent.MT, p)
                A_u = self.parent.alpha[idx_u]

                x_pj, idx_x_j = get_row(self.parent.XP, p)
                G_i = self.parent.gamma_p[idx_x_j]

                rsd = x_pj - self.parent.bias_d_p[p] - self.parent.bias_e_p[idx_x_j] - self.parent.global_x_p

                if self.parent.FXP is not None:
                    f_i, _ = get_row(self.parent.FXP, p)
                    GTG = G_i.T.dot(G_i * f_i[:, np.newaxis])
                    rsd *= f_i
                else:
                    GTG = G_i.T.dot(G_i)

                B = self.parent.TTTpR + A_u.T.dot((self.parent.c1 - self.parent.c0) * A_u) + GTG
                a = m_u.dot(self.parent.c1 * A_u) + self.parent.mu_p_p * np.dot(rsd, G_i)
                beta_batch[pi] = LA.solve(B, a)

        elif self.parent.mode == 'negative':
            print 'negative mode'
            for pi, p in enumerate(xrange(self.lo, self.hi)):
                m_u, idx_u = get_row(self.parent.MT, p)
                A_u = self.parent.alpha[idx_u]

                x_pj, idx_x_j = get_row(self.parent.XN, p)
                G_i = self.parent.gamma_n[idx_x_j]

                rsd = x_pj - self.parent.bias_d_n[p] - self.parent.bias_e_n[idx_x_j] - self.parent.global_x_n

                if self.parent.FXN is not None:
                    f_i, _ = get_row(self.parent.FXN, p)
                    GTG = G_i.T.dot(G_i * f_i[:, np.newaxis])
                    rsd *= f_i
                else:
                    GTG = G_i.T.dot(G_i)

                B = self.parent.TTTpR + A_u.T.dot((self.parent.c1 - self.parent.c0) * A_u) + GTG
                a = m_u.dot(self.parent.c1 * A_u) + self.parent.mu_p_n * np.dot(rsd, G_i)

                beta_batch[pi] = LA.solve(B, a)

        elif self.parent.mode == 'hybrid':
            #update project latent factor with pos and neg embedding
            for pi, p in enumerate(xrange(self.lo, self.hi)):
                m_u, idx_u = get_row(self.parent.MT, p)
                A_u = self.parent.alpha[idx_u]

                x_pj_p, idx_x_j_p = get_row(self.parent.XP, p)
                G_i_p = self.parent.gamma_p[idx_x_j_p]
                rsd_p = x_pj_p - self.parent.bias_d_p[p] - self.parent.bias_e_p[idx_x_j_p] - self.parent.global_x_p
                if self.parent.FXP is not None:
                    f_i_p, _ = get_row(self.parent.FXP, p)
                    GTG_p = G_i_p.T.dot(G_i_p * f_i_p[:, np.newaxis])
                    rsd_p *= f_i_p
                else:
                    GTG_p = G_i_p.T.dot(G_i_p)

                x_pj_n, idx_x_j_n = get_row(self.parent.XN, p)
                G_i_n = self.parent.gamma_n[idx_x_j_n]
                rsd_n = x_pj_n - self.parent.bias_d_n[p] - self.parent.bias_e_n[idx_x_j_n] - self.parent.global_x_n
                if self.parent.FXN is not None:
                    f_i_n, _ = get_row(self.parent.FXN, p)
                    GTG_n = G_i_n.T.dot(G_i_n * f_i_n[:, np.newaxis])
                    rsd_n *= f_i_n
                else:
                    GTG_n = G_i_n.T.dot(G_i_n)

                B = self.parent.TTTpR + A_u.T.dot((self.parent.c1 - self.parent.c0) * A_u) + \
                    GTG_p + \
                    self.parent.mu_p_n *GTG_n
                a = m_u.dot(self.parent.c1 * A_u) + self.parent.mu_p_p * np.dot(rsd_p, G_i_p) + \
                                                    np.dot(rsd_n, G_i_n)

                beta_batch[pi] = LA.solve(B, a)

        self.parent.thread_lock.acquire()
        self.parent.beta[self.lo:self.hi] = beta_batch
        self.parent.thread_lock.release()

class UpdateEmbeddingFactorParallel:
    # beta, bias_d, bias_e, global_x, XT, FXT, f, lam_gamma, mu_p
    def __init__(self,
                 main_factor, bias_main, bias_embedding, intercept, XT, FXT, f, lam_embedding, mu=1,
                 n_jobs = 8):

        # self.para_type = para_type  #update user factor, project factor,
        #                             #cofactor, bias, or global term?
        #                             # value will be : user, project, embedding,
        #                             # bias, and global
        self.main_factor = main_factor
        self.bias_main = bias_main
        self.bias_embedding = bias_embedding
        self.intercept = intercept
        self.XT = XT
        self.FXT = FXT
        self.f =  f
        self.lam_embedding = lam_embedding
        self.mu = mu
        self.n_jobs = n_jobs
        self.n = main_factor.shape[0]
        self.thread_lock = threading.Lock()


    def run(self):
        self.embedding_factor = np.zeros((self.n, self.f), dtype=self.main_factor.dtype)
        step = int(self.n/self.n_jobs)
        self.threads = []
        for i in range(0, self.n_jobs):
            lo = i*step
            hi = (i+1)*step
            if i == (self.n_jobs - 1):
                hi = self.n
            thread = EmbeddingFactorUpdateWorker('Thread_%d' % i, self, lo, hi)
            self.threads.append(thread)
            # self.threads = [UserFactorUpdateWorker('Thread_%d' % i, self) for i in range(self.n_jobs)]
        for t in self.threads:
            t.start()
        for t in self.threads:
            t.join()
        return self.embedding_factor

class EmbeddingFactorUpdateWorker(threading.Thread):
    def __init__(self, name, parent, lo, hi):
        threading.Thread.__init__(self)
        self.name = name
        self.parent = parent
        self.lo = lo
        self.hi = hi
    def run(self):
        embedding_batch = np.zeros((self.hi - self.lo, self.parent.f), dtype=self.parent.main_factor.dtype)
        for jb, j in enumerate(xrange(self.lo, self.hi)):
            x_jp, idx_p = get_row(self.parent.XT, j)
            rsd = x_jp - self.parent.bias_main[idx_p] - self.parent.bias_embedding[j] - self.parent.intercept
            B_j = self.parent.main_factor[idx_p]
            if self.parent.FXT is not None:
                f_j, _ = get_row(self.parent.FXT, j)
                BTB = B_j.T.dot(B_j * f_j[:, np.newaxis])
                rsd *= f_j
            else:
                BTB = B_j.T.dot(B_j)

            B = BTB + self.parent.lam_embedding * np.eye(self.parent.f, dtype=self.parent.main_factor.dtype)
            a = self.parent.mu * np.dot(rsd, B_j)
            embedding_batch[jb] = LA.solve(B, a)
        self.parent.thread_lock.acquire()
        self.parent.embedding_factor[self.lo:self.hi] = embedding_batch
        self.parent.thread_lock.release()

class UpdateBiasParallel:
    #beta, gamma, bias_e, global_x, X, FX, mu
    def __init__(self,
                 main_factor, embedding_factor, bias, intercept, X, FX, mu=1,
                 n_jobs = 8):

        # self.para_type = para_type  #update user factor, project factor,
        #                             #cofactor, bias, or global term?
        #                             # value will be : user, project, embedding,
        #                             # bias, and global
        self.main_factor = main_factor
        self.embedding_factor = embedding_factor
        self.bias = bias
        self.intercept = intercept
        self.X = X
        self.FX = FX
        self.mu = mu
        self.n_jobs = n_jobs
        self.n = main_factor.shape[0]
        self.thread_lock = threading.Lock()


    def run(self):
        self.bias = np.zeros(self.n, dtype=self.main_factor.dtype)
        step = int(self.n/self.n_jobs)
        self.threads = []
        for i in range(0, self.n_jobs):
            lo = i*step
            hi = (i+1)*step
            if i == (self.n_jobs - 1):
                hi = self.n
            thread = BiasUpdateWorker('Thread_%d' % i, self, lo, hi)
            self.threads.append(thread)
            # self.threads = [UserFactorUpdateWorker('Thread_%d' % i, self) for i in range(self.n_jobs)]
        for t in self.threads:
            t.start()
        for t in self.threads:
            t.join()
        return self.bias

class BiasUpdateWorker(threading.Thread):
    def __init__(self, name, parent, lo, hi):
        threading.Thread.__init__(self)
        self.name = name
        self.parent = parent
        self.lo = lo
        self.hi = hi
    def run(self):
        bias_batch = np.zeros(self.hi - self.lo, dtype=self.parent.main_factor.dtype)
        if self.parent.mu != 0:
            for ib, i in enumerate(xrange(self.lo, self.hi)):
                m_i, idx_i = get_row(self.parent.X, i)
                m_i_hat = self.parent.embedding_factor[idx_i].dot(self.parent.main_factor[i]) + \
                          self.parent.bias[idx_i] + self.parent.intercept
                rsd = m_i - m_i_hat

                if self.parent.FX is not None:
                    f_i, _ = get_row(self.parent.FX, i)
                    rsd *= f_i

                if rsd.size > 0:
                    bias_batch[ib] = self.parent.mu * rsd.mean()
                else:
                    bias_batch[ib] = 0.
        self.parent.thread_lock.acquire()
        self.parent.bias[self.lo:self.hi] = bias_batch
        self.parent.thread_lock.release()


class UpdateInterceptParallel:
    #beta, gamma, bias_d, bias_e, X, FX, mu
    def __init__(self,
                 main_factor, embedding_factor, bias_main, bias_embedding, X, FX, mu=1,
                 n_jobs = 8):

        # self.para_type = para_type  #update user factor, project factor,
        #                             #cofactor, bias, or global term?
        #                             # value will be : user, project, embedding,
        #                             # bias, and global
        self.main_factor = main_factor
        self.embedding_factor = embedding_factor
        self.bias_main = bias_main
        self.bias_embedding = bias_embedding
        self.X = X
        self.FX = FX
        self.mu = mu
        self.n_jobs = n_jobs
        self.n = main_factor.shape[0]
        self.thread_lock = threading.Lock()


    def run(self):
        step = int(self.n/self.n_jobs)
        self.threads = []
        self.intercept = np.zeros(self.n_jobs, dtype=float)
        for i in range(0, self.n_jobs):
            lo = i*step
            hi = (i+1)*step
            if i == (self.n_jobs - 1):
                hi = self.n
            thread = InterceptUpdateWorker('Thread_%d' % i, i, self, lo, hi)
            self.threads.append(thread)
            # self.threads = [UserFactorUpdateWorker('Thread_%d' % i, self) for i in range(self.n_jobs)]
        for t in self.threads:
            t.start()
        for t in self.threads:
            # print 'thread %s is running'%t.name
            t.join()
        return np.sum(self.intercept)/self.X.data.size

class InterceptUpdateWorker(threading.Thread):
    def __init__(self, name, thread_id, parent, lo, hi):
        threading.Thread.__init__(self)
        self.name = name
        self.thread_id = thread_id
        self.parent = parent
        self.lo = lo
        self.hi = hi
    def run(self):
        res = 0.
        if self.parent.mu != 0:
            for ib, i in enumerate(xrange(self.lo, self.hi)):
                m_i, idx_i = get_row(self.parent.X, i)
                m_i_hat = self.parent.embedding_factor[idx_i].dot(self.parent.main_factor[i]) + \
                          self.parent.bias_main[i] + self.parent.bias_embedding[idx_i]
                rsd = m_i - m_i_hat

                if self.parent.FX is not None:
                    f_i, _ = get_row(self.parent.FX, i)
                    rsd *= f_i
                if rsd.size > 0:
                     res += rsd.sum()
        self.parent.thread_lock.acquire()
        self.parent.intercept[self.thread_id] = res
        self.parent.thread_lock.release()
