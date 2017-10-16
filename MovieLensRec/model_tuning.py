import glob
import os
import numpy as np
import rec_eval as rec_eval
# import mf_pos_embedding_user_project as mymodel2
import parallel_mf_pos_user_pos_project as mymodel2
import parallel_mf_posneg_embedding_project as mymodel3
import parallel_mf_posneg_embedding_user as mymodel4
import parallel_mf_pos_user_posneg_project as mymodel5
import parallel_mf_posneg_user_pos_project as mymodel6
import parallel_mf_posneg_embedding_user_project as mymodel7
import cofactor as cofactor
class ModelTuning:

    def __init__(self, train_data, vad_data, test_data,
                 X_sppmi, X_neg_sppmi, Y_sppmi, Y_neg_sppmi,
                 save_dir ):

        self.save_dir = save_dir

        self.train_data = train_data
        self.test_data = test_data
        self.vad_data = vad_data
        self.X_sppmi = X_sppmi
        self.Y_sppmi = Y_sppmi
        self.X_neg_sppmi = X_neg_sppmi
        self.Y_neg_sppmi = Y_neg_sppmi

    def clean_savedir(self):
        print 'cleaning folder'
        lst = glob.glob(os.path.join(self.save_dir, '*.npz'))
        for f in lst:
            os.remove(f)
    def local_alone_eval(self, U, V):
        recall100 = 0.0
        ndcg100 = 0.0
        map100 = 0.0
        for K in [5,10,20,50,100]:
            recall_at_K = rec_eval.parallel_recall_at_k(self.train_data, self.test_data, U, V, k=K,
                                                        vad_data=self.vad_data, n_jobs=4, clear_invalid=False)
            print 'Test Recall@%d: %.4f' % (K, recall_at_K)
            ndcg_at_K = rec_eval.parallel_normalized_dcg_at_k(self.train_data, self.test_data, U, V, k=K,
                                                              vad_data=self.vad_data, n_jobs=4, clear_invalid=False)
            print 'Test NDCG@%d: %.4f' % (K, ndcg_at_K)
            map_at_K = rec_eval.parallel_map_at_k(self.train_data, self.test_data, U, V, k=K,
                                                  vad_data=self.vad_data, n_jobs=4, clear_invalid=False)
            print 'Test MAP@%d: %.4f' % (K, map_at_K)
            if K == 100:
                recall100 = recall_at_K
                ndcg100 = ndcg_at_K
                map100 = map_at_K
        return (recall100, ndcg100, map100)
    def local_eval(self, U, V, best_ndcg_10):
        best_U = None
        best_V = None
        is_better = False
        for K in [5,10,20,50,100]:
            recall_at_K = rec_eval.parallel_recall_at_k(self.train_data, self.test_data, U, V, k=K, vad_data=self.vad_data, n_jobs=4)
            print 'Test Recall@%d: %.4f' % (K, recall_at_K)
            ndcg_at_K = rec_eval.parallel_normalized_dcg_at_k(self.train_data, self.test_data, U, V, k=K, vad_data=self.vad_data, n_jobs=4)
            print 'Test NDCG@%d: %.4f' % (K, ndcg_at_K)
            map_at_K = rec_eval.parallel_map_at_k(self.train_data, self.test_data, U, V, k=K, vad_data=self.vad_data, n_jobs=4)
            print 'Test MAP@%d: %.4f' % (K, map_at_K)
            if K == 10:
                if ndcg_at_K > best_ndcg_10:
                    best_ndcg_10 = ndcg_at_K
                    best_U = U
                    best_V = V
                    is_better = True
        return (is_better, best_U, best_V, best_ndcg_10)
    def run_alone(self, type, n_jobs = 8, n_components = 100, max_iter = 50, vad_K = 100, **kwargs):
        lam_alpha = lam_beta = 1e-1
        lam_theta = lam_gamma = 1e-1
        lam_gamma_p = 1e-1
        lam_gamma_n = 1e-1
        lam_theta_p = 1e-1
        lam_theta_n = 1e-1
        c0 = 1.
        c1 = 20.
        fold = kwargs.get('fold', -1)
        if fold == -1:
            fold = ''
        else:
            fold = 'fold%d_'%fold
        if type == 'cofactor':
            print 'cofactor model'
            print self.save_dir
            self.clean_savedir()
            coder = cofactor.CoFacto(
                n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, dtype=np.float32,
                n_jobs=n_jobs,
                random_state=98765, save_params=True, save_dir=self.save_dir, early_stopping=True, verbose=True,
                lambda_alpha=lam_alpha, lambda_theta=lam_theta, lambda_beta=lam_beta, lambda_gamma=lam_gamma, c0=c0,
                c1=c1)
            coder.fit(self.train_data, self.X_sppmi, vad_data=self.vad_data, batch_users=300, k=vad_K,
                      clear_invalid=False, n_jobs = 4)
            self.test_data.data = np.ones_like(self.test_data.data)
            n_params = len(glob.glob(os.path.join(self.save_dir, '*.npz')))
            params = np.load(os.path.join(self.save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
            U, V = params['U'], params['V']
            (recall100, ndcg100, map100) = self.local_alone_eval(U, V)
            model_out_name = 'Cofactor_%srecall100_%.4f_ndcg100_%.4f_map100_%.4f.npz' % (fold,recall100, ndcg100, map100)
            np.savez(model_out_name, U=U, V=V)

        if type == 'model2':
            print 'positive project embedding + positive user embedding'
            mu_p = float(kwargs.get('mu_p_p', 0.6))
            mu_u = float(kwargs.get('mu_u_p', -1.0))
            if mu_u == -1.0:
                mu_u = 1.0 - mu_p
            print 'mu_u = %.2f , mu_p = %.2f' % (mu_u, mu_p)
            print self.save_dir
            self.clean_savedir()
            # coder = mymodel2.MFPositiveUserProjectEmbedding(mu_u = mu_u, mu_p = mu_p,
            #                  n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, dtype=np.float32, n_jobs=n_jobs,
            #                  random_state=98765, save_params=True, save_dir=self.save_dir, early_stopping=True, verbose=True,
            #                  lambda_alpha = lam_alpha, lambda_theta=lam_theta, lambda_beta=lam_beta, lambda_gamma=lam_gamma, c0=c0, c1=c1)
            coder = mymodel2.ParallelMFPosUserPosProjectEmbedding(mu_u = mu_u, mu_p = mu_p,
                             n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, dtype=np.float32, n_jobs=n_jobs,
                             random_state=98765, save_params=True, save_dir=self.save_dir, early_stopping=True, verbose=True,
                             lambda_alpha = lam_alpha, lambda_theta=lam_theta, lambda_beta=lam_beta, lambda_gamma=lam_gamma, c0=c0, c1=c1)
            coder.fit(self.train_data, self.X_sppmi, self.Y_sppmi, vad_data=self.vad_data, batch_users=300, k=vad_K,
                      clear_invalid=False, n_jobs = 4)

            self.test_data.data = np.ones_like(self.test_data.data)
            n_params = len(glob.glob(os.path.join(self.save_dir, '*.npz')))
            params = np.load(os.path.join(self.save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
            U, V = params['U'], params['V']
            (recall100, ndcg100, map100) = self.local_alone_eval(U, V)
            model_out_name = 'Model2_K100_%srecall100_%.4f_ndcg100_%.4f_map100_%.4f.npz'%(fold, recall100, ndcg100, map100)
            np.savez(model_out_name, U=U, V=V)

        if type == 'model3':
            print 'positive project embedding + negative project embedding'
            mu_p_p = float(kwargs.get('mu_p_p', 0.4))
            mu_p_n = float(kwargs.get('mu_p_n', -1.0))
            if mu_p_n == -1.0:
                mu_p_n = 1.0 - mu_p_p
            print 'mu_p_p = %.2f , mu_p_n = %.2f' % (mu_p_p, mu_p_n)
            print self.save_dir
            self.clean_savedir()

            coder = mymodel3.ParallelMFPositiveNegativeProjectEmbedding(mu_p_p=mu_p_p, mu_p_n=mu_p_n,
                             n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, dtype=np.float32, n_jobs=n_jobs,
                             random_state=98765, save_params=True, save_dir=self.save_dir, early_stopping=True, verbose=True,
                             lambda_alpha=lam_alpha, lambda_theta=lam_theta, lambda_beta=lam_beta,
                             lambda_gamma_p=lam_gamma_p, lambda_gamma_n=lam_gamma_n, c0=c0, c1=c1)  # lambda_gamma = 1e-1
            coder.fit(self.train_data, self.X_sppmi, self.X_neg_sppmi, vad_data=self.vad_data, batch_users=300, k=vad_K,
                      clear_invalid=False, n_jobs = 4) #for active_cate selection, should use clear_invalid = False

            self.test_data.data = np.ones_like(self.test_data.data)
            n_params = len(glob.glob(os.path.join(self.save_dir, '*.npz')))
            params = np.load(os.path.join(self.save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
            U, V = params['U'], params['V']
            (recall100, ndcg100, map100) = self.local_alone_eval(U, V)
            model_out_name = 'Model3_K100_%srecall100_%.4f_ndcg100_%.4f_map100_%.4f.npz' % (fold, recall100, ndcg100, map100)
            np.savez(model_out_name, U=U, V=V)

        if type == 'model4':
            print 'positive user embedding + negative user embedding'
            mu_u_p = float(kwargs.get('mu_u_p', 0.4))
            mu_u_n = float(kwargs.get('mu_u_n', -1.0))
            if mu_u_n == -1.0:
                mu_u_n = 1.0 - mu_u_p
            print 'mu_u_p = %.2f , mu_u_n = %.2f' % (mu_u_p, mu_u_n)
            print self.save_dir
            self.clean_savedir()

            coder = mymodel4.ParallelMFPositiveNegativeUserEmbedding(mu_u_p=mu_u_p, mu_u_n=mu_u_n,
                                n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, dtype=np.float32, n_jobs=n_jobs,
                                random_state=98765, save_params=True, save_dir=self.save_dir, early_stopping=True, verbose=True,
                                lambda_alpha=lam_alpha, lambda_theta_p=lam_theta_p, lambda_theta_n=lam_theta_n,
                                lambda_beta=lam_beta, c0 = c0, c1=c1)
            coder.fit(self.train_data, self.Y_sppmi, self.Y_neg_sppmi, vad_data=self.vad_data, batch_users=300, k=vad_K,
                      clear_invalid=False, n_jobs = 4)

            self.test_data.data = np.ones_like(self.test_data.data)
            n_params = len(glob.glob(os.path.join(self.save_dir, '*.npz')))
            params = np.load(os.path.join(self.save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
            U, V = params['U'], params['V']
            (recall100, ndcg100, map100) = self.local_alone_eval(U, V)
            model_out_name = 'Model4_K100_%srecall100_%.4f_ndcg100_%.4f_map100_%.4f.npz' % (fold, recall100, ndcg100, map100)
            np.savez(model_out_name, U=U, V=V)
        if type == 'model5':
            print 'positive and negative project embedding + positive user embedding'
            mu_p_p = float(kwargs.get('mu_p_p', 0.4))
            mu_p_n = float(kwargs.get('mu_p_n', 0.4))
            mu_u_p = float(kwargs.get('mu_u_p', -1.0))
            if mu_u_p == -1.0:
                mu_u_p = 1.0 - mu_p_p - mu_p_n

            print 'mu_u_p = %.1f, mu_p_p = %.1f, mu_p_n = %.1f' % (mu_u_p, mu_p_p, mu_p_n)
            print self.save_dir
            self.clean_savedir()

            coder = mymodel5.ParallelMFPosUserPosNegProjectEmbedding(mu_u_p=mu_u_p, mu_p_p=mu_p_p, mu_p_n=mu_p_n,
                             n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, dtype=np.float32, n_jobs=n_jobs,
                             random_state=98765, save_params=True, save_dir=self.save_dir, early_stopping=True, verbose=True,
                             lambda_alpha=lam_alpha, lambda_theta_p=lam_theta_p,
                             lambda_beta=lam_beta, lambda_gamma_p=lam_gamma_p, lambda_gamma_n=lam_gamma_n,
                             c0=c0, c1=c1)
            coder.fit(self.train_data, self.X_sppmi, self.X_neg_sppmi, self.Y_sppmi,
                      vad_data=self.vad_data, batch_users=300, k=vad_K, clear_invalid=False, n_jobs = 4)

            self.test_data.data = np.ones_like(self.test_data.data)
            n_params = len(glob.glob(os.path.join(self.save_dir, '*.npz')))
            params = np.load(os.path.join(self.save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
            U, V = params['U'], params['V']
            (recall100, ndcg100, map100) = self.local_alone_eval(U, V)
            model_out_name = 'Model5_K100_%srecall100_%.4f_ndcg100_%.4f_map100_%.4f.npz' % (fold, recall100, ndcg100, map100)
            np.savez(model_out_name, U=U, V=V)
        if type == 'model6':
            print 'positive project embedding + positive + negative user embedding'
            mu_u_p = float(kwargs.get('mu_u_p', 0.4))
            mu_u_n = float(kwargs.get('mu_u_n', 0.4))
            mu_p_p = float(kwargs.get('mu_p_p', -1.0))
            if mu_p_p == -1.0:
                mu_p_p = 1.0 - mu_u_p - mu_u_n

            print 'mu_u_p = %.1f, mu_u_n = %.1f, mu_p_p = %.1f'%(mu_u_p, mu_u_n, mu_p_p)
            print self.save_dir
            self.clean_savedir()

            coder = mymodel6.ParallelMFPosNegUserPosProjectEmbedding(mu_u_p=mu_u_p, mu_u_n=mu_u_n, mu_p_p=mu_p_p,
                             n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, dtype=np.float32, n_jobs=n_jobs,
                             random_state=98765, save_params=True, save_dir=self.save_dir, early_stopping=True, verbose=True,
                             lambda_alpha=lam_alpha, lambda_theta_p=lam_theta_p, lambda_theta_n=lam_theta_n,
                             lambda_beta=lam_beta, lambda_gamma_p=lam_gamma_p,
                              c0=c0, c1=c1)

            coder.fit(self.train_data, self.X_sppmi, self.Y_sppmi, self.Y_neg_sppmi,
                      vad_data=self.vad_data, batch_users=300, k=vad_K, clear_invalid=False, n_jobs = 4)

            self.test_data.data = np.ones_like(self.test_data.data)
            n_params = len(glob.glob(os.path.join(self.save_dir, '*.npz')))
            params = np.load(os.path.join(self.save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
            U, V = params['U'], params['V']
            (recall100, ndcg100, map100) = self.local_alone_eval(U, V)
            model_out_name = 'Model6_K100_%srecall100_%.4f_ndcg100_%.4f_map100_%.4f.npz' % (fold, recall100, ndcg100, map100)
            np.savez(model_out_name, U=U, V=V)
        if type == 'model7':
            print 'positive + negative project embedding + positive + negative user embedding'
            mu_u_p = float(kwargs.get('mu_u_p', 0.3))
            mu_u_n = float(kwargs.get('mu_u_n', 0.3))
            mu_p_p = float(kwargs.get('mu_p_p', 0.3))
            mu_p_n = float(kwargs.get('mu_p_n', -1.0))
            if mu_p_n == -1.0:
                mu_p_n = 1.0 - mu_u_p - mu_u_n - mu_p_p

            print 'mu_u_p = %.1f, mu_u_n = %.1f, mu_p_p = %.1f, mu_p_n = %.1f'%(mu_u_p, mu_u_n, mu_p_p, mu_p_n)
            print self.save_dir
            self.clean_savedir()

            coder = mymodel7.ParallelMFPositiveNegativeUserProjectEmbedding(mu_u_p=mu_u_p, mu_u_n=mu_u_n, mu_p_p=mu_p_p, mu_p_n=mu_p_n,
                             n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, dtype=np.float32, n_jobs=n_jobs,
                             random_state=98765, save_params=True, save_dir=self.save_dir, early_stopping=True, verbose=True,
                             lambda_alpha=lam_alpha, lambda_theta_p=lam_theta_p, lambda_theta_n=lam_theta_n,
                             lambda_beta=lam_beta, lambda_gamma_p=lam_gamma_p, lambda_gamma_n=lam_gamma_n,
                              c0=c0, c1=c1)
            coder.fit(self.train_data, self.X_sppmi, self.X_neg_sppmi, self.Y_sppmi, self.Y_neg_sppmi,
                      vad_data=self.vad_data, batch_users=300, k=vad_K, clear_invalid=False, n_jobs = 4)

            self.test_data.data = np.ones_like(self.test_data.data)
            n_params = len(glob.glob(os.path.join(self.save_dir, '*.npz')))
            params = np.load(os.path.join(self.save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
            U, V = params['U'], params['V']
            (recall100, ndcg100, map100) = self.local_alone_eval(U, V)
            model_out_name = 'Model7_K100_%srecall100_%.4f_ndcg100_%.4f_map100_%.4f.npz' % (fold, recall100, ndcg100, map100)
            np.savez(model_out_name, U=U, V=V)

    def run(self, type, n_jobs = 8, n_components = 100, max_iter = 50, vad_K = 100):
        lam_alpha = lam_beta = 1e-1
        lam_theta = lam_gamma = 1e-1
        lam_gamma_p = 1e-1
        lam_gamma_n = 1e-1
        lam_theta_p = 1e-1
        lam_theta_n = 1e-1
        c0 = 1.
        c1 = 20.

        # print 'lam_alpha:', lam_alpha
        # print 'lam_beta:', lam_beta
        # print 'lam_theta:', lam_theta
        # print 'lam_gamma:', lam_gamma
        # print 'lam_gamma_p:', lam_gamma_p
        # print 'lam_gamma_n:', lam_gamma_n

        best_ndcg_10 = 0.0
        best_U = None
        best_V = None
        if type == 'cofactor':
            print 'modified cofactor model, using weight for the embedding'
            best_mu = 1.0
            for mu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                print 'mu = %.2f' % (mu)
                print self.save_dir
                self.clean_savedir()
                coder = cofactor.CoFacto(mu = mu,
                                     n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, dtype=np.float32, n_jobs=n_jobs,
                                     random_state=98765, save_params=True, save_dir=self.save_dir, early_stopping=True, verbose=True,
                                     lambda_alpha = lam_alpha, lambda_theta=lam_theta, lambda_beta=lam_beta, lambda_gamma=lam_gamma, c0=c0, c1=c1)
                coder.fit(self.train_data, self.X_sppmi, vad_data=self.vad_data, batch_users=300, k=vad_K,
                          clear_invalid=False, n_jobs = 4)
                self.test_data.data = np.ones_like(self.test_data.data)
                n_params = len(glob.glob(os.path.join(self.save_dir, '*.npz')))
                params = np.load(os.path.join(self.save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
                U, V = params['U'], params['V']
                (is_better, new_U, new_V, new_ndcg_10) = self.local_eval(U, V, best_ndcg_10)
                if is_better:
                    best_ndcg_10 = new_ndcg_10
                    best_U = new_U
                    best_V = new_V
                    best_mu = mu
            print 'Best with mu = %.2f' % (best_mu)
            model_out_name = 'ModifiedCofactor_K100_best_ndcd10_%.4f.npz' % (best_ndcg_10)
            np.savez(model_out_name, U=best_U, V=best_V)

        if type == 'model2':
            print 'positive project embedding + positive user embedding'
            best_mu_u = 0.0
            for mu_u in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                print 'mu_u = %.2f , mu_p = %.2f' % (mu_u, (1.0 - mu_u))
                print self.save_dir
                self.clean_savedir()
                coder = mymodel2.ParallelMFPosUserPosProjectEmbedding(mu_u = mu_u, mu_p = 1.0-mu_u,
                                 n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, dtype=np.float32, n_jobs=n_jobs,
                                 random_state=98765, save_params=True, save_dir=self.save_dir, early_stopping=True, verbose=True,
                                 lambda_alpha = lam_alpha, lambda_theta=lam_theta, lambda_beta=lam_beta, lambda_gamma=lam_gamma, c0=c0, c1=c1)
                coder.fit(self.train_data, self.X_sppmi, self.Y_sppmi, vad_data=self.vad_data, batch_users=300, k=vad_K,
                          clear_invalid=False, n_jobs = 4)

                self.test_data.data = np.ones_like(self.test_data.data)
                n_params = len(glob.glob(os.path.join(self.save_dir, '*.npz')))
                params = np.load(os.path.join(self.save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
                U, V = params['U'], params['V']
                (is_better, new_U, new_V, new_ndcg_10) = self.local_eval(U, V, best_ndcg_10)
                if is_better:
                    best_ndcg_10 = new_ndcg_10
                    best_U = new_U
                    best_V = new_V
                    best_mu_u = mu_u
            print 'Best with mu_u = %.2f and mu_p = %.2f'%(best_mu_u, 1.0 - best_mu_u)
            model_out_name = 'Model2_K100_best_ndcd10_%.4f.npz'%(best_ndcg_10)
            np.savez(model_out_name, U=best_U, V=best_V)

        if type == 'model3':
            print 'positive project embedding + negative project embedding'
            best_mu_p_p = 0.0
            for mu_p_p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                print 'mu_p_p = %.2f , mu_p_n = %.2f' % (mu_p_p, (1.0 - mu_p_p))
                print self.save_dir
                self.clean_savedir()

                coder = mymodel3.ParallelMFPositiveNegativeProjectEmbedding(mu_p_p=mu_p_p, mu_p_n=1.0-mu_p_p,
                                 n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, dtype=np.float32, n_jobs=n_jobs,
                                 random_state=98765, save_params=True, save_dir=self.save_dir, early_stopping=True, verbose=True,
                                 lambda_alpha=lam_alpha, lambda_theta=lam_theta, lambda_beta=lam_beta,
                                 lambda_gamma_p=lam_gamma_p, lambda_gamma_n=lam_gamma_n, c0=c0, c1=c1)  # lambda_gamma = 1e-1
                coder.fit(self.train_data, self.X_sppmi, self.X_neg_sppmi, vad_data=self.vad_data, batch_users=300, k=vad_K,
                          clear_invalid=False, n_jobs = 4)
                self.test_data.data = np.ones_like(self.test_data.data)
                n_params = len(glob.glob(os.path.join(self.save_dir, '*.npz')))
                params = np.load(os.path.join(self.save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
                U, V = params['U'], params['V']
                (is_better, new_U, new_V, new_ndcg_10) = self.local_eval(U, V, best_ndcg_10)
                if is_better:
                    best_ndcg_10 = new_ndcg_10
                    best_U = new_U
                    best_V = new_V
                    best_mu_p_p = mu_p_p
            print 'Best with mu_p_p = %.2f and mu_p_n = %.2f' % (best_mu_p_p, 1.0 - best_mu_p_p)
            model_out_name = 'Model3_K100_best_ndcd10_%.4f.npz'%(best_ndcg_10)
            np.savez(model_out_name, U=best_U, V=best_V)

        if type == 'model4':
            print 'positive user embedding + negative user embedding'
            best_mu_u_p = 0.1
            for mu_u_p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                print 'mu_u_p = %.2f , mu_u_n = %.2f' % (mu_u_p, (1.0 - mu_u_p))
                print self.save_dir
                self.clean_savedir()

                coder = mymodel4.ParallelMFPositiveNegativeUserEmbedding(mu_u_p=mu_u_p, mu_u_n=1.0 - mu_u_p,
                                    n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, dtype=np.float32, n_jobs=n_jobs,
                                    random_state=98765, save_params=True, save_dir=self.save_dir, early_stopping=True, verbose=True,
                                    lambda_alpha=lam_alpha, lambda_theta_p=lam_theta_p, lambda_theta_n=lam_theta_n,
                                    lambda_beta=lam_beta, c0 = c0, c1=c1)
                coder.fit(self.train_data, self.Y_sppmi, self.Y_neg_sppmi, vad_data=self.vad_data, batch_users=300, k=vad_K,
                          clear_invalid=False, n_jobs = 4)

                self.test_data.data = np.ones_like(self.test_data.data)
                n_params = len(glob.glob(os.path.join(self.save_dir, '*.npz')))
                params = np.load(os.path.join(self.save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
                U, V = params['U'], params['V']
                (is_better, new_U, new_V, new_ndcg_10) = self.local_eval(U, V, best_ndcg_10)
                if is_better:
                    best_ndcg_10 = new_ndcg_10
                    best_U = new_U
                    best_V = new_V
                    best_mu_u_p = mu_u_p
            print 'Best with mu_u_p = %.2f and mu_u_n = %.2f' % (best_mu_u_p, 1.0 - best_mu_u_p)
            model_out_name = 'Model4_K100_best_ndcd10_%.4f.npz' % (best_ndcg_10)
            np.savez(model_out_name, U=best_U, V=best_V)
        if type == 'model5':
            print 'positive and negative project embedding + positive user embedding'
            best_mu_u_p = 0.0
            best_mu_p_p = 0.0
            best_mu_p_n = 0.0
            count = 0
            for mu_u_p in np.arange(0.1, 0.9, 0.1):
                for mu_p_p in np.arange(0.1, 1.0 - mu_u_p, 0.1):
                    mu_p_n = 1.0 - mu_u_p - mu_p_p
                    if mu_p_n <= 0.001:
                        continue
                    else:
                        count += 1
                        print 'mu_u_p = %.1f, mu_p_p = %.1f, mu_p_n = %.1f'%(mu_u_p, mu_p_p, mu_p_n)
                        print self.save_dir
                        self.clean_savedir()

                        coder = mymodel5.ParallelMFPosUserPosNegProjectEmbedding(mu_u_p=mu_u_p, mu_p_p=mu_p_p, mu_p_n=mu_p_n,
                                         n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, dtype=np.float32, n_jobs=n_jobs,
                                         random_state=98765, save_params=True, save_dir=self.save_dir, early_stopping=True, verbose=True,
                                         lambda_alpha=lam_alpha, lambda_theta_p=lam_theta_p,
                                         lambda_beta=lam_beta, lambda_gamma_p=lam_gamma_p, lambda_gamma_n=lam_gamma_n,
                                         c0=c0, c1=c1)
                        coder.fit(self.train_data, self.X_sppmi, self.X_neg_sppmi, self.Y_sppmi,
                                  vad_data=self.vad_data, batch_users=300, k=vad_K, clear_invalid=False, n_jobs = 4)

                        self.test_data.data = np.ones_like(self.test_data.data)
                        n_params = len(glob.glob(os.path.join(self.save_dir, '*.npz')))
                        params = np.load(
                            os.path.join(self.save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
                        U, V = params['U'], params['V']
                        (is_better, new_U, new_V, new_ndcg_10) = self.local_eval(U, V, best_ndcg_10)
                        if is_better:
                            best_ndcg_10 = new_ndcg_10
                            best_U = new_U
                            best_V = new_V
                            best_mu_u_p = mu_u_p
                            best_mu_p_p = mu_p_p
                            best_mu_p_n = mu_p_n
            print count, ' cases'
            print 'Best with mu_u_p = %.1f, mu_p_p = %.1f, mu_p_n = %.1f'% (best_mu_u_p, best_mu_p_p, best_mu_p_n)
            model_out_name = 'Model5_K100_best_ndcd10_%.4f.npz' % (best_ndcg_10)
            np.savez(model_out_name, U=best_U, V=best_V)
        if type == 'model6':
            print 'positive project embedding + positive + negative user embedding'
            best_mu_u_p = 0.0
            best_mu_u_n = 0.0
            best_mu_p_p = 0.0
            count = 0
            for mu_u_p in np.arange(0.1, 0.9, 0.1):
                for mu_u_n in np.arange(0.1, 1.0 - mu_u_p, 0.1):
                    mu_p_p = 1.0 - mu_u_p - mu_u_n
                    if mu_p_p <= 0.001:
                        continue
                    else:
                        count += 1
                        print 'mu_u_p = %.1f, mu_u_n = %.1f, mu_p_p = %.1f'%(mu_u_p, mu_u_n, mu_p_p)
                        print self.save_dir
                        self.clean_savedir()

                        coder = mymodel6.ParallelMFPosNegUserPosProjectEmbedding(mu_u_p=mu_u_p, mu_u_n=mu_u_n, mu_p_p=mu_p_p,
                                         n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, dtype=np.float32, n_jobs=n_jobs,
                                         random_state=98765, save_params=True, save_dir=self.save_dir, early_stopping=True, verbose=True,
                                         lambda_alpha=lam_alpha, lambda_theta_p=lam_theta_p, lambda_theta_n=lam_theta_n,
                                         lambda_beta=lam_beta, lambda_gamma_p=lam_gamma_p,
                                          c0=c0, c1=c1)
                        coder.fit(self.train_data, self.X_sppmi, self.Y_sppmi, self.Y_neg_sppmi,
                                  vad_data=self.vad_data, batch_users=300, k=vad_K, clear_invalid=False, n_jobs = 4)

                        self.test_data.data = np.ones_like(self.test_data.data)
                        n_params = len(glob.glob(os.path.join(self.save_dir, '*.npz')))
                        params = np.load(
                            os.path.join(self.save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
                        U, V = params['U'], params['V']
                        (is_better, new_U, new_V, new_ndcg_10) = self.local_eval(U, V, best_ndcg_10)
                        if is_better:
                            best_ndcg_10 = new_ndcg_10
                            best_U = new_U
                            best_V = new_V
                            best_mu_u_p = mu_u_p
                            best_mu_u_n = mu_u_n
                            best_mu_p_p = mu_p_p
            print count, ' cases'
            print 'Best with mu_u_p = %.1f, mu_u_n = %.1f, mu_p_p = %.1f'% (best_mu_u_p, best_mu_u_n, best_mu_p_p)
            model_out_name = 'Model6_K100_best_ndcd10_%.4f.npz' % (best_ndcg_10)
            np.savez(model_out_name, U=best_U, V=best_V)
        if type == 'model7':
            print 'positive + negative project embedding + positive + negative user embedding'
            best_mu_u_p = 0.0
            best_mu_u_n = 0.0
            best_mu_p_p = 0.0
            best_mu_p_n = 0.0
            count = 0
            for mu_u_p in np.arange(0.1, 0.8, 0.1):
                for mu_u_n in np.arange(0.1, 0.9 - mu_u_p, 0.1):
                    for mu_p_p in np.arange(0.1, 1.0 - mu_u_p - mu_u_n, 0.1):
                        mu_p_n = 1.0 - mu_u_p - mu_u_n - mu_p_p
                        if mu_p_n <= 0.001:
                            continue
                        else:
                            count += 1
                            print 'mu_u_p = %.1f, mu_u_n = %.1f, mu_p_p = %.1f, mu_p_n = %.1f'%(mu_u_p, mu_u_n, mu_p_p, mu_p_n)
                            print self.save_dir
                            self.clean_savedir()

                            coder = mymodel7.ParallelMFPositiveNegativeUserProjectEmbedding(mu_u_p=mu_u_p, mu_u_n=mu_u_n, mu_p_p=mu_p_p, mu_p_n=mu_p_n,
                                             n_components=n_components, max_iter=max_iter, batch_size=1000, init_std=0.01, dtype=np.float32, n_jobs=n_jobs,
                                             random_state=98765, save_params=True, save_dir=self.save_dir, early_stopping=True, verbose=True,
                                             lambda_alpha=lam_alpha, lambda_theta_p=lam_theta_p, lambda_theta_n=lam_theta_n,
                                             lambda_beta=lam_beta, lambda_gamma_p=lam_gamma_p, lambda_gamma_n=lam_gamma_n,
                                              c0=c0, c1=c1)
                            coder.fit(self.train_data, self.X_sppmi, self.X_neg_sppmi, self.Y_sppmi, self.Y_neg_sppmi,
                                      vad_data=self.vad_data, batch_users=300, k=vad_K, clear_invalid=False, n_jobs = 4)

                            self.test_data.data = np.ones_like(self.test_data.data)
                            n_params = len(glob.glob(os.path.join(self.save_dir, '*.npz')))
                            params = np.load(
                                os.path.join(self.save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
                            U, V = params['U'], params['V']
                            (is_better, new_U, new_V, new_ndcg_10) = self.local_eval(U, V, best_ndcg_10)
                            if is_better:
                                best_ndcg_10 = new_ndcg_10
                                best_U = new_U
                                best_V = new_V
                                best_mu_u_p = mu_u_p
                                best_mu_u_n = mu_u_n
                                best_mu_p_p = mu_p_p
                                best_mu_p_n = mu_p_n
            print count, ' cases'
            print 'Best with mu_u_p = %.1f, mu_u_n = %.1f, mu_p_p = %.1f, mu_p_n = %.1f'% (best_mu_u_p, best_mu_u_n, best_mu_p_p, best_mu_p_n)
            model_out_name = 'Model7_K100_best_ndcd10_%.4f.npz' % (best_ndcg_10)
            np.savez(model_out_name, U=best_U, V=best_V)

