import itertools
import os
import sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from scipy import sparse

import content_wmf
import batched_inv_joblib
import rec_eval
import glob

DEBUG_MODE = False
DATA_DIR = 'data/rec_data/all'
if DEBUG_MODE:
    DATA_DIR = 'data/rec_data/debug'
unique_uid = list()
with open(os.path.join(DATA_DIR, 'unique_uid_sub.txt'), 'r') as f:
    for line in f:
        unique_uid.append(line.strip())

unique_sid = list()
with open(os.path.join(DATA_DIR, 'unique_sid_sub.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())
n_projects = len(unique_sid)
n_users = len(unique_uid)
print n_users, n_projects

def load_data(csv_file, shape=(n_users, n_projects)):
    tp = pd.read_csv(csv_file)
    count, rows, cols = np.array(tp['count']), np.array(tp['uid']), np.array(tp['sid']) #rows will be user ids, cols will be projects-ids.
    seq = np.concatenate((  rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int'),
                            count[:, None]
                          ), axis=1)
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
    return data, seq, tp

for FOLD in range(5):
    print '********************************** FOLD %d********************************'%FOLD
    vad_data, vad_raw, vad_df = load_data(os.path.join(DATA_DIR, 'vad.num.sub.fold%d.csv'%FOLD))
    test_data, test_raw, test_df = load_data(os.path.join(DATA_DIR, 'test.num.sub.fold%d.csv'%FOLD))
    train_data, train_raw, train_df =  load_data(os.path.join(DATA_DIR, 'train.num.sub.fold%d.csv'%FOLD))
    #train the model
    num_factors = 100
    num_iters = 50
    batch_size = 1000

    n_jobs = 1
    lam_theta = lam_beta = 1e-1 #grid search [1e-5, 1e-3, 1e-1, 1, 10]

    best_ndcg = -np.inf
    U_best = None
    V_best = None
    best_alpha = 0

    #for alpha in [2, 5, 10, 20, 30, 50]:
    for alpha in [20]:
        S = content_wmf.linear_surplus_confidence_matrix(train_data, alpha=alpha)

        U, V, vad_ndcg = content_wmf.factorize(S, num_factors, vad_data=vad_data, num_iters=num_iters,
                                           init_std=0.01, lambda_U_reg=lam_theta, lambda_V_reg=lam_beta,
                                           dtype='float32', random_state=98765, verbose=True,
                                           recompute_factors=batched_inv_joblib.recompute_factors_batched,
                                           batch_size=batch_size, n_jobs=n_jobs)
        if vad_ndcg > best_ndcg:
            best_ndcg = vad_ndcg
            U_best = U.copy()
            V_best = V.copy()
            best_alpha = alpha
    print best_alpha, best_ndcg

    np.savez('Baseline1_MF_K100.npz', U=U_best, V=V_best)

    # alpha = 20 gives the best validation performance
    for K in [5, 10,20,50,100]:
        print 'Test Recall@%d: %.4f' % (K, rec_eval.parallel_recall_at_k(train_data, test_data, U_best, V_best, k=K, vad_data=vad_data, n_jobs=16))
        print 'Test NDCG@%d: %.4f' % (K, rec_eval.parallel_normalized_dcg_at_k(train_data, test_data, U_best, V_best, k=K, vad_data=vad_data, n_jobs=16))
        print 'Test MAP@%d: %.4f' % (K, rec_eval.parallel_map_at_k(train_data, test_data, U_best, V_best, k=K, vad_data=vad_data, n_jobs=16))
