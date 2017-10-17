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

DEBUG_MODE = True
DATA_DIR = 'data/rec_data/all'
if DEBUG_MODE:
    DATA_DIR = 'data/rec_data/debug'
unique_uid = list()
with open(os.path.join(DATA_DIR, 'unique_uid.txt'), 'r') as f:
    for line in f:
        unique_uid.append(line.strip())

unique_movieId = list()
with open(os.path.join(DATA_DIR, 'unique_mid.txt'), 'r') as f:
    for line in f:
        unique_movieId.append(line.strip())
n_projects = len(unique_movieId)
n_users = len(unique_uid)
print n_users, n_projects

def load_data(csv_file, shape=(n_users, n_projects)):
    tp = pd.read_csv(csv_file)
    timestamps, rows, cols = np.array(tp['timestamp']), np.array(tp['userId']), np.array(tp['movieId']) #rows will be user ids, cols will be projects-ids.
    seq = np.concatenate((rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int'), timestamps[:, None]), axis=1)
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
    return data, seq


vad_data, vad_raw = load_data(os.path.join(DATA_DIR, 'validation.csv'))
test_data, test_raw = load_data(os.path.join(DATA_DIR, 'test.csv'))
#train the model
num_factors = 100
num_iters = 50
batch_size = 1000

n_jobs = 1
lam_theta = lam_beta = 1e-1

best_ndcg = -np.inf
U_best = None
V_best = None
best_alpha = 0

#for alpha in [2, 5, 10, 30, 50]:
for FOLD in range(10):
    print '*************************************FOLD %d ******************************************' % FOLD
    train_data, train_raw = load_data(os.path.join(DATA_DIR, 'train_fold%d.csv'%FOLD))
    for alpha in [20.]:
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


    recall100,ndcg100,map100 = 0.0, 0.0, 0.0

    # alpha = 10 gives the best validation performance
    for K in [5,10,20,50,100]:
        recall = rec_eval.parallel_recall_at_k(train_data, test_data, U_best, V_best, k=K, vad_data=vad_data)
        print 'Test Recall@%d: %.4f' % (K, recall)
        ndcg = rec_eval.parallel_normalized_dcg_at_k(train_data, test_data, U_best, V_best, k=K, vad_data=vad_data)
        print 'Test NDCG@%d: %.4f' % (K, ndcg)
        map = rec_eval.parallel_map_at_k(train_data, test_data, U_best, V_best, k=K, vad_data=vad_data)
        print 'Test MAP@%d: %.4f' % (K, map)
        if K == 100:
            recall100 = recall
            ndcg100 = ndcg
            map100 = map
    fold_str = 'fold%d_'%FOLD
    np.savez('Baseline1_MF_%srecall100_%.4f_ndcg100_%.4f_map100_%.4f.npz'%(fold_str, recall100, ndcg100, map100), U=U_best, V=V_best)
