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

DEBUG_MODE = False
DATA_DIR = 'data/rec_data/all'
if DEBUG_MODE:
    DATA_DIR = 'data/rec_data/debug'
unique_uid = list()
with open(os.path.join(DATA_DIR, 'unique_uid.txt'), 'r') as f:
    for line in f:
        unique_uid.append(line.strip())

unique_pid = list()
with open(os.path.join(DATA_DIR, 'unique_pid.txt'), 'r') as f:
    for line in f:
        unique_pid.append(line.strip())
n_projects = len(unique_pid)
n_users = len(unique_uid)
print n_users, n_projects

def load_data(csv_file, shape=(n_users, n_projects)):
    tp = pd.read_csv(csv_file)
    timestamps, rows, cols = np.array(tp['timestamp']), np.array(tp['bid']), np.array(tp['pid']) #rows will be user ids, cols will be projects-ids.
    seq = np.concatenate((rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int'), timestamps[:, None]), axis=1)
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
    return data, seq

train_data, train_raw = load_data(os.path.join(DATA_DIR, 'train.csv'))
vad_data, vad_raw = load_data(os.path.join(DATA_DIR, 'validation.csv'))

if len(glob.glob(os.path.join(DATA_DIR, 'project_ts_df.csv'))) <= 0:
    project_ts_df_train = train_df[['pid', 'timestamp', 'timestamp_end']]
    # project_ts_df_train = project_ts_df_train.drop_duplicates(cols='pid').sort_index(by='pid', ascending=True)
    project_ts_df_train = project_ts_df_train.drop_duplicates('pid').sort_index(by='pid', ascending=True)
    project_ts_df_vad = vad_df[['pid', 'timestamp', 'timestamp_end']]
    # project_ts_df_vad = project_ts_df_vad.drop_duplicates(cols='pid').sort_index(by='pid', ascending=True)
    project_ts_df_vad = project_ts_df_vad.drop_duplicates('pid').sort_index(by='pid', ascending=True)
    project_ts_df_test = test_df[['pid', 'timestamp', 'timestamp_end']]
    # project_ts_df_test = project_ts_df_test.drop_duplicates(cols='pid').sort_index(by='pid', ascending=True)
    project_ts_df_test = project_ts_df_test.drop_duplicates('pid').sort_index(by='pid', ascending=True)
    test_data.data = np.ones_like(test_data.data)
    project_ts_df = pd.concat([project_ts_df_train, project_ts_df_test, project_ts_df_vad])
    # project_ts_df = project_ts_df.drop_duplicates(cols='pid').sort_index(by='pid', ascending=True)
    project_ts_df = project_ts_df.drop_duplicates('pid').sort_index(by='pid', ascending=True)
    project_ts_df_train = None
    project_ts_df_vad = None
    project_ts_df_test = None
    project_ts_df = project_ts_df.reset_index(drop=True)

    project_ts_df.to_csv(os.path.join(DATA_DIR, 'project_ts_df.csv'), index=False)
else:
    project_ts_df = pd.read_csv(os.path.join(DATA_DIR, 'project_ts_df.csv'))
rec_eval.project_ts_df = project_ts_df
print 'Project_ts_df took: %d mb' %((sys.getsizeof(project_ts_df))/(1024*1024))


#train the model
num_factors = 100
num_iters = 20
batch_size = 1000

n_jobs = 4
lam_theta = lam_beta = 1e-5

best_ndcg = -np.inf
U_best = None
V_best = None
best_alpha = 0

# for alpha in [2, 5, 10, 30, 50]:
for alpha in [2, 5, 10, 20]:
    S = content_wmf.linear_surplus_confidence_matrix(train_data, alpha=alpha)

    U, V, vad_ndcg = content_wmf.factorize(S, num_factors, vad_data=vad_data, num_iters=num_iters,
                                           init_std=0.01, lambda_U_reg=lam_theta, lambda_V_reg=lam_beta,
                                           dtype='float32', random_state=98765, verbose=False,
                                           recompute_factors=batched_inv_joblib.recompute_factors_batched,
                                           batch_size=batch_size, n_jobs=n_jobs)
    if vad_ndcg > best_ndcg:
        best_ndcg = vad_ndcg
        U_best = U.copy()
        V_best = V.copy()
        best_alpha = alpha
print best_alpha, best_ndcg


test_data, test_raw = load_data(os.path.join(DATA_DIR, 'test.csv'))
# alpha = 10 gives the best validation performance
print 'Test Recall@10: %.4f' % rec_eval.recall_at_k(train_data, test_data, U_best, V_best, k=10, vad_data=vad_data)
print 'Test Recall@10: %.4f' % rec_eval.recall_at_k(train_data, test_data, U_best, V_best, k=10, vad_data=vad_data)
print 'Test NDCG@10: %.4f' % rec_eval.normalized_dcg_at_k(train_data, test_data, U_best, V_best, k=10, vad_data=vad_data)
print 'Test MAP@10: %.4f' % rec_eval.map_at_k(train_data, test_data, U_best, V_best, k=10, vad_data=vad_data)
