import numpy as np
import rec_eval as rec_eval
import os
from scipy import sparse
import pandas as pd
import glob
import sys

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

def load_data(csv_file, shape=(n_users, n_projects)):
    tp = pd.read_csv(csv_file)
    timestamps, timestamps_end, rows, cols = np.array(tp['timestamp']), np.array(tp['timestamp_end']), \
                                             np.array(tp['bid']), np.array(tp['pid']) #rows will be user ids, cols will be projects-ids.
    seq = np.concatenate((  rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int'),
                            timestamps[:, None], timestamps_end[:, None]
                          ), axis=1)
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
    return data, seq, tp

train_data, train_raw, train_df = load_data(os.path.join(DATA_DIR, 'train.csv'))
user_activity = np.asarray(train_data.sum(axis=1)).ravel()
numbackers_per_project = np.asarray(train_data.astype('int64').sum(axis=0)).ravel()

vad_data, vad_raw, vad_df = load_data(os.path.join(DATA_DIR, 'validation.csv'))
test_data, test_raw, test_df = load_data(os.path.join(DATA_DIR, 'test.csv'))

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

def local_alone_eval(train_data, test_data, vad_data, U, V):
    recall100 = 0.0
    ndcg100 = 0.0
    map100 = 0.0
    for K in [10, 20, 30, 100]:
        recall_at_K = rec_eval.parallel_recall_at_k(train_data, test_data, U, V, k=K,
                                           vad_data=vad_data, clear_invalid = True, n_jobs=15)
        print 'Test Recall@%d: %.4f' % (K, recall_at_K)
        ndcg_at_K = rec_eval.parallel_normalized_dcg_at_k(train_data, test_data, U, V, k=K,
                                                 vad_data=vad_data, clear_invalid = True, n_jobs=15)
        print 'Test NDCG@%d: %.4f' % (K, ndcg_at_K)
        map_at_K = rec_eval.parallel_map_at_k(train_data, test_data, U, V, k=K,
                                     vad_data=vad_data, clear_invalid = True, n_jobs=15)
        print 'Test MAP@%d: %.4f' % (K, map_at_K)
        if K == 100:
            recall100 = recall_at_K
            ndcg100 = ndcg_at_K
            map100 = map_at_K
    return (recall100, ndcg100, map100)

model_names = []
model_names.append('Baseline1_MF_K100_vad100.npz')
model_names.append('Baseline2_Cofactor_K100_vad100.npz')
model_names.append('Model2_K100_0.0269_vad100.npz')
model_names.append('Model2_K100_ndcg10_0.0269.npz')

model_names.append('Baseline1_MF_K100.npz')
model_names.append('Baseline2_Cofactor_K100.npz')
model_names.append('Model4_K100_0.0260.npz')
model_names.append('Model4_K100_ndcg10_0.0235_vad100.npz')

model_names.append('Model2_K100_0.0263.npz')
model_names.append('Model2_K100_0.0287.npz')

for model in model_names:
    print 'working on model:',model
    params = np.load(model)
    U, V = params['U'], params['V']
    (recall100, ndcg100, map100) = local_alone_eval(train_data, test_data, vad_data, U, V)
