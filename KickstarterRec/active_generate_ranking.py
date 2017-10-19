import numpy as np
import os
import pandas as pd
import time
import glob
import rec_eval_active as rec_eval
# import ranked_rec_eval as rec_eval
from scipy import sparse
from joblib import Parallel, delayed
DEBUG_MODE = False
DATA_DIR = 'data/rec_data/all'
model_names = []
model_names.append('BESTMODEL/Model5_K100_recall100_0.3310_ndcg100_0.1805_map100_0.0843.npz')
#model_names.append('BESTMODEL/Model3_K100_recall100_0.3290_ndcg100_0.1795_map100_0.0840_active_all.npz')
#model_names.append('BESTMODEL/Model2_K100_recall100_0.3214_ndcg100_0.1758_map100_0.0822.npz')
model_names.append('BESTMODEL/Cofactor_recall100_0.3198_ndcg100_0.1743_map100_0.0813.npz')
model_names.append('BESTMODEL/WMF_recall100_0.3194_ndcg100_0.1720_map100_0.0792.npz')

if DEBUG_MODE:
    DATA_DIR = 'data/rec_data/debug'
    model_names = []
    model_names.append('CoFactor_K100_ML20M.npz')
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


vad_data, vad_raw, vad_df = load_data(os.path.join(DATA_DIR, 'validation.csv'))


test_data, test_raw, test_df = load_data(os.path.join(DATA_DIR, 'test.csv'))

if len(glob.glob(os.path.join(DATA_DIR, 'project_ts_df.csv'))) <= 0:
    project_ts_df_train = train_df[['pid', 'timestamp', 'timestamp_end']]
    project_ts_df_train = project_ts_df_train.drop_duplicates('pid').sort_index(by='pid', ascending=True)
    project_ts_df_vad = vad_df[['pid', 'timestamp', 'timestamp_end']]
    project_ts_df_vad = project_ts_df_vad.drop_duplicates('pid').sort_index(by='pid', ascending=True)
    project_ts_df_test = test_df[['pid', 'timestamp', 'timestamp_end']]
    project_ts_df_test = project_ts_df_test.drop_duplicates('pid').sort_index(by='pid', ascending=True)
    test_data.data = np.ones_like(test_data.data)

    project_ts_df = pd.concat([project_ts_df_train, project_ts_df_test, project_ts_df_vad])
    project_ts_df = project_ts_df.drop_duplicates('pid').sort_index(by='pid', ascending=True)
    project_ts_df_train = None
    project_ts_df_vad = None
    project_ts_df_test = None
    project_ts_df = project_ts_df.reset_index(drop=True)
    project_ts_df.to_csv(os.path.join(DATA_DIR, 'project_ts_df.csv'), index=False)
else:
    project_ts_df = pd.read_csv(os.path.join(DATA_DIR, 'project_ts_df.csv'))
rec_eval.project_ts_df = project_ts_df

user_activity = np.asarray(train_data.sum(axis=1)).ravel()

#decide activeness by percentage
sorted_ua = sorted(user_activity)
ACTIVE_THRESHOLD = 0.2
cold_indx = sorted_ua[int(ACTIVE_THRESHOLD*len(sorted_ua)) + 1]
high_active_indx = sorted_ua[int((1.0-ACTIVE_THRESHOLD)*len(sorted_ua)) + 1]
active_condition = [0, cold_indx, high_active_indx]

total_t1=time.time()
topk_range = [5,10,20,50,100]
#topk_range = [50]
#active_condition = [0,25,50]
for model_name in model_names:
    t1 = time.time()
    params = np.load(model_name)
    U, V = params['U'], params['V']
    # print U.shape
    # print V.shape
    print '****************************'
    print model_name

    recall_store = []
    ndcg_store = []
    map_store = []
    for K in topk_range:
        print 'working with top N recommendations N = %d'%K
        recall_all = rec_eval.parallel_recall_at_k(train_data, test_data, U, V, k=K,
                                                   vad_data=vad_data, n_jobs=16,
                                                   clear_invalid=True, call_agg = False)
        ndcg_all = rec_eval.parallel_normalized_dcg_at_k(train_data, test_data, U, V, k=K,
                                                         vad_data=vad_data, n_jobs=16,
                                                         clear_invalid=True, call_agg = False)
        map_all = rec_eval.parallel_map_at_k(train_data, test_data, U, V, k=K,
                                             vad_data=vad_data, n_jobs=16,
                                             clear_invalid=True, call_agg = False)

        for i in range(len(active_condition)):
            active_from = active_condition[i]
            active_to = active_from
            if i == (len(active_condition) - 1):
                idx_active = [j for j, count in enumerate(user_activity) if count >= active_from]
            else:
                active_to = active_condition[i + 1]
                idx_active = [j for j, count in enumerate(user_activity) if count >= active_from and count < active_to]
            res_recall = np.nanmean(recall_all[idx_active])
            res_ndcg = np.nanmean(ndcg_all[idx_active])
            res_map = np.nanmean(map_all[idx_active])
            print 'Range %d to %d, Recall@%d\t%.4f'%(active_from, active_to, K, res_recall)
            print 'Range %d to %d, NDCG@%d\t%.4f' % (active_from, active_to, K, res_ndcg)
            print 'Range %d to %d, MAP@%d\t%.4f' % (active_from, active_to, K, res_map)

total_t2=time.time()
print 'Total Time : %d seconds or %d minutes'%(total_t2-total_t1, float(total_t2 - total_t1)/60.0)
