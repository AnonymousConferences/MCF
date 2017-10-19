import math
import bottleneck as bn
import numpy as np
import pandas as pd
from multiprocessing import Queue
import multiprocessing
from scipy import sparse
import os
import glob
"""
All the data should be in the shape of (n_users, n_items)
All the latent factors should in the shape of (n_users/n_items, n_components)
1. train_data refers to the data that was used to train the model
2. heldout_data refers to the data that was used for evaluation (could be test
set or validation set)
3. vad_data refers to the data that should be excluded as validation set, which
should only be used when calculating test scores
"""

# max_item_backed_train = [] #index is the user_id, value is the max project_id
# max_item_backed_vad = [] #index is the user_id, value is the max project_id
#
# def init_global_vars(nusers):
#     global max_item_backed_train
#     global max_item_backed_vad
#     max_item_backed_train = np.zeros(nusers, dtype=int)
#     max_item_backed_vad = np.zeros(nusers, dtype=int)
#
# def prepare_global_dict(train_data, vad_data):
#     global max_item_backed_train
#     global max_item_backed_vad
#     nrow, ncol = train_data.shape
#     for i in range(0, nrow):
#         max_item_backed_train[i] = np.max(train_data[i].nonzero())
#         max_item_backed_vad[i]   = np.max(vad_data[i].nonzero())
project_ts_df = None
def clear_invalid_project(train_data, vad_data, X_pred, lo, hi):
    for i, ui in enumerate(xrange(lo, hi)):
        tmp = project_ts_df.copy()
        invalid_projs = []
        train_backed_proj = train_data[ui].nonzero()[1]
        max_backed_ts_train = np.max(project_ts_df.loc[train_backed_proj, 'timestamp'])
        if vad_data != None:
            vad_backed_proj = vad_data[ui].nonzero()[1]
            max_backed_ts_vad = np.max(project_ts_df.loc[vad_backed_proj, 'timestamp'])
            max_backed_ts_train = max(max_backed_ts_train, max_backed_ts_vad)
        tmp['is_valid'] = tmp['timestamp_end'] > max_backed_ts_train
        invalid_projs.extend(tmp[tmp.is_valid == False].pid)
        invalid_projs.extend(train_backed_proj)
        X_pred[i, invalid_projs] = 0.0
    return X_pred

def single_recall_at_k(X_true_binary, X_pred, k):
    idx = bn.argpartition(-X_pred, k) #find the partition (indexes) so that the first k elements are smallest.
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[idx[:k]] = True
    # print X_pred_binary
    #clear the unrecommended projects
    X_pred_binary[X_pred <= 0] = False
    # print X_pred_binary
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall
def single_map_at_k(X_true, X_pred, k):
    idx_topk_part = bn.argpartition(-X_pred, k)
    topk_part = X_pred[idx_topk_part[:k]]
    idx_part = np.argsort(-topk_part)
    idx_topk = idx_topk_part[idx_part]
    if len(X_true) > 0:
        return apk(X_true, idx_topk, k =k)
    else:
        return np.nan
def single_ndcg_at_k(X_true, X_pred, k):
    #we could do np.argsort(X_true_binary) to then take top k to get k largest but it is really slow.
    idx_topk_part = bn.argpartition(-X_pred, k)  # find the partition (indexes) so that the first k elements are smallest.
                                            # order is not ensured in the partition
    topk_part = X_pred[idx_topk_part[:k]] #take the values of top k
    idx_part = np.argsort(-topk_part) #sort the values of top k, return the index in topk_part
    idx_topk = idx_topk_part[idx_part]
    # if X_pred[idx_topk[-1]] == 0:
    #     #remove items that are not recommended but appear in the list
    #     last_item = len(idx_topk) - 1
    #     for j in range(len(idx_topk) - 2, -1, -1):
    #         if X_pred[idx_topk[j]] == 0:
    #             last_item = j
    #         else:
    #             break
    #     idx_topk = idx_topk[:last_item]
    # print idx_topk
    tp = 1. / np.log2(np.arange(2, k + 2)) #discount function
    # print idx_topk
    # print X_true
    # print X_true[idx_topk]
    # print tp*X_true[idx_topk]

    DCG = (tp*X_true[idx_topk]).sum()
    #count number of backed projects for the users: count the number of non-zeros:
    n = np.sum(X_true[X_true > 0])
    IDCG = (tp[:min(n, k)]).sum()
    # if IDCG <= 0:
    #     return 0.0
    # print 'DCG : %.4f, IDCG: %.4f'%(DCG, IDCG)
    # print '\n'
    return DCG/IDCG
def batch_ndcg_at_k(heldout_batch, X_pred, lo, hi, k):
    idx_topk_part = bn.argpartition(-X_pred, k, axis = 1)
    topk_part = X_pred[np.arange(hi - lo)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(hi - lo)[:, np.newaxis], idx_part]
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(hi - lo)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    # print topk_part[6]
    # print X_pred[6, idx_topk[6]]
    # print 'my DCG: \n',DCG
    # print '\n'
    return DCG / IDCG
def batch_map_at_k(heldout_batch, X_pred, lo, hi, k):
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(hi - lo)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    idx_topk = idx_topk_part[np.arange(hi - lo)[:, np.newaxis], idx_part]
    aps = np.zeros(hi - lo)
    for i, idx in enumerate(xrange(lo, hi)):
        actual = heldout_batch[i].nonzero()[1]
        if len(actual) > 0:
            predicted = idx_topk[i]
            # print 'actual:',actual
            # print predicted
            # print '\n'
            aps[i] = apk(actual, predicted, k=k)
        else:
            aps[i] = np.nan
    return aps
def batch_recall_at_k(heldout_batch, X_pred, lo, hi, k):
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(hi - lo)[:, np.newaxis], idx[:, :k]] = True
    X_pred_binary[X_pred <= 0] = False # add this line is important?

    X_true_binary = heldout_batch
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall
def rank_func(pro_df, weight = 2.0, to=86400.0, num_windows = 100):
    score_from = range(0,num_windows,1)
    for i in range(0,num_windows,1): score_from[i] = score_from[i]/float(num_windows)
    score_to = score_from[1:] + [1.0]
    for lo, hi in zip(score_from, score_to):
        if len(pro_df[pro_df['alpha-dot-beta'] >= lo].pid.values) <= 0:
            break
        pro_df['ranking_score'] = pro_df.apply(lambda row:( row['is_valid']*(row['alpha-dot-beta'] + weight*math.exp(-row['delta_t']/to))
                                                        if row['alpha-dot-beta'] >= lo and row['alpha-dot-beta'] <= hi
                                                        else row['ranking_score']),
                                           axis = 1)
        # print pro_df
    return pro_df['ranking_score'].values
    # return pro_df['alpha-dot-beta'].values
def extract_prediction_res(model_temp_path, lo, hi, project_ts_df, train_data,
                          U, V, vad_data = None, threshold = 0.5, weight=2.0, to = 86400.0, num_windows = 100):
    pred_file = os.path.join(model_temp_path, 'X_pred_lo_%d_hi_%d_threshold_%.2f_weight_%.2f_to_%d.npz'
                             %(lo, hi, threshold, weight, to))
    if os.path.isfile(pred_file):
        X_ranked_pred = np.load(pred_file)['X_ranked_pred']
        return X_ranked_pred
    else:
        n_users = U.shape[0]
        n_projects = V.shape[0]
        X_pred = U[lo:hi].dot(V.T)
        # X_pred[X_pred <= threshold] = 0.0

        # index of items appeared in train and vad data
        item_idx = np.zeros((hi - lo, n_projects), dtype=bool)
        item_idx[train_data[lo:hi].nonzero()] = True
        if vad_data is not None:
            item_idx[vad_data[lo:hi].nonzero()] = True
        # clear them in test data.
        X_pred[item_idx] = 0.0

        X_ranked_pred = np.zeros((hi - lo, n_projects), dtype=np.float32)


        # clear invalid projects (.e.g. ended projects)
        for i, ui in enumerate(xrange(lo, hi)):
            tmp = project_ts_df.copy()
            valid = np.ones(n_projects, dtype=np.int8)
            if vad_data is not None:
                vad_backed_proj = vad_data[ui].nonzero()[1]
                max_backed_ts_vad = np.max(project_ts_df.loc[vad_backed_proj, 'timestamp'])
                valid[vad_backed_proj] = 0.0
            train_backed_proj = train_data[ui].nonzero()[1]
            valid[train_backed_proj] = 0.0
            max_backed_ts_train = np.max(project_ts_df.loc[train_backed_proj, 'timestamp'])
            # lst_ts_thresholds.append(max_backed_ts)
            # clear passed projects
            # invalid_projs = project_ts_df[project_ts_df.timestamp_end < max_backed_ts]['pid'].values
            # X_pred[i, invalid_projs] = -np.inf
            if (np.isfinite(max_backed_ts_vad)):
                max_backed_ts = max(max_backed_ts_vad, max_backed_ts_train)
            else:
                max_backed_ts = max_backed_ts_train
            tmp['t_timestamp'] = max_backed_ts - tmp.timestamp
            tmp['t_timestamp_end'] = tmp.timestamp_end - max_backed_ts
            # print tmp
            tmp['is_valid'] = (tmp['t_timestamp_end'] > 0).values
            # tmp['step_func_val'] = np.logical_and((tmp['t_timestamp_end'] > 0).values, X_pred[i,] > threshold)
            # tmp['step_func_val'] = X_pred[i,] > threshold

            tmp['delta_t'] = tmp[['t_timestamp_end', 't_timestamp']].abs().min(axis=1)
            tmp['alpha-dot-beta'] = X_pred[i]
            tmp['ranking_score'] = tmp['alpha-dot-beta'] ######## ADDDDDD NEWWWWWWW

            ranking_lst = rank_func(tmp, weight, to, num_windows)

            X_ranked_pred[i] = ranking_lst
            # print heldout_data_df
            # print heldout_data_df[heldout_data_df.bid == ui]
            # true_backed_pid =  heldout_data_df[heldout_data_df.bid == ui].pid.values #get backed projects in test data
            # X_true_ranked = np.zeros(n_projects, dtype=np.float32)
            # n_rec_projs = len(true_backed_pid)
            # X_true_ranked[true_backed_pid[:k]] = 1. / np.log2(np.arange(2, min(k + 2, n_rec_projs)))
            # if (ui == 8):
            #     print X_pred[i][53]
            #     print heldout_data[ui].toarray()[0][53]

            #store to file for the next use.
        np.savez(pred_file, X_ranked_pred = X_ranked_pred)
        return X_ranked_pred

def extract_prediction_res2(model_temp_path, lo, hi, project_ts_df, train_data,
                          U, V, vad_data = None, threshold = 0.5, weight=2.0, to = 86400.0, num_windows = 100):
    pred_file = os.path.join(model_temp_path, 'X_pred_lo_%d_hi_%d_threshold_%.2f_weight_%.2f_to_%d.npz'
                             %(lo, hi, threshold, weight, to))
    if os.path.isfile(pred_file):
        X_ranked_pred = np.load(pred_file)['X_ranked_pred']
        return X_ranked_pred
    else:
        X_ranked_pred = _make_prediction(train_data, U, V, slice(lo,hi), hi-lo, mu=None, vad_data=vad_data)

        for i, ui in enumerate(xrange(lo, hi)):
            tmp = project_ts_df.copy()
            invalid_projs = []

            train_backed_proj = train_data[ui].nonzero()[1]
            invalid_projs.extend(train_backed_proj)
            max_backed_ts_train = np.max(tmp.loc[train_backed_proj, 'timestamp'])

            max_backed_ts_vad = 0
            if vad_data != None:
                vad_backed_proj = vad_data[ui].nonzero()[1]
                max_backed_ts_vad = np.max(tmp.loc[vad_backed_proj, 'timestamp'])
                invalid_projs.extend(vad_backed_proj)
            max_backed_ts = max(max_backed_ts_train, max_backed_ts_vad)
            tmp['is_valid'] = tmp['timestamp_end'] > max_backed_ts
            invalid_projs.extend(tmp[tmp.is_valid == False].pid)

            # clear invalid projects (.e.g. ended projects)
            X_ranked_pred[i, invalid_projs] = 0.0

            tmp['t_timestamp'] = max_backed_ts - tmp.timestamp
            tmp['t_timestamp_end'] = tmp.timestamp_end - max_backed_ts
            # print tmp
            # tmp['is_valid'] = (tmp['t_timestamp_end'] > 0).values

            # tmp['step_func_val'] = np.logical_and((tmp['t_timestamp_end'] > 0).values, X_pred[i,] > threshold)
            # tmp['step_func_val'] = X_pred[i,] > threshold

            tmp['delta_t'] = tmp[['t_timestamp_end', 't_timestamp']].abs().min(axis=1)
            tmp['alpha-dot-beta'] = X_ranked_pred[i]
            tmp['ranking_score'] = tmp['alpha-dot-beta'] ######## ADDDDDD NEWWWWWWW

            ranking_lst = rank_func(tmp, weight, to, num_windows)
            X_ranked_pred[i] = ranking_lst

        np.savez(pred_file, X_ranked_pred = X_ranked_pred)
        return X_ranked_pred

def make_prediction_batch(model_temp_path, lo, hi, project_ts_df, train_data, heldout_data, heldout_data_df,
                          U, V, vad_data = None, threshold = 0.5, k = 10, weight=2.0, to = 86400.0):
    recall_batch = np.zeros(hi - lo, dtype=np.float32)
    # ndcg_batch = np.zeros(hi-lo, dtype=np.float32)
    # map_batch = np.zeros(hi - lo, dtype=np.float32)
    ranked_prediction_path = os.path.join(model_temp_path,'ranked_prediction')
    if not os.path.exists(ranked_prediction_path):
        os.makedirs(ranked_prediction_path)
    X_ranked_pred = extract_prediction_res(ranked_prediction_path, lo, hi, project_ts_df, train_data,
                                           U, V, vad_data, threshold, weight, to)
    # for i, ui in enumerate(xrange(lo, hi)):
    #     ranking_lst = X_ranked_pred[i]
        #computer rec metric at k
        # recall_score = single_recall_at_k((heldout_data[ui] > 0).toarray(), ranking_lst, k)
        # recall_batch[i]= recall_score
        # ndcg_score = single_ndcg_at_k(heldout_data[ui].toarray()[0],ranking_lst , k)
        # ndcg_batch[i] = ndcg_score
        # map_score = single_map_at_k(heldout_data[ui].nonzero()[1], ranking_lst, k)
        # map_batch[i] = map_score
    recall_batch = batch_recall_at_k(heldout_batch=(heldout_data[slice(lo,hi)] > 0).toarray(),
                                     X_pred = X_ranked_pred, lo=lo, hi=hi, k = k)
    ndcg_batch = batch_ndcg_at_k(heldout_batch=heldout_data[slice(lo,hi)],
                                 X_pred = X_ranked_pred, lo=lo, hi=hi, k = k)
    map_batch = batch_map_at_k(heldout_batch=heldout_data[slice(lo, hi)],
                               X_pred=X_ranked_pred, lo=lo, hi=hi, k=k)

    #we can put result into a queue but since the result is large,
    #the message returned by the process got error --> need to store in files.
    #following are commands to store the result into queue but we don't need them anymore
    # out_q.put((lo, hi, recall_batch, ndcg_batch, map_batch))
    # out_q.put((lo, hi, ndcg_batch))
    # out_q.put((lo, hi, map_batch))

    #store to files:
    recall_topk_res_path = os.path.join(model_temp_path, 'recall_topk')
    ndcg_topk_res_path = os.path.join(model_temp_path, 'ndcg_topk')
    map_topk_res_path = os.path.join(model_temp_path, 'map_topk')
    recall_topk_res_file = os.path.join(recall_topk_res_path,
                                        'recall-topk_lo_%d_hi_%d_threshold_%.2f_weight_%.2f_to_%d.npz'
                                        %(lo, hi, threshold, weight, to) )
    ndcg_topk_res_file = os.path.join(ndcg_topk_res_path,
                                        'ndcg-topk_lo_%d_hi_%d_threshold_%.2f_weight_%.2f_to_%d.npz'
                                        % (lo, hi, threshold, weight, to))
    map_topk_res_file = os.path.join(map_topk_res_path,
                                      'map-topk_lo_%d_hi_%d_threshold_%.2f_weight_%.2f_to_%d.npz'
                                      % (lo, hi, threshold, weight, to))
    np.savez(recall_topk_res_file, recall_batch = recall_batch)
    np.savez(ndcg_topk_res_file, ndcg_batch=ndcg_batch)
    np.savez(map_topk_res_file, map_batch=map_batch)
def make_prediction_batch2(model_temp_path, lo, hi, project_ts_df, train_data, heldout_data, heldout_data_df,
                           U, V, vad_data = None, threshold = 0.5, k = 10, weight=2.0, to = 86400.0,
                           batch_size = 2000, num_windows = 100):

    recall_batch = np.zeros(hi - lo, dtype=np.float32)
    ndcg_batch = np.zeros(hi-lo, dtype=np.float32)
    map_batch = np.zeros(hi - lo, dtype=np.float32)

    ranked_prediction_path = os.path.join(model_temp_path, 'ranked_prediction')
    if not os.path.exists(ranked_prediction_path):
        os.makedirs(ranked_prediction_path)
    start_idx = range(lo, hi, batch_size)
    end_idx = start_idx[1:] + [hi]

    for local_lo, local_hi in zip(start_idx, end_idx):
        X_ranked_pred = extract_prediction_res(ranked_prediction_path, local_lo, local_hi, project_ts_df,
                                               train_data, U, V, vad_data, threshold, weight, to, num_windows)

        recall_batch[local_lo-lo:local_hi-lo] = batch_recall_at_k(heldout_batch=(heldout_data[slice(local_lo,local_hi)] > 0).toarray(),
                                         X_pred = X_ranked_pred, lo=local_lo, hi=local_hi, k = k)
        ndcg_batch[local_lo-lo:local_hi-lo] = batch_ndcg_at_k(heldout_batch=heldout_data[slice(local_lo,local_hi)],
                                         X_pred = X_ranked_pred, lo=local_lo, hi=local_hi, k = k)
        map_batch[local_lo-lo:local_hi-lo] = batch_map_at_k(heldout_batch=heldout_data[slice(local_lo, local_hi)],
                                         X_pred=X_ranked_pred, lo=local_lo, hi=local_hi, k=k)

    #store to files:
    recall_topk_res_path = os.path.join(model_temp_path, 'recall_topk')
    ndcg_topk_res_path = os.path.join(model_temp_path, 'ndcg_topk')
    map_topk_res_path = os.path.join(model_temp_path, 'map_topk')
    recall_topk_res_file = os.path.join(recall_topk_res_path,
                                        'recall-topk_lo_%d_hi_%d_threshold_%.2f_weight_%.2f_to_%d.npz'
                                        %(lo, hi, threshold, weight, to) )
    ndcg_topk_res_file = os.path.join(ndcg_topk_res_path,
                                        'ndcg-topk_lo_%d_hi_%d_threshold_%.2f_weight_%.2f_to_%d.npz'
                                        % (lo, hi, threshold, weight, to))
    map_topk_res_file = os.path.join(map_topk_res_path,
                                      'map-topk_lo_%d_hi_%d_threshold_%.2f_weight_%.2f_to_%d.npz'
                                      % (lo, hi, threshold, weight, to))
    np.savez(recall_topk_res_file, recall_batch = recall_batch)
    np.savez(ndcg_topk_res_file, ndcg_batch=ndcg_batch)
    np.savez(map_topk_res_file, map_batch=map_batch)
def merging_file_res(n_users, path, param_name):
    res = np.zeros(n_users, dtype=np.float32)
    for f in glob.glob(os.path.join(path, '*.npz')):
        batch = np.load(f)[param_name]
        filename = f.split('/')[-1]
        tokens = filename.split('_')
        lo = int(tokens[2])
        hi = int(tokens[4])
        res[lo:hi] = batch
    return res
def evaluate(model_temp_path, project_ts_df, train_data, heldout_data, heldout_data_df, U, V, vad_data = None,
             n_jobs = 8, threshold=0.5, k= 10, weight=2.0, to=86400.0, batch_size = 2000, num_windows = 100):
    #init some temp folder to store the result:
    recall_topk_res_path = os.path.join(model_temp_path, 'recall_topk')
    ndcg_topk_res_path = os.path.join(model_temp_path, 'ndcg_topk')
    map_topk_res_path = os.path.join(model_temp_path, 'map_topk')

    if not os.path.exists(recall_topk_res_path):
        os.makedirs(recall_topk_res_path)
    else:
        for f in glob.glob(os.path.join(recall_topk_res_path, '*.npz')):
            os.remove(f)
    if not os.path.exists(ndcg_topk_res_path):
        os.makedirs(ndcg_topk_res_path)
    else:
        for f in glob.glob(os.path.join(ndcg_topk_res_path, '*.npz')):
            os.remove(f)
    if not os.path.exists(map_topk_res_path):
        os.makedirs(map_topk_res_path)
    else:
        for f in glob.glob(os.path.join(map_topk_res_path, '*.npz')):
            os.remove(f)

    ranked_prediction_path = os.path.join(model_temp_path, 'ranked_prediction')
    if not os.path.exists(ranked_prediction_path):
        os.makedirs(ranked_prediction_path)
    ######################################################################

    n_users = U.shape[0]
    n_projects = V.shape[0]
    batch_users = n_users/n_jobs
    start_idx = range(0, n_users, batch_users)
    end_idx = start_idx[1:] + [n_users]
    # out_q = Queue()
    procs = []

    # recall_at_k_all= np.zeros(n_users, dtype=np.float32)
    # ndcg_at_k_all = np.zeros(n_users, dtype=np.float32)
    # map_at_k_all = np.zeros(n_users, dtype=np.float32)
    for lo, hi in zip(start_idx, end_idx):
        p = multiprocessing.Process(
            target=make_prediction_batch2,
            args=(model_temp_path, lo, hi, project_ts_df, train_data, heldout_data, heldout_data_df,
                  U, V, vad_data, threshold, k, weight, to, batch_size, num_windows)
            # args=(model_temp_path, lo, hi, project_ts_df, train_data, heldout_data, heldout_data_df,
            #       U, V, vad_data, threshold, k, weight, to)

            # args= (model_temp_path, out_q, lo, hi, project_ts_df, train_data, heldout_data, heldout_data_df,
            #         U, V, vad_data, threshold, k, weight, to)
        )
        p.start()
        procs.append(p)
    # for i in range(n_jobs):
    #     [lo, hi, partly_recall, partly_ndcg, partly_map] = out_q.get()
    #     recall_at_k_all[lo:hi] = partly_recall
    #     ndcg_at_k_all[lo:hi] =  partly_ndcg
    #     map_at_k_all[lo:hi] = partly_map

        # [lo, hi, partly_ndcg] = out_q.get()
        # ndcg_at_k_all[lo:hi] =  partly_ndcg

        # [lo, hi, partly_map] = out_q.get()
        # map_at_k_all[lo:hi] =  partly_map
    for p in procs:
        p.join()

    #merging file in here:
    recall_at_k_all = merging_file_res(n_users=n_users, path=recall_topk_res_path, param_name='recall_batch')
    ndcg_at_k_all = merging_file_res(n_users=n_users, path=ndcg_topk_res_path, param_name='ndcg_batch')
    map_at_k_all = merging_file_res(n_users=n_users, path=map_topk_res_path, param_name='map_batch')

    return (np.nanmean(recall_at_k_all), np.nanmean(ndcg_at_k_all), np.nanmean(map_at_k_all))
def prec_at_k(train_data, heldout_data, U, V, batch_users=5000, k=20,
              mu=None, vad_data=None, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(precision_at_k_batch(train_data, heldout_data,
                                        U, V.T, user_idx, k=k,
                                        mu=mu, vad_data=vad_data))
    mn_prec = np.hstack(res)
    if callable(agg):
        return agg(mn_prec)
    return mn_prec


def recall_at_k(train_data, heldout_data, U, V, batch_users=5000, k=20,
                mu=None, vad_data=None, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(recall_at_k_batch(train_data, heldout_data,
                                     U, V.T, user_idx, k=k,
                                     mu=mu, vad_data=vad_data))
    mn_recall = np.hstack(res)
    if callable(agg):
        return agg(mn_recall)
    return mn_recall


def ric_rank_at_k(train_data, heldout_data, U, V, batch_users=5000, k=5,
                  mu=None, vad_data=None):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(mean_rrank_at_k_batch(train_data, heldout_data,
                                         U, V.T, user_idx, k=k,
                                         mu=mu, vad_data=vad_data))
    mrrank = np.hstack(res)
    return mrrank[mrrank > 0].mean()


def mean_perc_rank(train_data, heldout_data, U, V, batch_users=5000,
                   mu=None, vad_data=None):
    n_users = train_data.shape[0]
    mpr = 0
    for user_idx in user_idx_generator(n_users, batch_users):
        mpr += mean_perc_rank_batch(train_data, heldout_data, U, V.T, user_idx,
                                    mu=mu, vad_data=vad_data)
    mpr /= heldout_data.sum()
    return mpr


def normalized_dcg(train_data, heldout_data, U, V, batch_users=5000,
                   mu=None, vad_data=None, agg=np.nanmean):
    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(NDCG_binary_batch(train_data, heldout_data, U, V.T,
                                     user_idx, mu=mu, vad_data=vad_data))
    ndcg = np.hstack(res)
    if callable(agg):
        return agg(ndcg)
    return ndcg


def normalized_dcg_at_k(train_data, heldout_data, U, V, batch_users=12,
                        k=100, mu=None, vad_data=None, agg=np.nanmean):

    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(NDCG_binary_at_k_batch(train_data, heldout_data, U, V.T,
                                          user_idx, k=k, mu=mu,
                                          vad_data=vad_data))
    ndcg = np.hstack(res)
    # print 'Final ',ndcg
    if callable(agg):
        return agg(ndcg)
    return ndcg


def map_at_k(train_data, heldout_data, U, V, batch_users=5000, k=100, mu=None,
             vad_data=None, agg=np.nanmean):

    n_users = train_data.shape[0]
    res = list()
    for user_idx in user_idx_generator(n_users, batch_users):
        res.append(MAP_at_k_batch(train_data, heldout_data, U, V.T, user_idx,
                                  k=k, mu=mu, vad_data=vad_data))
    map = np.hstack(res)
    if callable(agg):
        return agg(map)
    return map


# helper functions #

def user_idx_generator(n_users, batch_users):
    ''' helper function to generate the user index to loop through the dataset
    '''
    for start in xrange(0, n_users, batch_users):
        end = min(n_users, start + batch_users)
        yield slice(start, end)


def _make_prediction(train_data, Et, Eb, user_idx, batch_users, mu=None,
                     vad_data=None):
    n_songs = train_data.shape[1]
    # exclude examples from training and validation (if any)
    item_idx = np.zeros((batch_users, n_songs), dtype=bool)
    item_idx[train_data[user_idx].nonzero()] = True
    # print user_idx
    # print train_data[user_idx].nonzero()
    if vad_data is not None:
        item_idx[vad_data[user_idx].nonzero()] = True
    X_pred = Et[user_idx].dot(Eb)
    if mu is not None:
        if isinstance(mu, np.ndarray):
            assert mu.size == n_songs  # mu_i
            X_pred *= mu
        elif isinstance(mu, dict):  # func(mu_ui)
            params, func = mu['params'], mu['func']
            args = [params[0][user_idx], params[1]]
            if len(params) > 2:  # for bias term in document or length-scale
                args += [params[2][user_idx]]
            if not callable(func):
                raise TypeError("expecting a callable function")
            X_pred *= func(*args)
        else:
            raise ValueError("unsupported mu type")
    X_pred[item_idx] = -np.inf
    #remove passed projects here.
    #adding timedecay function in here., maybe change X_pred = Et[user_idx].dot(Eb)?
    return X_pred


def precision_at_k_batch(train_data, heldout_data, Et, Eb, user_idx,
                         k=20, normalize=True, mu=None, vad_data=None):
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_data[user_idx] > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)

    if normalize:
        precision = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    else:
        precision = tmp / k
    return precision


def recall_at_k_batch(train_data, heldout_data, Et, Eb, user_idx,
                      k=20, normalize=True, mu=None, vad_data=None):
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_data[user_idx] > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


def mean_rrank_at_k_batch(train_data, heldout_data, Et, Eb,
                          user_idx, k=5, mu=None, vad_data=None):
    '''
    mean reciprocal rank@k: For each user, make predictions and rank for
    all the items. Then calculate the mean reciprocal rank for the top K that
    are in the held-out set.
    '''
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    all_rrank = 1. / (np.argsort(np.argsort(-X_pred, axis=1), axis=1) + 1)
    X_true_binary = (heldout_data[user_idx] > 0).toarray()

    heldout_rrank = X_true_binary * all_rrank
    top_k = bn.partition(-heldout_rrank, k, axis=1)
    return -top_k[:, :k].mean(axis=1)


def NDCG_binary_batch(train_data, heldout_data, Et, Eb, user_idx,
                      mu=None, vad_data=None):
    '''
    normalized discounted cumulative gain for binary relevance
    '''
    batch_users = user_idx.stop - user_idx.start
    n_items = train_data.shape[1]

    X_pred = _make_prediction(train_data, Et, Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    X_pred[X_pred <= 0] = 0
    all_rank = np.argsort(np.argsort(-X_pred, axis=1), axis=1)
    # build the discount template
    tp = 1. / np.log2(np.arange(2, n_items + 2))
    all_disc = tp[all_rank]

    X_true_binary = (heldout_data[user_idx] > 0).tocoo()
    disc = sparse.csr_matrix((all_disc[X_true_binary.row, X_true_binary.col],
                              (X_true_binary.row, X_true_binary.col)),
                             shape=all_disc.shape)
    DCG = np.array(disc.sum(axis=1)).ravel()
    IDCG = np.array([tp[:n].sum()
                     for n in heldout_data[user_idx].getnnz(axis=1)])

    return DCG / IDCG


def NDCG_binary_at_k_batch(train_data, heldout_data, Et, Eb, user_idx,
                           mu=None, k=100, vad_data=None):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx,
                              batch_users, mu=mu, vad_data=vad_data)
    X_pred[X_pred <=0] = 0

    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    heldout_batch = heldout_data[user_idx]
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    # print idx_topk
    # print 'DCG : %.4f, IDCG: %.4f' % (DCG, IDCG)
    # print topk_part[6]
    # print X_pred[6,idx_topk[6]]
    # print 'DCG rec eval:\n ', DCG
    # print '\n'
    return DCG / IDCG


def MAP_at_k_batch(train_data, heldout_data, Et, Eb, user_idx, mu=None, k=100,
                   vad_data=None):
    '''
    mean average precision@k
    '''
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx, batch_users, mu=mu,
                              vad_data=vad_data)
    X_pred[X_pred <= 0] = 0
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    aps = np.zeros(batch_users)
    for i, idx in enumerate(xrange(user_idx.start, user_idx.stop)):
        actual = heldout_data[idx].nonzero()[1]
        if len(actual) > 0:
            predicted = idx_topk[i]
            # print 'actual:',actual
            # print predicted
            # print '\n'
            aps[i] = apk(actual, predicted, k=k)
        else:
            aps[i] = np.nan
    return aps


def mean_perc_rank_batch(train_data, heldout_data, Et, Eb, user_idx,
                         mu=None, vad_data=None):
    '''
    mean percentile rank for a batch of users
    MPR of the full set is the sum of batch MPR's divided by the sum of all the
    feedbacks. (Eq. 8 in Hu et al.)
    This metric not necessarily constrains the data to be binary
    '''
    batch_users = user_idx.stop - user_idx.start

    X_pred = _make_prediction(train_data, Et, Eb, user_idx, batch_users,
                              mu=mu, vad_data=vad_data)
    all_perc = np.argsort(np.argsort(-X_pred, axis=1), axis=1) / \
        np.isfinite(X_pred).sum(axis=1, keepdims=True).astype(np.float32)
    perc_batch = (all_perc[heldout_data[user_idx].nonzero()] *
                  heldout_data[user_idx].data).sum()
    return perc_batch


## steal from https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py
def apk(actual, predicted, k=100):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual: #and p not in predicted[:i]: # not necessary for us since we will not make duplicated recs
            num_hits += 1.0
            score += num_hits / (i+1.0)

    # we handle this part before making the function call
    #if not actual:
    #    return np.nan

    return score / min(len(actual), k)
