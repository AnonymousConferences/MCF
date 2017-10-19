import numpy as np
import os
import pandas as pd
import time
import glob
import rec_eval as rec_eval
import ranked_rec_eval as ranked_eval
import ranked_rec_eval_range as ranked_eval2
from scipy import sparse
DEBUG_MODE = True
DATA_DIR = 'data/rec_data/all'
model_names = []
# model_names.append('Model2_K100_0.0287.npz')
# model_names.append('Model3_K100_0.0259.npz')
# model_names.append('Baseline2_Cofactor_K100.npz')
# model_names.append('Baseline1_MF_K100.npz')
# model_names.append('Model2_K100_0.0263.npz')
# model_names.append('Model2_K100_0.03.npz')

model_names.append('Baseline1_MF_K100_recall100_0.2575_ndcg100_0.1172_map100_0.0444.npz')
model_names.append('Cofactor_recall100_0.2607_ndcg100_0.1202_map100_0.0464.npz')

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
# train_data, train_raw, train_df = load_data(os.path.join(DATA_DIR, 'train.csv'))
# project_ts_df_train = train_df[['pid', 'timestamp', 'timestamp_end']]
# # project_ts_df_train = project_ts_df_train.drop_duplicates(cols='pid').sort_index(by='pid', ascending=True)
# project_ts_df_train = project_ts_df_train.drop_duplicates('pid').sort_index(by='pid', ascending=True)
#
#
# vad_data, vad_raw, vad_df = load_data(os.path.join(DATA_DIR, 'validation.csv'))
# project_ts_df_vad = vad_df[['pid', 'timestamp', 'timestamp_end']]
# # project_ts_df_vad = project_ts_df_vad.drop_duplicates(cols='pid').sort_index(by='pid', ascending=True)
# project_ts_df_vad = project_ts_df_vad.drop_duplicates('pid').sort_index(by='pid', ascending=True)
#
# test_data, test_raw, test_df = load_data(os.path.join(DATA_DIR, 'test.csv'))
# project_ts_df_test = test_df[['pid', 'timestamp', 'timestamp_end']]
# # project_ts_df_test = project_ts_df_test.drop_duplicates(cols='pid').sort_index(by='pid', ascending=True)
# project_ts_df_test = project_ts_df_test.drop_duplicates('pid').sort_index(by='pid', ascending=True)
# test_data.data = np.ones_like(test_data.data)
#
# project_ts_df = pd.concat([project_ts_df_train, project_ts_df_test, project_ts_df_vad])
# # project_ts_df = project_ts_df.drop_duplicates(cols='pid').sort_index(by='pid', ascending=True)
# project_ts_df = project_ts_df.drop_duplicates('pid').sort_index(by='pid', ascending=True)
# project_ts_df_train = None
# project_ts_df_vad = None
# project_ts_df_test = None
# project_ts_df = project_ts_df.reset_index(drop=True)
# # print project_ts_df

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


#create a folder to store the temp-file:
if not os.path.exists('temp-files'):
    os.makedirs('temp-files')

total_t1=time.time()
topk_range = [10,20,30,100]
for model_name in model_names:
    t1 = time.time()

    # create a temp folder for the model:
    model_temp_path = os.path.join('temp-files', model_name)
    if not os.path.exists(model_temp_path):
        os.makedirs(model_temp_path)

    # clear the ranking prediction
    ranked_prediction_path = os.path.join(model_temp_path, 'ranked_prediction')
    print 'clearing ', ranked_prediction_path
    if os.path.exists(ranked_prediction_path):
        for f in glob.glob(os.path.join(ranked_prediction_path, '*.npz')):
            os.remove(f)

    params = np.load(model_name)
    U, V = params['U'], params['V']
    # print U.shape
    # print V.shape
    print '****************************'
    print model_name

    print 'Original ranking:'
    for K in topk_range:
        print 'Test Recall@%d: %.4f'%(K, rec_eval.parallel_recall_at_k(train_data, test_data, U, V, k=K,
                                                                   vad_data=vad_data, n_jobs=1, clear_invalid=True))
        print 'Test NDCG@%d: %.4f'%(K, rec_eval.parallel_normalized_dcg_at_k(train_data, test_data, U, V, k=K,
                                                                         vad_data=vad_data, n_jobs=1, clear_invalid=True))
        print 'Test MAP@%d: %.4f'%(K, rec_eval.parallel_map_at_k(train_data, test_data, U, V, k=K,
                                                             vad_data=vad_data, n_jobs=1, clear_invalid=True))

    print 'After applying ranking function'
    for threshold in [0.1]:
        for weight in [0.1]:
            for num_windows  in [10,100]:
                print 'threshold : %.5f , weight: %.5f, num_windows: %d' % (threshold, weight, num_windows)
                for K in topk_range:
                    res = ranked_eval2.evaluate(model_temp_path, project_ts_df, train_data, test_data, test_df, U, V,
                                               vad_data=vad_data, n_jobs=1, threshold=threshold, k=K, weight=weight,
                                               to = 86400.0, batch_size = 2000, num_windows=100)
                    # res = ranked_eval.evaluate(model_temp_path, project_ts_df, train_data, test_data, test_df, U, V,
                    #                            vad_data=vad_data, n_jobs=1, threshold=threshold, k=K, weight=weight,
                    #                            to=86400.0, batch_size=2000)
                    print 'Recal@%d: %.4f'%(K, res[0])
                    print 'NDCG@%d: %.4f'%(K, res[1])
                    print 'MAP@%d: %.4f' % (K, res[2])

                # clear the ranking prediction
                ranked_prediction_path = os.path.join(model_temp_path, 'ranked_prediction')
                print 'clearing ', ranked_prediction_path
                if os.path.exists(ranked_prediction_path):
                    for f in glob.glob(os.path.join(ranked_prediction_path, '*.npz')):
                        os.remove(f)
    t2 = time.time()
    print 'Time : %d seconds or %d minutes'%(t2-t1, float(t2-t1)/60.0)
total_t2=time.time()
print 'Total Time : %d seconds or %d minutes'%(total_t2-total_t1, float(total_t2 - total_t1)/60.0)
