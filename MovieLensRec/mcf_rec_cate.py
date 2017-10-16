import sys

import itertools
import glob
import os
import sys
import mf_pos_embedding_user_project_new as mymodel2
# import mf_posneg_embedding_project as mymodel3
import parallel_mf_posneg_embedding_project as mymodel3
import parallel_mf_posneg_embedding_user as mymodel4
import parallel_mf_posneg_embedding_user_project as mymodel7
# import ranked_rec_eval as rec_eval
import rec_eval as rec_eval
import cofactor
import model_tuning as mt
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import text_utils
import pandas as pd
from scipy import sparse
import seaborn as sns
sns.set(context="paper", font_scale=1.5, rc={"lines.linewidth": 2}, font='DejaVu Serif')
DEBUG_MODE = True
DATA_DIR = 'data/rec_data/all'
if DEBUG_MODE:
    DATA_DIR = 'data/rec_data/debug'
unique_uid = list()
with open(os.path.join(DATA_DIR, 'unique_uid_cate.txt'), 'r') as f:
    for line in f:
        unique_uid.append(line.strip())

unique_pid = list()
with open(os.path.join(DATA_DIR, 'unique_pid_cate.txt'), 'r') as f:
    for line in f:
        unique_pid.append(line.strip())
n_projects = len(unique_pid)
n_users = len(unique_uid)

print n_users, n_projects

def load_data(csv_file, shape=(n_users, n_projects)):
    tp = pd.read_csv(csv_file)
    timestamps, timestamps_end, category, rows, cols = np.array(tp['timestamp']), np.array(tp['timestamp_end']), np.array(tp['category']), \
                                                       np.array(tp['bid']), np.array(tp['pid']) #rows will be user ids, cols will be projects-ids.
    seq = np.concatenate((  rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int'),
                            timestamps[:, None], timestamps_end[:, None]
                            ,category[:, None]
                          ), axis=1)
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
    return data, seq, tp

train_data, train_raw, train_df = load_data(os.path.join(DATA_DIR, 'train_cate.csv'))
# print train_data
# print train_raw
# print '\n'
# print np.max(train_df[train_df['bid'] == 0]['timestamp'])
# newdf =  train_df[train_df['bid'] == slice(0,2)]['timestamp']
# print newdf[0]
# print newdf[1]

user_activity = np.asarray(train_data.sum(axis=1)).ravel()
numbackers_per_project = np.asarray(train_data.astype('int64').sum(axis=0)).ravel()
vad_data, vad_raw, vad_df = load_data(os.path.join(DATA_DIR, 'validation_cate.csv'))

test_data, test_raw, test_df = load_data(os.path.join(DATA_DIR, 'test_cate.csv'))

PLOT_INFO = False
if PLOT_INFO:
    plt.semilogx(1 + np.arange(n_users), -np.sort(-user_activity), 'o')
    plt.ylabel('Number of invested projects')
    plt.xlabel('Backers')
    plt.show()
    pass

    plt.semilogx(1 + np.arange(n_projects), -np.sort(-numbackers_per_project), 'o')
    plt.ylabel('Number of investors')
    plt.xlabel('Projects')
    plt.show()
    pass

####################Generate project-project co-occurrence matrix based on the user backed projects history ############
# ##################       This will build a project-project co-occurrence matrix           ############################
#user 1: project 1, project 2, ... project k --> project 1, 2, ..., k will be seen as a sentence ==> do co-occurrence.
np.random.seed(1024) #set random seed
def _coord_batch(lo, hi, train_data, prefix = 'cate-project', max_neighbor_words = 200, choose='macro'):
    rows = []
    cols = []

    for u in xrange(lo, hi):
        #print train_data[u].nonzero()[1] #names all the item ids that the user at index u watched nonzero return a
        # 2D array, index 0 will be the row index and index 1 will be columns whose values are not equal to 0
        lst_words = train_data[u].nonzero()[1]
        if len(lst_words) > max_neighbor_words:
            if choose == 'micro':
                #approach 1: randomly select max_neighbor_words for each word.
                for w in lst_words:
                    tmp = lst_words.remove(w)
                    #random choose max_neigbor words in the list:
                    neighbors = np.random.choice(tmp, max_neighbor_words, replace=False)
                    for c in neighbors:
                        rows.append(w)
                        cols.append(c)
            if choose == 'macro':
                #approach 2: randomly select the sentence with length of max_neigbor_words + 1, then do permutation.
                lst_words = np.random.choice(lst_words, max_neighbor_words + 1, replace=False)
                for w, c in itertools.permutations(lst_words, 2):
                    rows.append(w)
                    cols.append(c)
        else:
            for w, c in itertools.permutations(lst_words, 2):
                rows.append(w)
                cols.append(c)
    np.save(os.path.join(DATA_DIR, 'co-temp' ,'cate-%s_coo_%d_%d.npy' % (prefix, lo, hi)),
            np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1)) #append column wise.
    pass

from joblib import Parallel, delayed

batch_size = 5000

GENERATE_PROJECT_PROJECT_COOCCURENCE_FILE = True
if GENERATE_PROJECT_PROJECT_COOCCURENCE_FILE:
    t1 = time.time()
    print 'Generating project project co-occurrence matrix'
    start_idx = range(0, n_users, batch_size)
    end_idx = start_idx[1:] + [n_users]
    Parallel(n_jobs=8)(delayed(_coord_batch)(lo, hi, train_data, prefix = 'cate-project') for lo, hi in zip(start_idx, end_idx))
    t2 = time.time()
    print 'Time : %d seconds'%(t2-t1)
    pass
########################################################################################################################
####################Generate user-user co-occurrence matrix based on the same projects they backed######################
#####################        THis will build a user-user co-occurrence matrix ##########################################

GENERATE_USER_USER_COOCCURENCE_FILE = False
if GENERATE_USER_USER_COOCCURENCE_FILE:
    t1 = time.time()
    print 'Generating user user co-occurrence matrix'
    start_idx = range(0, n_projects, batch_size)
    end_idx = start_idx[1:] + [n_projects]
    Parallel(n_jobs=8)(delayed(_coord_batch)(lo, hi, train_data.T, prefix = 'backer') for lo, hi in zip(start_idx, end_idx))
    t2 = time.time()
    print 'Time : %d seconds'%(t2 - t1)
    pass
########################################################################################################################

def _load_coord_matrix(start_idx, end_idx, nrow, ncol, prefix = 'project'):
    X = sparse.csr_matrix((nrow, ncol), dtype='float32')

    for lo, hi in zip(start_idx, end_idx):
        coords = np.load(os.path.join(DATA_DIR, 'co-temp', '%s_coo_%d_%d.npy' % (prefix, lo, hi)))

        rows = coords[:, 0]
        cols = coords[:, 1]

        tmp = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(nrow, ncol), dtype='float32').tocsr()
        X = X + tmp

        print("%s %d to %d finished" % (prefix, lo, hi))
        sys.stdout.flush()
    return X

BOOLEAN_LOAD_PP_COOCC_FROM_FILE = False
X, Y = None, None
if BOOLEAN_LOAD_PP_COOCC_FROM_FILE:
    print 'Loading project project co-occurrence matrix'
    t1 = time.time()
    start_idx = range(0, n_users, batch_size)
    end_idx = start_idx[1:] + [n_users]
    X = _load_coord_matrix(start_idx, end_idx, n_projects, n_projects, prefix = 'project') #project project co-occurrence matrix
    print X
    print 'dumping matrix ...'
    text_utils.save_pickle(X, os.path.join(DATA_DIR,'pro_pro_cooc.dat'))
    t2 = time.time()
    print 'Time : %d seconds'%(t2-t1)
else:
    print 'test loading model from pickle file'
    t1 = time.time()
    X = text_utils.load_pickle(os.path.join(DATA_DIR,'pro_pro_cooc.dat'))
    t2 = time.time()
    print '[INFO]: sparse matrix size of project project co-occurrence matrix: %d mb\n' % (
                                                    (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / (1024 * 1024))
    print 'Time : %d seconds'%(t2-t1)

#X = None
BOOLEAN_LOAD_UU_COOCC_FROM_FILE = False
if BOOLEAN_LOAD_UU_COOCC_FROM_FILE:
    print 'Loading user user co-occurrence matrix'
    t1 = time.time()
    start_idx = range(0, n_projects, batch_size)
    end_idx = start_idx[1:] + [n_projects]
    Y = _load_coord_matrix(start_idx, end_idx, n_users, n_users, prefix = 'backer') #user user co-occurrence matrix

    t2 = time.time()
    print 'Time : %d seconds' % (t2 - t1)

    print 'dumping matrix ...'
    t1 = time.time()
    text_utils.save_pickle(Y, os.path.join(DATA_DIR, 'user_user_cooc.dat'))
    t2 = time.time()
    print 'Time : %d seconds'%(t2-t1)
else:
    print 'test loading model from pickle file'
    t1 = time.time()
    Y = text_utils.load_pickle(os.path.join(DATA_DIR, 'user_user_cooc.dat'))
    t2 = time.time()
    print '[INFO]: sparse matrix size of user user co-occurrence matrix: %d mb\n' % (
                                                    (Y.data.nbytes + Y.indices.nbytes + Y.indptr.nbytes) / (1024 * 1024))
    print 'Time : %d seconds'%(t2-t1)
################# LOADING NEGATIVE CO-OCCURRENCE MATRIX ########################################
LOAD_NEGATIVE_MATRIX = True
if LOAD_NEGATIVE_MATRIX:
    def _load_negative_coord_matrix(start_idx, end_idx, nrow, ncol, prefix='project'):
        X = sparse.csr_matrix((nrow, ncol), dtype='float32')

        for lo, hi in zip(start_idx, end_idx):
            coords = np.load(os.path.join(DATA_DIR, 'negative-co-temp', 'negative_%s_coo_%d_%d.npy' % (prefix, lo, hi)))
            rows = coords[:, 0]
            cols = coords[:, 1]
            tmp = sparse.coo_matrix((np.ones_like(rows), (rows, cols)), shape=(nrow, ncol), dtype='float32').tocsr()
            X = X + tmp
            print("%s %d to %d finished" % (prefix, lo, hi))
            sys.stdout.flush()
        return X

    BOOLEAN_NEGATIVE_LOAD_PP_COOCC_FROM_FILE = False
    X_neg, Y_neg = None, None
    if BOOLEAN_NEGATIVE_LOAD_PP_COOCC_FROM_FILE:
        print 'Loading negative project project co-occurrence matrix'
        t1 = time.time()
        start_idx = range(0, n_users, batch_size)
        end_idx = start_idx[1:] + [n_users]
        X_neg = _load_negative_coord_matrix(start_idx, end_idx, n_projects, n_projects,
                               prefix='project')  # project project co-occurrence matrix
        print X_neg
        print 'dumping matrix ...'
        text_utils.save_pickle(X_neg, os.path.join(DATA_DIR, 'negative_pro_pro_cooc.dat'))
        t2 = time.time()
        print 'Time : %d seconds' % (t2 - t1)
    else:
        print 'test loading model from pickle file'
        t1 = time.time()
        X_neg = text_utils.load_pickle(os.path.join(DATA_DIR, 'negative_pro_pro_cooc.dat'))
        t2 = time.time()
        print '[INFO]: sparse matrix size of negative project project co-occurrence matrix: %d mb\n' % (
            (X_neg.data.nbytes + X_neg.indices.nbytes + X_neg.indptr.nbytes) / (1024 * 1024))
        print 'Time : %d seconds' % (t2 - t1)

    BOOLEAN_LOAD_NEGATIVE_UU_COOCC_FROM_FILE = False
    if BOOLEAN_LOAD_NEGATIVE_UU_COOCC_FROM_FILE:
        print 'Loading negative user user co-occurrence matrix'
        t1 = time.time()
        start_idx = range(0, n_projects, batch_size)
        end_idx = start_idx[1:] + [n_projects]
        Y_neg = _load_negative_coord_matrix(start_idx, end_idx, n_users, n_users, prefix='backer')  # user user co-occurrence matrix

        t2 = time.time()
        print 'Time : %d seconds' % (t2 - t1)

        print 'dumping matrix ...'
        t1 = time.time()
        text_utils.save_pickle(Y_neg, os.path.join(DATA_DIR, 'negative_user_user_cooc.dat'))
        t2 = time.time()
        print 'Time : %d seconds' % (t2 - t1)
    else:
        print 'test loading model from pickle file'
        t1 = time.time()
        Y_neg = text_utils.load_pickle(os.path.join(DATA_DIR, 'negative_user_user_cooc.dat'))
        t2 = time.time()
        print '[INFO]: sparse matrix size of negative user user co-occurrence matrix: %d mb\n' % (
            (Y_neg.data.nbytes + Y_neg.indices.nbytes + Y_neg.indptr.nbytes) / (1024 * 1024))
        print 'Time : %d seconds' % (t2 - t1)

################################################################################################
########## converting CO-OCCURRENCE MATRIX INTO Shifted Positive Pointwise Mutual Information (SPPMI) matrix ###########
####### We already know the user-user co-occurrence matrix Y and project-project co-occurrence matrix X
SHIFTED_K_VALUE = 1
def get_row(M, i):
    #get the row i of sparse matrix:
    lo,hi = M.indptr[i], M.indptr[i + 1]
    # M.indices[lo:hi] will contain all column index,
    # while M.data[lo:hi] contain all the column values > 0 of those columns
    return lo, hi, M.data[lo:hi], M.indices[lo:hi]

def convert_to_SPPMI_matrix(M, max_row, shifted_K = 1):
    # if we sum the co-occurrence matrix by row wise or column wise --> we have an array that contain the #(i) values
    obj_counts = np.asarray(M.sum(axis=1)).ravel()
    total_obj_pairs = M.data.sum()
    M_sppmi = M.copy()
    for i in xrange(max_row):
        lo, hi, data, indices = get_row(M, i)
        M_sppmi.data[lo:hi] = np.log(data*total_obj_pairs/(obj_counts[i]*obj_counts[indices]))
    M_sppmi.data[M_sppmi.data < 0 ] = 0
    M_sppmi.eliminate_zeros()
    if shifted_K == 1:
        return M_sppmi
    else:
        M_sppmi.data -= np.log(shifted_K)
        M_sppmi.data[M_sppmi.data < 0] = 0
        M_sppmi.eliminate_zeros()
print 'converting co-occurrence matrix into sppmi matrix'
t1 = time.time()
X_sppmi = convert_to_SPPMI_matrix(X, max_row = n_projects, shifted_K=SHIFTED_K_VALUE)
Y_sppmi = convert_to_SPPMI_matrix(Y, max_row = n_users, shifted_K=SHIFTED_K_VALUE)
t2 = time.time()
print 'Time : %d seconds'%(t2-t1)
# if DEBUG_MODE:
#     print 'project sppmi matrix'
#     print X_sppmi
#     print 'user sppmi matrix'
#     print Y_sppmi

############### Negative SPPMI matrix ##########################
if LOAD_NEGATIVE_MATRIX:
    print 'converting negative co-occurrence matrix into sppmi matrix'
    t1 = time.time()
    X_neg_sppmi = convert_to_SPPMI_matrix(X_neg, max_row=n_projects, shifted_K=SHIFTED_K_VALUE)
    Y_neg_sppmi = convert_to_SPPMI_matrix(Y_neg, max_row=n_users, shifted_K=SHIFTED_K_VALUE)
    t2 = time.time()
    print 'Time : %d seconds' % (t2 - t1)
################################################################


######## Finally, we have train_data, vad_data, test_data,
# X_sppmi: project project Shifted Positive Pointwise Mutual Information matrix
# Y_sppmi: user-user       Shifted Positive Pointwise Mutual Information matrix
start = time.time()

print 'Training data',train_data.shape
print 'Validation data',vad_data.shape
print 'Testing data',test_data.shape

print 'The model options are:'
print 'model2: positive project embedding + positive user embedding'
print 'model3: positive project embedding + negative project embedding'
print 'model4: positive user embedding + negative user embedding'
print 'model5: positive and negative project embedding + positive user embedding'
print 'model6: positive project embedding + positive + negative user embedding'
print 'model7: positive + negative project embedding + positive + negative user embedding'
print 'enter the model: (example : model2)'

n_jobs = 1 #default value
model_type = 'model2' #default value
save_dir = os.path.join(DATA_DIR, 'model_res', 'ML20M_ns%d_scale1.0' % (SHIFTED_K_VALUE))
tuner = mt.ModelTuning(train_data, vad_data, test_data, X_sppmi, X_neg_sppmi, Y_sppmi, Y_neg_sppmi,
                           save_dir=save_dir)
n_arg = len(sys.argv)
if n_arg < 4:
    print 'error input, please do: python ks_rec.py [TUNING(0/1)] [model_type] [n_jobs]'
    sys.exit(1)
else:
    TUNING = bool(sys.argv[1])
    model_type = sys.argv[2]
    n_jobs = int(sys.argv[3])
print 'you choosed model %s, with TUNING: %r and n_jobs: %d'%(model_type, TUNING, n_jobs)

# TUNING = True
if TUNING:
    # model_type = raw_input()
    # print 'you choose: ', model_type
    # found_model = False
    if model_type != 'cofactor':
        for i in range(2,8,1):
            possible_model_type = 'model%d'%(i)
            if possible_model_type == model_type:
                found_model = True
                break
        if not found_model:
            print 'wrong model type'
            sys.exit(1)
    # print 'Enter the number of jobs:'
    # n_jobs = int(input())
    # print 'Setting n_jobs to %d'%(n_jobs)
    print 'Running ...'
    tuner.run(model_type, n_jobs = n_jobs, n_components = 100, max_iter = 20, vad_K = 100)
else:
    tuner.run_alone("model2", n_jobs = n_jobs, mu_u_p = 0.3)
    tuner.run_alone("model3", n_jobs = n_jobs, mu_p_p = 0.3)
    tuner.run_alone("model4", n_jobs = n_jobs, mu_u_p = 0.3)
    tuner.run_alone("model5", n_jobs = n_jobs, mu_p_p = 0.3, mu_p_n = 0.3)
    tuner.run_alone("model6", n_jobs = n_jobs, mu_u_p = 0.3, mu_u_n = 0.3)
    tuner.run_alone("model7", n_jobs = n_jobs, mu_u_p = 0.3, mu_p_n = 0.3, mu_p_p = 0.1)

end = time.time()
print 'Total time : %d seconds or %f minutes'%(end-start, float((end-start))/60.0)
