import sys

import itertools
import glob
import os
import sys
import mf_pos_embedding_user_project_new as mymodel1
import rec_eval
import cofactor

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import text_utils
import pandas as pd
from scipy import sparse
import seaborn as sns

sns.set(context="paper", font_scale=1.5, rc={"lines.linewidth": 2}, font='DejaVu Serif')
DEBUG_MODE = False
DATA_DIR = 'data/rec_data/all'
if DEBUG_MODE:
    DATA_DIR = 'data/rec_data/debug'
unique_uid = list()
with open(os.path.join(DATA_DIR, 'unique_uid.txt'), 'r') as f:
    for line in f:
        unique_uid.append(line.strip())

unique_pid = list()
with open(os.path.join(DATA_DIR, 'unique_mid.txt'), 'r') as f:
    for line in f:
        unique_pid.append(line.strip())
n_projects = len(unique_pid)
n_users = len(unique_uid)

print n_users, n_projects

def transfer_dataframe(tp, shape=(n_users, n_projects)):
    timestamps, rows, cols = np.array(tp['timestamp']), np.array(tp['userId']), np.array(
        tp['movieId'])  # rows will be user ids, cols will be projects-ids.
    seq = np.concatenate((rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int'), timestamps[:, None]),
                         axis=1)
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
    return data, seq

def load_data(csv_file, shape=(n_users, n_projects)):
    tp = pd.read_csv(csv_file)
    data, seq = transfer_dataframe(tp, shape)
    return data, seq, tp


train_data, train_raw, train_df = load_data(os.path.join(DATA_DIR, 'train.csv'))
user_activity = np.asarray(train_data.sum(axis=1)).ravel()
numbackers_per_project = np.asarray(train_data.astype('int64').sum(axis=0)).ravel()
vad_data, vad_raw, vad_df = load_data(os.path.join(DATA_DIR, 'validation.csv'))

test_data, test_raw, test_df = load_data(os.path.join(DATA_DIR, 'test.csv'))

####################Generate project-project negative co-occurrence matrix based on the user backed projects history ############
# ##################       This will build a project-project co-occurrence matrix           ############################
# user 1: project 1, project 2, ... project k --> project 1, 2, ..., k will be seen as a sentence ==> do co-occurrence.
np.random.seed(98765)  # set random seed


all_data = pd.read_csv('data/ml/ratings.csv')
movieId_map = pd.read_csv('data/rec_data/movieId_map.csv')
userId_map = pd.read_csv('data/rec_data/userId_map.csv')
#choose the data in all_data that contain old userId and old movieId (before reindexing) in training data only
print all_data.shape
all_data = all_data[all_data['userId'].isin(pd.unique(userId_map.userId))]
all_data = all_data[all_data['movieId'].isin(pd.unique(movieId_map.movieId))]
print all_data
neg_all_data = all_data[all_data.rating <= 2]
print neg_all_data.shape
#merge new_userId and new_movieId into all_data
neg_all_data = pd.merge(neg_all_data, movieId_map, on = 'movieId', how='left')
neg_all_data = pd.merge(neg_all_data, userId_map, on = 'userId', how='left')
print neg_all_data.shape
print neg_all_data
#recheck if merging got something wrong
if neg_all_data[neg_all_data['new_movieId'] >= 0].shape == neg_all_data.shape:
    print 'merging is good'
print len(pd.unique(neg_all_data.new_movieId))
print np.max(neg_all_data.new_movieId)
print len(pd.unique(neg_all_data.new_userId))
print np.max(neg_all_data.new_userId)
#keep new_ids only:
neg_all_data = neg_all_data[['new_userId', 'new_movieId', 'timestamp']]
neg_all_data.columns = ['userId', 'movieId', 'timestamp']
print neg_all_data.shape
#get only the low rating movies --> dislike data, then transfer it to sparse matrix
neg_all_data,_ = transfer_dataframe(neg_all_data)
print neg_all_data

def _coord_batch(lo, hi, train_data, prefix = 'project', max_neighbor_words = 50, choose='macro'):
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
    np.save(os.path.join(DATA_DIR, 'negative-co-temp' ,'negative_%s_coo_%d_%d.npy' % (prefix, lo, hi)),
            np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1)) #append column wise.
    pass

def _negative_coord_batch(lo, hi, train_data, neg_all_data = neg_all_data, prefix='project', max_neighbor_words=200, choose='macro'):
    rows = []
    cols = []
    nprojects = train_data.shape[1]
    for u in xrange(lo, hi):
        # print train_data[u].nonzero()[1] #names all the item ids that the user at index u watched nonzero return a
        # 2D array, index 0 will be the row index and index 1 will be columns whose values are not equal to 0
        positive_lst_words = train_data[u].nonzero()[1]
        negative_lst_words = range(0, nprojects)
        lst_words = [w for w in negative_lst_words if w not in positive_lst_words]
        if len(lst_words) > max_neighbor_words:
            if choose == 'micro':
                # approach 1: randomly select max_neighbor_words for each word.
                for w in lst_words:
                    tmp = lst_words.remove(w)
                    # random choose max_neigbor words in the list:
                    neighbors = np.random.choice(tmp, max_neighbor_words, replace=False)
                    for c in neighbors:
                        rows.append(w)
                        cols.append(c)
            if choose == 'macro':
                # approach 2: randomly select the sentence with length of max_neigbor_words + 1, then do permutation.
                lst_words = np.random.choice(lst_words, max_neighbor_words + 1, replace=False)
                for w, c in itertools.permutations(lst_words, 2):
                    rows.append(w)
                    cols.append(c)
        else:
            for w, c in itertools.permutations(lst_words, 2):
                rows.append(w)
                cols.append(c)
    np.save(os.path.join(DATA_DIR, 'negative-co-temp', 'negative_%s_coo_%d_%d.npy' % (prefix, lo, hi)),
            np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1))  # append column wise.
    pass


from joblib import Parallel, delayed

batch_size = 300

GENERATE_NEGATIVE_PROJECT_PROJECT_COOCCURENCE_FILE = True
if GENERATE_NEGATIVE_PROJECT_PROJECT_COOCCURENCE_FILE:
    t1 = time.time()
    print 'Generating negative project project co-occurrence matrix'
    start_idx = range(0, n_users, batch_size)
    end_idx = start_idx[1:] + [n_users]
    Parallel(n_jobs=16)(
        delayed(_coord_batch)(lo, hi, neg_all_data, prefix='project') for lo, hi in zip(start_idx, end_idx))
    t2 = time.time()
    print 'Time : %d seconds' % (t2 - t1)
    pass
########################################################################################################################
####################Generate user-user co-occurrence matrix based on the same projects they backed######################
#####################        THis will build a user-user co-occurrence matrix ##########################################

GENERATE_NEGATIVE_USER_USER_COOCCURENCE_FILE = True
if GENERATE_NEGATIVE_USER_USER_COOCCURENCE_FILE:
    t1 = time.time()
    print 'Generating negative user user co-occurrence matrix'
    start_idx = range(0, n_projects, batch_size)
    end_idx = start_idx[1:] + [n_projects]
    Parallel(n_jobs=16)(
        delayed(_coord_batch)(lo, hi, neg_all_data.T, prefix='backer') for lo, hi in zip(start_idx, end_idx))
    t2 = time.time()
    print 'Time : %d seconds' % (t2 - t1)
    pass


########################################################################################################################

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


BOOLEAN_NEGATIVE_LOAD_PP_COOCC_FROM_FILE = True
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
    print '[INFO]: sparse matrix size of project project co-occurrence matrix: %d mb\n' % (
        (X_neg.data.nbytes + X_neg.indices.nbytes + X_neg.indptr.nbytes) / (1024 * 1024))
    print 'Time : %d seconds' % (t2 - t1)

# X = None
BOOLEAN_LOAD_NEGATIVE_UU_COOCC_FROM_FILE = True
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
    print '[INFO]: sparse matrix size of user user co-occurrence matrix: %d mb\n' % (
        (Y_neg.data.nbytes + Y_neg.indices.nbytes + Y_neg.indptr.nbytes) / (1024 * 1024))
    print 'Time : %d seconds' % (t2 - t1)

########## converting CO-OCCURRENCE MATRIX INTO Shifted Positive Pointwise Mutual Information (SPPMI) matrix ###########
####### We already know the user-user co-occurrence matrix Y and project-project co-occurrence matrix X
SHIFTED_K_VALUE = 1


def get_row(M, i):
    # get the row i of sparse matrix:
    lo, hi = M.indptr[i], M.indptr[i + 1]
    # M.indices[lo:hi] will contain all column index,
    # while M.data[lo:hi] contain all the column values > 0 of those columns
    return lo, hi, M.data[lo:hi], M.indices[lo:hi]


def convert_to_SPPMI_matrix(M, max_row, shifted_K=1):
    # if we sum the co-occurrence matrix by row wise or column wise --> we have an array that contain the #(i) values
    obj_counts = np.asarray(M.sum(axis=1)).ravel()
    total_obj_pairs = M.data.sum()
    M_sppmi = M.copy()
    for i in xrange(max_row):
        lo, hi, data, indices = get_row(M, i)
        M_sppmi.data[lo:hi] = np.log(data * total_obj_pairs / (obj_counts[i] * obj_counts[indices]))
    M_sppmi.data[M_sppmi.data < 0] = 0
    M_sppmi.eliminate_zeros()
    if shifted_K == 1:
        return M_sppmi
    else:
        M_sppmi.data -= np.log(shifted_K)
        M_sppmi.data[M_sppmi.data < 0] = 0
        M_sppmi.eliminate_zeros()


print 'converting co-occurrence matrix into sppmi matrix'
t1 = time.time()
X_neg_sppmi = convert_to_SPPMI_matrix(X_neg, max_row=n_projects, shifted_K=SHIFTED_K_VALUE)
Y_neg_sppmi = convert_to_SPPMI_matrix(Y_neg, max_row=n_users, shifted_K=SHIFTED_K_VALUE)
t2 = time.time()
print 'Time : %d seconds' % (t2 - t1)
# if DEBUG_MODE:
#     print 'project sppmi matrix'
#     print X_sppmi
#     print 'user sppmi matrix'
#     print Y_sppmi

