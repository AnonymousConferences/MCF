import sys

sys.path.append(r'/home/thanh/PenStateCollaboration2016/python/KickstarterRec/utils')
sys.path.append(r'/home/thanh/PenStateCollaboration2016/python/KickstarterRec')
sys.path.append(r'/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/utils')
import itertools
import glob
import os
import sys
import mf_pos_embedding_user_project as mymodel2
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
from sklearn.utils import shuffle
import seaborn as sns
sns.set(context="paper", font_scale=1.5, rc={"lines.linewidth": 2}, font='DejaVu Serif')

from joblib import Parallel, delayed

np.random.seed(98765) #set random seed

#ACTIVE_CATE_PREFIX='active_all_' #or 'active_cate' or ''_
ACTIVE_CATE_PREFIX=''
if ACTIVE_CATE_PREFIX == 'active_all_':
    NEG_SAMPLE_MODE = ''
else:
    NEG_SAMPLE_MODE = 'micro'
if ACTIVE_CATE_PREFIX == '':
    NEG_SAMPLE_MODE=''

DEBUG_MODE = False
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
    seq = np.concatenate((  rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int'),
                            timestamps[:, None]
                          ), axis=1)
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
    return data, seq, tp

def load_data_10folds(csv_file, shape=(n_users, n_projects)):
    original_tp = pd.read_csv(csv_file)
    nrow = original_tp.shape[0]
    folds = []
    j = 0
    for i in range(0, nrow):
        j += 1
        if j > 10:
            j = 1
        folds.append(j)
    folds = shuffle(folds)
    original_tp['fold'] = folds
    print original_tp
    res = []

    for fold in range(1,11):
        tp = original_tp[original_tp['fold'] != fold]
        tp = tp[['userId', 'movieId','timestamp']]
        timestamps, rows, cols = np.array(tp['timestamp']), np.array(tp['userId']), np.array(tp['movieId']) #rows will be user ids, cols will be projects-ids.
        seq = np.concatenate((  rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int'),
                                timestamps[:, None]
                              ), axis=1)
        data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
        res.append((data, seq, tp))
    return res


vad_data, vad_raw, vad_df = load_data(os.path.join(DATA_DIR, 'validation.csv'))
test_data, test_raw, test_df = load_data(os.path.join(DATA_DIR, 'test.csv'))
train_data, train_raw, train_df = load_data(os.path.join(DATA_DIR, 'train.csv'))
user_activity = np.asarray(train_data.sum(axis=1)).ravel()
numbackers_per_project = np.asarray(train_data.astype('int64').sum(axis=0)).ravel()

train_10folds = load_data_10folds(os.path.join(DATA_DIR, 'train.csv'))






PLOT_INFO = False
if PLOT_INFO:
    plt.semilogx(1 + np.arange(n_users), -np.sort(-user_activity), 'o')
    plt.ylabel('Number of liked movies')
    plt.xlabel('Users')
    plt.show()
    pass

    plt.semilogx(1 + np.arange(n_projects), -np.sort(-numbackers_per_project), 'o')
    plt.ylabel('Number of users')
    plt.xlabel('Movies')
    plt.show()
    pass

####################Generate project-project co-occurrence matrix based on the user backed projects history ############
# ##################       This will build a project-project co-occurrence matrix           ############################
#user 1: project 1, project 2, ... project k --> project 1, 2, ..., k will be seen as a sentence ==> do co-occurrence.

def _coord_batch(lo, hi, train_data, prefix = 'project', max_neighbor_words = 200, choose='macro'):
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
    np.save(os.path.join(DATA_DIR, 'co-temp' ,'%s_coo_%d_%d.npy' % (prefix, lo, hi)),
            np.concatenate([np.array(rows)[:, None], np.array(cols)[:, None]], axis=1)) #append column wise.
    pass


batch_size = 5000
for  i in range(10):
    FOLD = i
    train_data, train_raw, train_df = train_10folds[i]
    train_df.to_csv(os.path.join(DATA_DIR, 'train_fold%d.csv'%FOLD), index=False)
    #clear the co-temp folder:
    if os.path.exists(os.path.join(DATA_DIR, 'co-temp')):
        for f in glob.glob(os.path.join(DATA_DIR, 'co-temp', '*.npy')):
            os.remove(f)

    GENERATE_PROJECT_PROJECT_COOCCURENCE_FILE = True
    if GENERATE_PROJECT_PROJECT_COOCCURENCE_FILE:
        t1 = time.time()
        print 'Generating project project co-occurrence matrix'
        start_idx = range(0, n_users, batch_size)
        end_idx = start_idx[1:] + [n_users]
        Parallel(n_jobs=16)(delayed(_coord_batch)(lo, hi, train_data, prefix = 'project') for lo, hi in zip(start_idx, end_idx))
        t2 = time.time()
        print 'Time : %d seconds'%(t2-t1)
        pass
    ########################################################################################################################
    ####################Generate user-user co-occurrence matrix based on the same projects they backed######################
    #####################        This will build a user-user co-occurrence matrix ##########################################

    GENERATE_USER_USER_COOCCURENCE_FILE = True
    if GENERATE_USER_USER_COOCCURENCE_FILE:
        t1 = time.time()
        print 'Generating user user co-occurrence matrix'
        start_idx = range(0, n_projects, batch_size)
        end_idx = start_idx[1:] + [n_projects]
        Parallel(n_jobs=16)(delayed(_coord_batch)(lo, hi, train_data.T, prefix = 'backer') for lo, hi in zip(start_idx, end_idx))
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

    BOOLEAN_LOAD_PP_COOCC_FROM_FILE = True
    X, Y = None, None
    if BOOLEAN_LOAD_PP_COOCC_FROM_FILE:
        print 'Loading project project co-occurrence matrix'
        t1 = time.time()
        start_idx = range(0, n_users, batch_size)
        end_idx = start_idx[1:] + [n_users]
        X = _load_coord_matrix(start_idx, end_idx, n_projects, n_projects, prefix = 'project') #project project co-occurrence matrix
        print 'dumping matrix ...'
        text_utils.save_pickle(X, os.path.join(DATA_DIR,'pro_pro_cooc_fold%d.dat'%FOLD))
        t2 = time.time()
        print 'Time : %d seconds'%(t2-t1)
    else:
        print 'test loading model from pickle file'
        t1 = time.time()
        X = text_utils.load_pickle(os.path.join(DATA_DIR,'pro_pro_cooc_fold%d.dat'%FOLD))
        t2 = time.time()
        print '[INFO]: sparse matrix size of project project co-occurrence matrix: %d mb\n' % (
                                                        (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / (1024 * 1024))
        print 'Time : %d seconds'%(t2-t1)

    #X = None
    BOOLEAN_LOAD_UU_COOCC_FROM_FILE = True
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
        text_utils.save_pickle(Y, os.path.join(DATA_DIR, 'user_user_cooc_fold%d.dat'%FOLD))
        t2 = time.time()
        print 'Time : %d seconds'%(t2-t1)
    else:
        print 'test loading model from pickle file'
        t1 = time.time()
        Y = text_utils.load_pickle(os.path.join(DATA_DIR, 'user_user_cooc_fold%d.dat'%FOLD))
        t2 = time.time()
        print '[INFO]: sparse matrix size of user user co-occurrence matrix: %d mb\n' % (
                                                        (Y.data.nbytes + Y.indices.nbytes + Y.indptr.nbytes) / (1024 * 1024))
        print 'Time : %d seconds'%(t2-t1)


sys.path.remove(r'/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/utils')
sys.path.remove(r'/home/thanh/PenStateCollaboration2016/python/KickstarterRec/utils')
sys.path.remove(r'/home/thanh/PenStateCollaboration2016/python/KickstarterRec')
