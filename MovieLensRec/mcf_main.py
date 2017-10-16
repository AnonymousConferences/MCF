import sys

sys.path.append(r'/home/thanh/PenStateCollaboration2016/python/PenRecProject/utils')
sys.path.append(r'/home/thanh/PenStateCollaboration2016/python/PenRecProject')
#sys.path.append(r'/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/utils')
import itertools
import glob
import os
import sys
import mf_embedding
import rec_eval
import cofactor
#os.environ['OPENBLAS_NUM_THREADS'] = '1'

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
user_activity = np.asarray(train_data.sum(axis=1)).ravel()
numbackers_per_project = np.asarray(train_data.astype('int64').sum(axis=0)).ravel()
vad_data, vad_raw = load_data(os.path.join(DATA_DIR, 'validation.csv'))

test_data, test_raw = load_data(os.path.join(DATA_DIR, 'test.csv'))

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

from joblib import Parallel, delayed

batch_size = 5000

GENERATE_PROJECT_PROJECT_COOCCURENCE_FILE = True
if GENERATE_PROJECT_PROJECT_COOCCURENCE_FILE:
    t1 = time.time()
    print 'Generating project project co-occurrence matrix'
    start_idx = range(0, n_users, batch_size)
    end_idx = start_idx[1:] + [n_users]
    Parallel(n_jobs=8)(delayed(_coord_batch)(lo, hi, train_data, prefix = 'project_500', max_neighbor_words=500) for lo, hi in zip(start_idx, end_idx))
    t2 = time.time()
    print 'Time : %d seconds'%(t2-t1)
    pass
########################################################################################################################
####################Generate user-user co-occurrence matrix based on the same projects they backed######################
#####################        THis will build a user-user co-occurrence matrix ##########################################

GENERATE_USER_USER_COOCCURENCE_FILE = True
if GENERATE_USER_USER_COOCCURENCE_FILE:
    t1 = time.time()
    print 'Generating user user co-occurrence matrix'
    start_idx = range(0, n_projects, batch_size)
    end_idx = start_idx[1:] + [n_projects]
    Parallel(n_jobs=8)(delayed(_coord_batch)(lo, hi, train_data.T, prefix = 'backer_100', max_neighbor_words=100) for lo, hi in zip(start_idx, end_idx))
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
    X = _load_coord_matrix(start_idx, end_idx, n_projects, n_projects, prefix = 'project_500') #project project co-occurrence matrix
    print X
    print 'dumping matrix ...'
    text_utils.save_pickle(X, os.path.join(DATA_DIR,'pro_pro_cooc_500.dat'))
    t2 = time.time()
    print 'Time : %d seconds'%(t2-t1)
else:
    print 'test loading model from pickle file'
    t1 = time.time()
    X = text_utils.load_pickle(os.path.join(DATA_DIR,'pro_pro_cooc_500.dat'))
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
    Y = _load_coord_matrix(start_idx, end_idx, n_users, n_users, prefix = 'backer_100') #user user co-occurrence matrix

    t2 = time.time()
    print 'Time : %d seconds' % (t2 - t1)

    print 'dumping matrix ...'
    t1 = time.time()
    text_utils.save_pickle(Y, os.path.join(DATA_DIR, 'user_user_cooc_100.dat'))
    t2 = time.time()
    print 'Time : %d seconds'%(t2-t1)
else:
    print 'test loading model from pickle file'
    t1 = time.time()
    Y = text_utils.load_pickle(os.path.join(DATA_DIR, 'user_user_cooc_100.dat'))
    t2 = time.time()
    print '[INFO]: sparse matrix size of user user co-occurrence matrix: %d mb\n' % (
                                                    (Y.data.nbytes + Y.indices.nbytes + Y.indptr.nbytes) / (1024 * 1024))
    print 'Time : %d seconds'%(t2-t1)

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
print 'project sppmi matrix'
print X_sppmi
print 'user sppmi matrix'
print Y_sppmi



######## Finally, we have train_data, vad_data, test_data,
# X_sppmi: project project Shifted Positive Pointwise Mutual Information matrix
# Y_sppmi: user-user       Shifted Positive Pointwise Mutual Information matrix
start = time.time()

print 'Training data',train_data.shape
print 'Validation data',vad_data.shape
print 'Testing data',test_data.shape

scale = 1.0

n_components = 100
max_iter = 20
n_jobs = 1
lam_alpha = lam_beta = 1e-3 * scale
lam_theta = lam_gamma = 1e-5 * scale
c0 = 1. * scale
c1 = 20. * scale

save_dir = os.path.join(DATA_DIR, 'model_res', 'ML20M_ns%d_scale%1.2E' % (SHIFTED_K_VALUE, scale))
print save_dir
print 'cleaning folder'
lst = glob.glob(os.path.join(save_dir, '*.npz'))
for f in lst:
   os.remove(f)

#save_dir = DATA_DIR + '/model_res/model'


#coder = mf_embedding.MFEmbedding(mu_u = 0.5, mu_p = 0.5,
#                        n_components=n_components, max_iter=max_iter, batch_size=11, init_std=0.01, dtype=np.float32, n_jobs=n_jobs,
#                        random_state=98765, save_params=True, save_dir=save_dir, early_stopping=True, verbose=True,
#                        lambda_alpha = lam_alpha, lambda_theta=lam_theta, lambda_beta=lam_beta, lambda_gamma=lam_gamma, c0=c0, c1=c1)
#coder.fit(train_data, X_sppmi, Y_sppmi, vad_data=vad_data, batch_users=2, k=10)

coder = cofactor.CoFacto(
                       n_components=n_components, max_iter=max_iter, batch_size=11, init_std=0.01, dtype=np.float32, n_jobs=n_jobs,
                       random_state=98765, save_params=True, save_dir=save_dir, early_stopping=True, verbose=True,
                       lambda_alpha = lam_alpha, lambda_theta=lam_theta, lambda_beta=lam_beta, lambda_gamma=lam_gamma, c0=c0, c1=c1)
coder.fit(train_data, X_sppmi, vad_data=vad_data, batch_users=2, k=10)

test_data.data = np.ones_like(test_data.data)
n_params = len(glob.glob(os.path.join(save_dir, '*.npz')))
params = np.load(os.path.join(save_dir, 'CoFacto_K%d_iter%d.npz' % (n_components, n_params - 1)))
U, V = params['U'], params['V']
print U.shape
print V.shape

if DEBUG_MODE:
    print 'Accuracy of training:', float(np.logical_and(train_data.toarray(), np.round(U.dot(V.T))).sum())/float(train_data.toarray().sum())
    true_predict = np.logical_and(test_data.toarray(), np.round(U.dot(V.T))).sum(1)
    print 'True predict for users:', true_predict
    print 'Total 1 values in testing data:',test_data.toarray().sum()
    print 'Accuracy of testing: ',float(true_predict.sum())/float(test_data.toarray().sum())
K = 10
print 'Test Recall@%d: %.4f' % (K, rec_eval.recall_at_k(train_data, test_data, U, V, k=K, vad_data=vad_data))
print 'Test Recall@%d: %.4f' % (K, rec_eval.recall_at_k(train_data, test_data, U, V, k=K, vad_data=vad_data))
print 'Test NDCG@%d: %.4f' % (K, rec_eval.normalized_dcg_at_k(train_data, test_data, U, V, k=K, vad_data=vad_data))
print 'Test MAP@%d: %.4f' % (K, rec_eval.map_at_k(train_data, test_data, U, V, k=K, vad_data=vad_data))
np.savez('CoFactor_K100_ML20M.npz', U=U, V=V)


end = time.time()
print 'Total time : %d seconds or %f minutes'%(end-start, float((end-start))/60.0)
#sys.path.remove(r'/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/utils')
sys.path.remove(r'/home/thanh/PenStateCollaboration2016/python/PenRecProject/utils')
sys.path.remove(r'/home/thanh/PenStateCollaboration2016/python/PenRecProject')
