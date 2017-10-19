import sys

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
import seaborn as sns
sns.set(context="paper", font_scale=1.5, rc={"lines.linewidth": 2}, font='DejaVu Serif')

from joblib import Parallel, delayed

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
with open(os.path.join(DATA_DIR, 'unique_uid_sub.txt'), 'r') as f:
    for line in f:
        unique_uid.append(line.strip())

unique_movieId = list()
with open(os.path.join(DATA_DIR, 'unique_sid_sub.txt'), 'r') as f:
    for line in f:
        unique_movieId.append(line.strip())
n_projects = len(unique_movieId)
n_users = len(unique_uid)

print n_users, n_projects

def load_data(csv_file, shape=(n_users, n_projects)):
    tp = pd.read_csv(csv_file)
    count, rows, cols = np.array(tp['count']), np.array(tp['uid']), np.array(tp['sid']) #rows will be user ids, cols will be projects-ids.
    seq = np.concatenate((  rows[:, None], cols[:, None], np.ones((rows.size, 1), dtype='int'),
                            count[:, None]
                          ), axis=1)
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int16, shape=shape)
    return data, seq, tp


vad_data, vad_raw, vad_df = load_data(os.path.join(DATA_DIR, 'vad.num.sub.csv'))
test_data, test_raw, test_df = load_data(os.path.join(DATA_DIR, 'test.num.sub.csv'))


print 'The model options are:'
print 'model2: positive project embedding + positive user embedding'
print 'model3: positive project embedding + negative project embedding'
print 'model4: positive user embedding + negative user embedding'
print 'model5: positive and negative project embedding + positive user embedding'
print 'model6: positive project embedding + positive + negative user embedding'
print 'model7: positive + negative project embedding + positive + negative user embedding'
print 'enter the model: (example : model2)'

train_data, train_raw, train_df =  load_data(os.path.join(DATA_DIR, 'train.num.sub.csv'))
LOAD_NEGATIVE_MATRIX = True
#for i in range(10):
for i in range(1):
# for i in range(9,-1,-1):
    FOLD = i
    print '*************************************FOLD %d ******************************************'%FOLD
    # train_data, train_raw, train_df = load_data(os.path.join(DATA_DIR, 'train_fold%d.csv'%FOLD))

    print 'loading pro_pro_cooc_fold%d.dat'%FOLD
    t1 = time.time()
    X = text_utils.load_pickle(os.path.join(DATA_DIR,'pro_pro_cooc_fold%d.dat'%FOLD))
    t2 = time.time()
    print '[INFO]: sparse matrix size of project project co-occurrence matrix: %d mb\n' % (
                                                    (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / (1024 * 1024))
    print 'Time : %d seconds'%(t2-t1)

    print 'loading user_user_cooc_fold%d.dat'%FOLD
    t1 = time.time()
    Y = text_utils.load_pickle(os.path.join(DATA_DIR, 'user_user_cooc_fold%d.dat'%FOLD))
    t2 = time.time()
    print '[INFO]: sparse matrix size of user user co-occurrence matrix: %d mb\n' % (
                                                    (Y.data.nbytes + Y.indices.nbytes + Y.indptr.nbytes) / (1024 * 1024))
    print 'Time : %d seconds'%(t2-t1)
    ################# LOADING NEGATIVE CO-OCCURRENCE MATRIX ########################################

    if LOAD_NEGATIVE_MATRIX:
        print 'test loading %snegative_pro_pro_cooc%s.dat' % (ACTIVE_CATE_PREFIX, NEG_SAMPLE_MODE)
        t1 = time.time()
        X_neg = text_utils.load_pickle(os.path.join(DATA_DIR, '%snegative_pro_pro_cooc%s_fold%d.dat'%(ACTIVE_CATE_PREFIX, NEG_SAMPLE_MODE, FOLD)))
        t2 = time.time()
        print '[INFO]: sparse matrix size of negative project project co-occurrence matrix: %d mb\n' % (
            (X_neg.data.nbytes + X_neg.indices.nbytes + X_neg.indptr.nbytes) / (1024 * 1024))
        print 'Time : %d seconds' % (t2 - t1)

        print 'test loading negative_user_user_cooc.dat'
        t1 = time.time()
        Y_neg = text_utils.load_pickle(os.path.join(DATA_DIR, 'negative_user_user_cooc_fold%d.dat'%FOLD))
        t2 = time.time()
        print '[INFO]: sparse matrix size of negative user user co-occurrence matrix: %d mb\n' % (
            (Y_neg.data.nbytes + Y_neg.indices.nbytes + Y_neg.indptr.nbytes) / (1024 * 1024))
        print 'Time : %d seconds' % (t2 - t1)

    ################################################################################################
    ########## converting CO-OCCURRENCE MATRIX INTO Shifted Positive Pointwise Mutual Information (SPPMI) matrix ###########
    ####### We already know the user-user co-occurrence matrix Y and project-project co-occurrence matrix X
    SHIFTED_K_VALUE = 10
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
	    return M_sppmi
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
    X_neg_sppmi = None
    Y_neg_sppmi = None
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
        TUNING = bool(int(sys.argv[1]))
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
        tuner.run(model_type, n_jobs = n_jobs, n_components = 100, max_iter = 50, vad_K = 100)
    else:
        if model_type == 'cofactor':
            tuner.run_alone("cofactor", n_jobs=n_jobs, vad_K=100, fold=FOLD)
        if model_type == 'model2':
            tuner.run_alone("model2", n_jobs = n_jobs, mu_u_p = 1.0, mu_p_p = 1.0, vad_K=100, fold=FOLD )
        if model_type == 'model3':
            tuner.run_alone("model3", n_jobs = n_jobs, mu_p_p = 1.0, mu_p_n = 1.0, vad_K=100, fold=FOLD)
        if model_type == 'mcf':
            tuner.run_alone("mcf", n_jobs = n_jobs, mu_p_p = 1.0, mu_p_n = 1.0, mu_u_p = 1.0, vad_K=100, fold=FOLD)

        if model_type == 'separatedmodel2':
            tuner.run_alone("separatedmodel2", n_jobs = n_jobs, max_iter = 10,  mu_u_p = 1.0, mu_p_p = 1.0, vad_K=100, fold=FOLD)
        if model_type == 'separatedmodel3':
            tuner.run_alone("separatedmodel3", n_jobs = n_jobs, max_iter = 10, mu_p_p = 1.0, mu_p_n = 1.0, vad_K=100, fold=FOLD)
        if model_type == 'separatedmcf':
            tuner.run_alone("separatedmcf", n_jobs = n_jobs, max_iter = 10, mu_u_p = 1.0, mu_p_p = 1.0, mu_p_n = 1.0, vad_K=100, fold=FOLD)
    end = time.time()
print 'Total time : %d seconds or %f minutes'%(end-start, float((end-start))/60.0)
