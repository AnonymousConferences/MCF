import json
import os
import sqlite3

import numpy as np
import pandas as pd
DEBUG_MODE = False

TPS_DIR = 'data/rec_data/all'
TP_file = os.path.join(TPS_DIR, 'yh-ratings.csv')
if DEBUG_MODE:
    TP_file = os.path.join(TPS_DIR, 'test-yh-ratings.csv')


print 'reading data'
tp = pd.read_csv(TP_file, header=None, names=['uid', 'sid', 'rating'])
print tp

MIN_USER_COUNT = 20
MIN_SONG_COUNT = 50

if DEBUG_MODE:
    MIN_USER_COUNT = 5
    MIN_SONG_COUNT = 10


def get_count(tp, id, id2=''):
    if id2!='':
        playcount_groupbyid = tp[[id, id2, 'rating']].groupby([id, id2], as_index=False)
    else:
        playcount_groupbyid = tp[[id, 'rating']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

def filter_triplets(tp, min_uc=MIN_USER_COUNT, min_sc=MIN_SONG_COUNT):
    # Only keep the triplets for songs which were listened to by at least min_sc users.
    songcount = get_count(tp, 'sid')
    tp = tp[tp['sid'].isin(songcount.index[songcount >= min_sc])]

    # Only keep the triplets for users who listened to at least min_uc songs
    # After doing this, some of the songs will have less than min_uc users, but should only be a small proportion
    usercount = get_count(tp, 'uid')
    tp = tp[tp['uid'].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and songcount after filtering
    usercount, songcount = get_count(tp, 'uid'), get_count(tp, 'sid')
    return tp, usercount, songcount

print 'filtering triplets'
tp, usercount, songcount = filter_triplets(tp)

sparsity_level = float(tp.shape[0]) / (usercount.shape[0] * songcount.shape[0])
print "After filtering, there are %d triplets from %d users and %d songs (sparsity level %.3f%%)" % (tp.shape[0],
                                                                                                      usercount.shape[0],
                                                                                                      songcount.shape[0],
                                                                                                      sparsity_level * 100)

# print usercount
print 'total unique songs:%d'%len(songcount)
print 'total unique users:%d'%len(usercount)

song1time = tp[tp['sid'].isin(songcount.index[songcount <= 1])]
print 'total songs with 1 count:%d'%song1time.shape[0]


print '*****************************'
import matplotlib
# matplotlib.use('Agg')
# %matplotlib inline
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(context="paper", font_scale=1.5, rc={"lines.linewidth": 2}, font='DejaVu Serif')
# plt.figure(figsize=(10, 4))
# plt.hist(songcount, bins=100)
# plt.xlabel('number of users by which each song is listened to')
# plt.show()


unique_uid = usercount.index
unique_sid = songcount.index
np.random.seed(98765)

n_users = 100000
n_songs = 25000
if DEBUG_MODE:
    n_users = 50
    n_songs = 25
if n_users > len(unique_uid): n_user =   len(unique_uid) - 1
if n_songs > len(unique_sid): n_songs = len(unique_sid) - 1

print n_users
print n_songs

p_users = usercount / usercount.sum()
idx = np.random.choice(len(unique_uid), size=n_users, replace=False, p=p_users.tolist())
unique_uid = unique_uid[idx]

tp = tp[tp['uid'].isin(unique_uid)]




p_songs = songcount / songcount.sum()
idx = np.random.choice(len(unique_sid), size=n_songs, replace=False, p=p_songs.tolist())
unique_sid = unique_sid[idx]
tp = tp[tp['sid'].isin(unique_sid)]
tp, usercount, songcount = filter_triplets(tp, min_uc=MIN_USER_COUNT, min_sc=MIN_SONG_COUNT)

tp.to_csv(os.path.join(TPS_DIR, 'all.csv'), index=False)

unique_uid = usercount.index
unique_sid = songcount.index
# sparsity_level = float(tp.shape[0]) / (usercount.shape[0] * songcount.shape[0])
print "After subsampling and filtering, there are %d triplets from %d users and %d songs (sparsity level %.3f%%)" % \
(tp.shape[0], usercount.shape[0], songcount.shape[0], sparsity_level * 100)
song2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
with open(os.path.join(TPS_DIR, 'unique_uid_sub.txt'), 'w') as f:
    for uid in unique_uid:
        f.write('%s\n' % uid)
with open(os.path.join(TPS_DIR, 'unique_sid_sub.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)
seeds =[(12345, 13579), (23456, 24680), (34567, 97531), (45678, 80642), (56789, 97531)]
FOLD = 0
for init_seed in seeds:
    np.random.seed(init_seed[0])
    n_ratings = tp.shape[0]
    test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
    test_idx = np.zeros(n_ratings, dtype=bool)
    test_idx[test] = True

    test_tp = tp[test_idx]
    train_tp = tp[~test_idx]


    print "There are total of %d unique users in the training set and %d unique users in the entire dataset" % \
    (len(pd.unique(train_tp['uid'])), len(pd.unique(tp['uid'])))

    print "There are total of %d unique items in the training set and %d unique items in the entire dataset" % \
    (len(pd.unique(train_tp['sid'])), len(pd.unique(tp['sid'])))

    np.random.seed(init_seed[1])
    n_ratings = train_tp.shape[0]
    vad = np.random.choice(n_ratings, size=int(0.10 * n_ratings), replace=False)

    vad_idx = np.zeros(n_ratings, dtype=bool)
    vad_idx[vad] = True

    vad_tp = train_tp[vad_idx]
    train_tp = train_tp[~vad_idx]

    print "There are total of %d unique users in the training set and %d unique users in the entire dataset" % \
    (len(pd.unique(train_tp['uid'])), len(pd.unique(tp['uid'])))


    print "There are total of %d unique items in the training set and %d unique items in the entire dataset" % \
    (len(pd.unique(train_tp['sid'])), len(pd.unique(tp['sid'])))

    def numerize(tp):
        uid = map(lambda x: user2id[x], tp['uid'])
        sid = map(lambda x: song2id[x], tp['sid'])
        tp['uid'] = uid
        tp['sid'] = sid
        return tp
    train_tp = numerize(train_tp)
    train_tp.to_csv(os.path.join(TPS_DIR, 'train.num.sub.fold%d.csv'%FOLD), index=False)
    test_tp = numerize(test_tp)
    test_tp.to_csv(os.path.join(TPS_DIR, 'test.num.sub.fold%d.csv'%FOLD), index=False)
    vad_tp = numerize(vad_tp)
    vad_tp.to_csv(os.path.join(TPS_DIR, 'vad.num.sub.fold%d.csv'%FOLD), index=False)
    FOLD += 1
