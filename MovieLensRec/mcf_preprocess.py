import sys
import datetime
import json
import os
import time

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import pandas as pd
import scipy.sparse
import utils

import seaborn as sns
sns.set(context="paper", font_scale=1.5, rc={"lines.linewidth": 2}, font='DejaVu Serif')

t1 = time.time()

DEBUG_MODE = True
REVERSE = True # if set to true, project with larger timestamp will have smaller id

def timestamp_to_date(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
def date_to_timestamp(date):
    return utils.convert_to_datetime(date)
def get_count(df, id):
    count_groupbyid = df[[id]].groupby(id, as_index=False)
    count = count_groupbyid.size()
    return count
#remove users who backed less than min_pc projects, and projects with less than min_uc users:
def filter(df, min_pc=10, min_uc=5):
    #keep users who backed at least min_pc projects
    current_size = 0
    next_size = df.shape[0]
    iter = 1
    while(current_size != next_size):
        print 'filter with loop %d, size: %d'%(iter, df.shape[0])
        iter += 1
        current_size = df.shape[0]
        if min_pc > 0:
            usercount = get_count(df, 'userId')
            df = df[df['userId'].isin(usercount.index[usercount >= min_pc])]
        if current_size != next_size:
            continue
        # keep projects which are backed by at least min_uc users
        # After doing this, some of the projects will have less than min_uc users, if we remove them,
        # some of users may have less than min_pc backed projects
        # ==> keep looping until stable.
        if min_uc > 0:
            projectcount = get_count(df, 'movieId')
            df = df[df['movieId'].isin(projectcount.index[projectcount >= min_uc])]
        next_size = df.shape[0]
        # Update both usercount and songcount after filtering
    usercount, projectcount = get_count(df, 'userId'), get_count(df, 'movieId')
    return df, usercount, projectcount


#read project-timestamp into panda dataframe and sort by timestamp.
PROJECT_INFO_PATH = "data/ml/ratings.csv"
user_pro_data = pd.read_csv(PROJECT_INFO_PATH, header=0,  sep=',') #userId,movieId,rating,timestamp
# all_data = user_pro_data
user_pro_data = user_pro_data.drop_duplicates(['userId','movieId'])
print 'project-info-path: %d'% user_pro_data.shape[0]
user_pro_data = user_pro_data[user_pro_data['rating'] >= 4.0]
user_pro_data = user_pro_data[user_pro_data['timestamp'] > 0]
print 'After removing projects with empty ts of project-info-path and rating >= 4: %d', user_pro_data.shape[0]
if REVERSE:
    user_pro_data = user_pro_data.sort_index(by=['timestamp'], ascending=False) #smaller id with larger ts
else:
    user_pro_data = user_pro_data.sort_index(by=['timestamp'], ascending=True) #smaller id with smaller ts
#print project_info_data
tstamp = np.array(user_pro_data['timestamp'])
print("Time span of the dataset: From %s to %s" %
      (timestamp_to_date(np.min(tstamp)), timestamp_to_date(np.max(tstamp))))
# apparently the timestamps are ordered, check to make sure
for i in xrange(tstamp.size - 1):
    if tstamp[i] < tstamp[i + 1] and REVERSE:
        print 'must reorder'
        sys.exit(1)
    if tstamp[i] > tstamp[i + 1] and not REVERSE:
        print 'must reorder'
        sys.exit(1)

# print project_info_data
# plt.hist(tstamp, bins=50)
# xticks = np.linspace(tstamp[0], tstamp[-1], 20)
# plt.xticks(xticks, map(lambda x: timestamp_to_date(x)[:7], xticks), rotation=90)
# plt.show()

user_pro_data = user_pro_data[np.isfinite(user_pro_data['timestamp'])]
print len(user_pro_data)
print 'Total unique projects before filtering : %d'%(len(user_pro_data.movieId.unique()))
print 'Total unique backers before filtering : %d'%(len(user_pro_data.userId.unique()))
user_pro_data, user_activities, project_popularity = filter(user_pro_data)
user_pro_data['index'] = range(0, user_pro_data.shape[0])
print len(user_pro_data)
# print user_pro_data
print user_pro_data
# #reindex user-id and project-id as we removed some of them.
# pid_ts_tmp = user_pro_data[['movieId','timestamp']]
# if REVERSE:
#     pid_ts_tmp = pid_ts_tmp.sort_index(by='timestamp', ascending=False) #latest project will have new pid of 0
# else:
#     pid_ts_tmp = pid_ts_tmp.sort_index(by='timestamp', ascending=True) #latest project will have largest pid
# pid_ts_tmp = pid_ts_tmp[['movieId']]
# pid_ts_tmp = pid_ts_tmp.drop_duplicates(cols='movieId')
# pid_ts_tmp['new_movieId'] = range(0, pid_ts_tmp.shape[0])
#
# bid_pid_tmp = user_pro_data[['userId']]
# bid_pid_tmp = bid_pid_tmp.drop_duplicates(cols='userId')
# bid_pid_tmp['new_userId'] = range(0, bid_pid_tmp.shape[0])
#
# # user_pro_data = user_pro_data.join(pid_ts_tmp, on = 'pid')
# user_pro_data = pd.merge(user_pro_data, pid_ts_tmp, on = 'movieId', how='left')
# print user_pro_data
# # user_pro_data = user_pro_data.join(bid_pid_tmp, on = 'bid')
# # user_pro_data = pd.merge(user_pro_data, bid_pid_tmp, on = 'bid').sort_index(by='new_bid', ascending=True)
# user_pro_data = pd.merge(user_pro_data, bid_pid_tmp, on = 'userId', how='left').sort_index(by='index', ascending=True)
# print user_pro_data
#
#
# #save reindexing indices to file for negative embedding later.
# pid_ts_tmp.to_csv('data/rec_data/movieId_map.csv', index=False)
# bid_pid_tmp.to_csv('data/rec_data/userId_map.csv', index=False)


# user_pro_data = user_pro_data[['new_userId', 'new_movieId', 'timestamp']]
# user_pro_data.columns = ['userId', 'movieId', 'timestamp']
# print user_pro_data


if not DEBUG_MODE:
    #re-computing the user-activities and project-popularity:
    print len(user_pro_data)
    user_pro_data, user_activities, project_popularity = filter(user_pro_data)
    print len(user_pro_data)

    unique_uid = user_activities.index #get all unique user ids
    unique_pid = project_popularity.index # get all unique item ids

DATA_DIR = 'data/rec_data/'
CONTAINED_DIR = "all"
# if DEBUG_MODE:
#     CONTAINED_DIR = 'debug'
#     unique_uid = user_pro_data.userId.unique()
#     unique_mid = user_pro_data.movieId.unique()
#
#
#
# with open(os.path.join(DATA_DIR, CONTAINED_DIR, 'unique_uid.txt'), 'w') as f:
#     for uid in unique_uid:
#         f.write('%s\n' % uid)
# with open(os.path.join(DATA_DIR, CONTAINED_DIR, 'unique_mid.txt'), 'w') as f:
#     for mid in unique_mid:
#         f.write('%s\n' % mid)
#split into training, validation and testing data with ratio 7:1:2
BOOLEAN_SPLIT_DATA = True


def _split_batch(min_uid, max_uid, data):
    first_try = True
    i = 0
    train_data, validation_data, test_data = None, None, None
    for uid in range(min_uid, max_uid + 1):
        tmp = data[data['userId'] == uid]
        tmp_test = tmp[:int(0.2 * tmp.shape[0])]
        tmp_valid = tmp[int(0.2 * tmp.shape[0]): int(0.3 * tmp.shape[0])]
        tmp_train = tmp[int(0.3 * tmp.shape[0]):]
        i += 1
        if i % 10000 == 0:
            print '%d passed' % (i)
        if first_try:
            test_data = tmp_test
        else:
            test_data = test_data.append(tmp_test)

        if first_try:
            validation_data = tmp_valid
        else:
            validation_data = validation_data.append(tmp_valid)

        if first_try:
            train_data = tmp_train
        else:
            train_data = train_data.append(tmp_train)
        first_try = False
    return train_data, validation_data, test_data
if BOOLEAN_SPLIT_DATA:
    # print 'Total userId: %d'%(len(userIds))
    # print 'Total movieId: %d'%(len(movieIds))
    # print userIds[0:10]
    # min_uid = np.min(userIds)
    # max_uid = np.max(userIds)
    #
    #
    # train_data, validation_data, test_data = _split_batch(min_uid, max_uid, user_pro_data)
    # print len(train_data)
    # print len(validation_data)
    # print len(test_data)
    user_pro_data = user_pro_data.sort_index(by=['timestamp'])
    train_data, validation_data, test_data = None, None , None
    tmp = user_pro_data
    while True:
        train_data = tmp[:int(0.7 * tmp.shape[0])]
        validation_data = tmp[int(0.7 * tmp.shape[0]):int(0.8 * tmp.shape[0])]
        test_data = tmp[int(0.8 * tmp.shape[0]):]

        #make sure movieId and userId in validation, test are in train data
        mids = set(pd.unique(train_data.movieId))
        uids = set(pd.unique(train_data.userId))
        validation_data = validation_data[validation_data['movieId'].isin(mids)]
        validation_data = validation_data[validation_data['userId'].isin(uids)]
        test_data = test_data[test_data['movieId'].isin(mids)]
        test_data = test_data[test_data['userId'].isin(uids)]



        # if test_data.shape[0] * 4 < train_data.shape[0] or validation_data.shape[0] *8 < train_data.shape[0] :
        #     tmp = train_data
        #     tmp = tmp.append(validation_data, ignore_index=True)
        #     tmp = tmp.append(test_data, ignore_index=True)
        # else:
        #     break
        break

    # reindex user-id and project-id as we removed some of them.
    print train_data.shape
    print validation_data.shape
    print test_data.shape

    all_data = train_data
    all_data = all_data.append(validation_data)
    all_data = all_data.append(test_data)

    print all_data
    print all_data.shape
    all_data = all_data.drop_duplicates(['userId', 'movieId'])
    print all_data.shape
    pid_ts_tmp = all_data[['movieId', 'timestamp']]
    if REVERSE:
        pid_ts_tmp = pid_ts_tmp.sort_index(by='timestamp', ascending=False)  # latest project will have new pid of 0
    else:
        pid_ts_tmp = pid_ts_tmp.sort_index(by='timestamp', ascending=True)  # latest project will have largest pid
    pid_ts_tmp = pid_ts_tmp[['movieId']]
    pid_ts_tmp = pid_ts_tmp.drop_duplicates('movieId')
    pid_ts_tmp['new_movieId'] = range(0, pid_ts_tmp.shape[0])

    bid_pid_tmp = all_data[['userId']]
    bid_pid_tmp = bid_pid_tmp.drop_duplicates('userId')
    bid_pid_tmp['new_userId'] = range(0, bid_pid_tmp.shape[0])

    # user_pro_data = user_pro_data.join(pid_ts_tmp, on = 'pid')
    all_data = pd.merge(all_data, pid_ts_tmp, on='movieId', how='left')
    print all_data
    # user_pro_data = user_pro_data.join(bid_pid_tmp, on = 'bid')
    # user_pro_data = pd.merge(user_pro_data, bid_pid_tmp, on = 'bid').sort_index(by='new_bid', ascending=True)
    all_data = pd.merge(all_data, bid_pid_tmp, on='userId', how='left').sort_index(by='index', ascending=True)
    print all_data
    all_data = all_data[['new_userId', 'new_movieId', 'timestamp']]
    all_data.columns = ['userId', 'movieId', 'timestamp']

    # check if reindex works:
    userIds = all_data.userId.unique()
    assert np.max(userIds) == (len(userIds) - 1)
    movieIds = all_data.movieId.unique()
    assert np.max(movieIds) == (len(movieIds) - 1)
    print len(userIds)
    print len(movieIds)

    unique_uid = all_data.userId.unique()
    unique_mid = all_data.movieId.unique()
    with open(os.path.join(DATA_DIR, CONTAINED_DIR, 'unique_uid.txt'), 'w') as f:
        for uid in unique_uid:
            f.write('%s\n' % uid)
    with open(os.path.join(DATA_DIR, CONTAINED_DIR, 'unique_mid.txt'), 'w') as f:
        for mid in unique_mid:
            f.write('%s\n' % mid)

    # save reindexing indices to file for negative embedding later.
    pid_ts_tmp.to_csv('data/rec_data/movieId_map.csv', index=False)
    bid_pid_tmp.to_csv('data/rec_data/userId_map.csv', index=False)

    #rematching train, validation and test data
    train_data = all_data[:train_data.shape[0]]
    validation_data = all_data[train_data.shape[0]:(train_data.shape[0]+validation_data.shape[0])]
    test_data = all_data[(train_data.shape[0]+validation_data.shape[0]):]
    print train_data.shape
    print validation_data.shape
    print test_data.shape


    train_data.to_csv(os.path.join(DATA_DIR, CONTAINED_DIR, 'train.csv'), index=False)
    validation_data.to_csv(os.path.join(DATA_DIR, CONTAINED_DIR, 'validation.csv'), index=False)
    test_data.to_csv(os.path.join(DATA_DIR, CONTAINED_DIR, 'test.csv'), index=False)

########PREPARE the SPPMI matrix for projec-project co-occurrence using user context ###################################
#each user will back some projects. We see all the projects backed by a user as a sentence

#######################################################################################################################

########PREPARE the SPPMI matrix for user-user co-occurrence using project context ####################################
#each project will be backed by some users. We see all the users backed the same project as a sentence

#######################################################################################################################
t2 = time.time()
print 'Time : %d seconds'%(t2 - t1)
