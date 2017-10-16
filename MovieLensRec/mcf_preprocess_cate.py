import sys
sys.path.append(r'/home/thanh/PenStateCollaboration2016/python/KickstarterRec/utils')
sys.path.append(r'/home/thanh/PenStateCollaboration2016/python/KickstarterRec')
sys.path.append(r'/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/utils')
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

DEBUG_MODE = False
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
            usercount = get_count(df, 'bid')
            df = df[df['bid'].isin(usercount.index[usercount >= min_pc])]

        # keep projects which are backed by at least min_uc users
        # After doing this, some of the projects will have less than min_uc users, if we remove them,
        # some of users may have less than min_pc backed projects
        # ==> keep looping until stable.
        if min_uc > 0:
            projectcount = get_count(df, 'pid')
            df = df[df['pid'].isin(projectcount.index[projectcount >= min_uc])]
        next_size = df.shape[0]
        # Update both usercount and songcount after filtering
    usercount, projectcount = get_count(df, 'bid'), get_count(df, 'pid')
    return df, usercount, projectcount


#read project-timestamp into panda dataframe and sort by timestamp.
PROJECT_INFO_PATH = "data/project-info/project-fundingduration-cate.csv"
project_info_data = pd.read_csv(PROJECT_INFO_PATH, header=0,  sep='\t')
project_info_data = project_info_data.drop_duplicates(cols='project-id')
print 'project-info-path: %d', project_info_data.shape[0]
project_info_data = project_info_data[project_info_data['funding-from'] != '']
project_info_data = project_info_data[project_info_data['funding-to'] != '']
print 'After removing projects with empty ts of project-info-path: %d', project_info_data.shape[0]
project_info_data = project_info_data[project_info_data['category'] != '']
print 'After removing projects with empty category info: %d', project_info_data.shape[0]
timestamps = []
timestamps_end = []
for funding_from_date in project_info_data['funding-from']:
    timestamps.append(utils.convert_to_timestamp(funding_from_date))
for funding_to_date in project_info_data['funding-to']:
    timestamps_end.append(utils.convert_to_timestamp(funding_to_date))
project_info_data['timestamp'] = timestamps
project_info_data['timestamp_end'] = timestamps_end
if REVERSE:
    project_info_data = project_info_data.sort_index(by=['timestamp'], ascending=False) #smaller id with larger ts
else:
    project_info_data = project_info_data.sort_index(by=['timestamp'], ascending=True) #smaller id with smaller ts
#print project_info_data
tstamp = np.array(project_info_data['timestamp'])
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

#add a new column id into project id to index (smaller id with larger timestamp)
# project_info_data['pid'] = range(0,project_info_data.shape[0])
#add a new column id into project id to index (smaller id with smaller timestamp)
project_info_data['pid'] = range(0,project_info_data.shape[0])
project_map = project_info_data[['project-id','pid']]
project_map.to_csv('data/rec_data/project_to_id_cate.csv', index=False)

#reading user_backed projects and make a new data with format: #user_id new_project_id timestamp_project
BOOLEAN_RESTRUCTURE_USER_BACKED_FILE = True
OUTPUT_RESTRUCTURED_USER_BACKEDPROJECT_FILE = 'data/rec_data/user_backed_projects_ts_cate.csv'

if BOOLEAN_RESTRUCTURE_USER_BACKED_FILE:
    USER_BACKED_INFO_FILE = 'data/kickstarter_v2/user_backed_projects_full.csv'
    project_id_map = dict()  # map project-id to pid and timestamp
    for index, row in project_info_data.iterrows():
        project_id_map[row['project-id']] = (row['pid'], row['timestamp'], row['timestamp_end'], row['category'])


    fr = open(USER_BACKED_INFO_FILE, 'r')
    fw = open(OUTPUT_RESTRUCTURED_USER_BACKEDPROJECT_FILE, 'w')
    fw.write('bid,pid,timestamp,timestamp_end,category\n')
    backer_set = set()
    backer_dic = dict()
    bid = -1
    for line in fr:
        line = line.replace('\n','').replace('\r', '')
        tokens = line.split(',')
        backer_id = tokens[0]

        if len(tokens) <= 1:
            continue
        if backer_id not in backer_set:
            bid += 1
            backer_set.add(backer_id)
            backer_dic[backer_id] = bid
            local_bid = bid
        else:
            local_bid = backer_dic[backer_id]
        for backed_project in tokens[1:]:
            pid, timestamp, timestamp_end, category = project_id_map.setdefault(backed_project, ('','','',''))
            new_line = str(local_bid) + ',' + str(pid) + ',' + str(timestamp) + ',' + str(timestamp_end) + ',' + str(category) +'\n'
            fw.write(new_line)
    fr.close()
    fw.flush()
    fw.close()
    bid = None
    backer_set = None

#add a new user_id columns from 0 to #users.
user_pro_data = pd.read_csv(OUTPUT_RESTRUCTURED_USER_BACKEDPROJECT_FILE, header=0,  sep=',')
TEST_HERE = True
if TEST_HERE:
    print 'Test if project ids are sorted by timestamp'
    t_user_pro_data = user_pro_data.sort_index(by=['timestamp'], ascending=True)
    pidarray = np.array(t_user_pro_data['pid'])
    tsarray = np.array(t_user_pro_data['timestamp'])
    for i in xrange(pidarray.size - 1):
        if tsarray[i] < tsarray[i + 1] and pidarray[i] < pidarray[i + 1] and REVERSE:
            print 'must reorder'
            sys.exit(1)
        if tsarray[i] < tsarray[i + 1] and pidarray[i] > pidarray[i + 1] and not REVERSE:
            print 'must reorder'
            sys.exit(1)

#write to file with format userid_index,projectid_index,timestamp_project

user_pro_data = user_pro_data[np.isfinite(user_pro_data['timestamp'])]
print len(user_pro_data)
print 'Total unique projects before filtering : %d'%(len(user_pro_data.pid.unique()))
print 'Total unique backers before filtering : %d'%(len(user_pro_data.bid.unique()))
user_pro_data, user_activities, project_popularity = filter(user_pro_data)
print len(user_pro_data)
# print user_pro_data
user_pro_data['index'] = range(0, user_pro_data.shape[0])
if DEBUG_MODE:
    #take first 360 rows of the data to use
    user_pro_data = user_pro_data[:360]
    print user_pro_data
#reindex user-id and project-id as we removed some of them.
pid_ts_tmp = user_pro_data[['pid','timestamp','timestamp_end','category']]
if REVERSE:
    pid_ts_tmp = pid_ts_tmp.sort_index(by='timestamp', ascending=False) #latest project will have new pid of 0
else:
    pid_ts_tmp = pid_ts_tmp.sort_index(by='timestamp', ascending=True) #latest project will have largest pid
pid_ts_tmp = pid_ts_tmp[['pid']]
pid_ts_tmp = pid_ts_tmp.drop_duplicates(cols='pid')
pid_ts_tmp['new_pid'] = range(0, pid_ts_tmp.shape[0])

bid_pid_tmp = user_pro_data[['bid']]
bid_pid_tmp = bid_pid_tmp.drop_duplicates(cols='bid')
bid_pid_tmp['new_bid'] = range(0, bid_pid_tmp.shape[0])

# user_pro_data = user_pro_data.join(pid_ts_tmp, on = 'pid')
user_pro_data = pd.merge(user_pro_data, pid_ts_tmp, on = 'pid', how='left')
# print user_pro_data
# user_pro_data = user_pro_data.join(bid_pid_tmp, on = 'bid')
# user_pro_data = pd.merge(user_pro_data, bid_pid_tmp, on = 'bid').sort_index(by='new_bid', ascending=True)
user_pro_data = pd.merge(user_pro_data, bid_pid_tmp, on = 'bid', how='left').sort_index(by='index', ascending=True)
# print user_pro_data
user_pro_data = user_pro_data[['new_bid', 'new_pid', 'timestamp','timestamp_end','category']]
user_pro_data.columns = ['bid', 'pid', 'timestamp', 'timestamp_end', 'category']
print user_pro_data
#check if reindex works:
bids = user_pro_data.bid.unique()
assert np.max(bids) == (len(bids) - 1)
pids = user_pro_data.pid.unique()
assert np.max(pids) == (len(pids) - 1)
print len(bids)
print len(pids)

if not DEBUG_MODE:
    #re-computing the user-activities and project-popularity:
    print len(user_pro_data)
    user_pro_data, user_activities, project_popularity = filter(user_pro_data)
    print len(user_pro_data)

    unique_uid = user_activities.index #get all unique user ids
    unique_pid = project_popularity.index # get all unique item ids

DATA_DIR = 'data/rec_data/'
CONTAINED_DIR = "all"
if DEBUG_MODE:
    CONTAINED_DIR = 'debug'
    unique_uid = user_pro_data.bid.unique()
    unique_pid = user_pro_data.pid.unique()



with open(os.path.join(DATA_DIR, CONTAINED_DIR, 'unique_uid_cate.txt'), 'w') as f:
    for uid in unique_uid:
        f.write('%s\n' % uid)
with open(os.path.join(DATA_DIR, CONTAINED_DIR, 'unique_pid_cate.txt'), 'w') as f:
    for pid in unique_pid:
        f.write('%s\n' % pid)
#split into training, validation and testing data with ratio 7:1:2
BOOLEAN_SPLIT_DATA = True


def _split_batch(min_bid, max_bid, data):
    first_try = True
    i = 0
    train_data, validation_data, test_data = None, None, None
    for bid in range(min_bid, max_bid + 1):
        tmp = data[data['bid'] == bid]
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
    print 'Total bid: %d'%(len(bids))
    print 'Total pid: %d'%(len(pids))
    print bids[0:10]
    min_bid = np.min(bids)
    max_bid = np.max(bids)


    train_data, validation_data, test_data = _split_batch(min_bid, max_bid, user_pro_data)
    print len(train_data)
    print len(validation_data)
    print len(test_data)

    train_data.to_csv(os.path.join(DATA_DIR, CONTAINED_DIR, 'train-cate.csv'), index=False)
    validation_data.to_csv(os.path.join(DATA_DIR, CONTAINED_DIR, 'validation-cate.csv'), index=False)
    test_data.to_csv(os.path.join(DATA_DIR, CONTAINED_DIR, 'test-cate.csv'), index=False)

########PREPARE the SPPMI matrix for projec-project co-occurrence using user context ###################################
#each user will back some projects. We see all the projects backed by a user as a sentence

#######################################################################################################################

########PREPARE the SPPMI matrix for user-user co-occurrence using project context ####################################
#each project will be backed by some users. We see all the users backed the same project as a sentence

#######################################################################################################################
t2 = time.time()
print 'Time : %d seconds'%(t2 - t1)
sys.path.remove(r'/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/utils')
sys.path.remove(r'/home/thanh/PenStateCollaboration2016/python/KickstarterRec/utils')
sys.path.remove(r'/home/thanh/PenStateCollaboration2016/python/KickstarterRec')
