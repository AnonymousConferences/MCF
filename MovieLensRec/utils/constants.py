#backer_id,
#art,comics,crafts,dance,design,fashion,filmvideo,food,games,music,journalism,photography,publishing,technology,theater,
#totalBacked,entropy,gender,gender_confidence,age,age_range,join_duration
BACKER_INFO_PATH = '/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/data/profile-avatar/backer/all_backer_info.csv'

#backer_url, project_url1, project_url2,...
BACKER_BACKED_PROJECTS_PATH = '/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/data/user_backedprojects_withurl.csv'
#backer_id  \t  label
BACKER_CLUSTER_LABELS = '/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/data/clustering/kickstarter_mse_labels.csv'
#
BACKER_COMMENTS_PATH = '/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/data/user-comments/'
BACKER_JOINED_DATE_LOCATION_INFO_PATH = '/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/data/profile-avatar/backer/backer_location_info.csv'

#project_id \t creator_id \t goal \t pledged_money \t is_success \t num_backers \t category \t location
PROJECT_INFO_PATH = '/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/data/project-info/project_info_tabseparation.csv'
PROJECT_INFO_MYSQL_MORE_PATH = '/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/data/project-info/project_info_mysql_tabseparation.csv.txt'

#project-id      funding-from    funding-to
PROJECT_DURATION_PATH = '/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/data/project-info/project-fundingduration.csv'
PROJECT_DURATION_MORE = '/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/data/project-info/project-fundingduration_mysql_more.csv'

OUTPUT_FINAL_PROJECT_INFO_PATH = '/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/data/kickstarter_v2/project_info.csv'
OUTPUT_FINAL_BACKER_BACKED_PROJECTS_PATH = '/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/data/kickstarter_v2/user_backed_projects_full.csv'
OUTPUT_FINAL_BACKER_BASIC_INFO_PATH = '/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/data/kickstarter_v2/backer_basic_info.csv'

PREDICTION_FILE_OUTPUT = '/media/thanhtd/Data/thanh-repo/PenStateCollaboration2016/python/PenRecProject/data/prediction/backer_cluster_prediction_v2_cleaned.csv'

#####################################################################################################################################################################
NUM_DAYS_SERIES = 30