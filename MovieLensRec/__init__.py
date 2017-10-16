import sys
from backer_loader import load_ks_backers
import constants
lst_backers = load_ks_backers()
print 'Total backers: %d'%(len(lst_backers))
fw = open(constants.PREDICTION_FILE_OUTPUT, 'w')

header = 'num_backed_projects,' \
         'backed_category_art,backed_category_comics,backed_category_crafts,backed_category_dance,backed_category_design,' \
         'backed_category_fashion,backed_category_film&video,backed_category_food,backed_category_games,backed_category_journalism,' \
         'backed_category_music, backed_category_photography,backed_category_publishing,backed_category_technology,backed_category_theater,' \
         'backed_entropy,' \
         'avg_backed_project_success, avg_backed_project_goal, avg_backed_project_pledged_money, avg_backed_project_backers, num_backed_project_success,' \
         'num_comments, first_backed_project_interval'
header += ',' + 'mean_backed_series, std_backed_series, min_backed_series, max_backed_series, median_backed_series'
header += ',' + 'num_backed_projects_1month_after_joined_date, num_backed_projects_2weeks_after_joined_date' \
          ','  + 'num_backed_projects_1week_after_joined_date, num_backed_projects_1day_after_joined_date'
# for i in range(1, constants.NUM_DAYS_SERIES + 1):
#     feature_name = "num_backed_series_day_" + str(i)
#     header += ',' + feature_name


header += ',' + 'cluster_id'
fw.write(header + '\n')
for backer in lst_backers:
    line = backer.to_string()
    tokens = line.split(',')
    if len(tokens) != 29:
        print line

    fw.write(line + '\n')
fw.flush()
fw.close()
