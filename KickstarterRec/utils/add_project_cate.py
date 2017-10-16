SOURCE = 'data/project-info/project-fundingduration.csv'
NEWSOURCE = 'data/project-info/project-fundingduration-cate.csv'
CATE_INFO = 'data/project-info/project_info.csv'
EXTRA_INFO = 'data/project-info/all-data-mysql.csv'

fr = open(CATE_INFO, 'r')
key_set = set()
key_map = dict()
cate_dis = set()
for line in fr:
    line = line.replace('\n', '')
    tokens = line.split(',')
    project_id = tokens[0]
    cate = tokens[6].strip()
    if (cate.strip() == 'category'):
        #header
        continue
    cate_dis.add(cate)
    key_set.add(project_id)
    key_map[project_id] = cate
fr.close()
print sorted(cate_dis)

fr = open(EXTRA_INFO, 'r')
cate_dis.clear()
for line in fr:
    line = line.replace('\n', '')
    tokens = line.split(',')
    project_id = tokens[2]
    cate = tokens[3].split('-')[0].strip()
    if (cate == 'category'):
        #header
        continue
    if cate == 'film':
        cate = 'film&video'
    cate_dis.add(cate)
    if project_id not in key_set:
        key_set.add(project_id)
        key_map[project_id] = cate
fr.close()
print sorted(cate_dis)

fr = open(SOURCE, 'r')
newlines = []
miss = 0
extra_map = dict()
extra_map['backpacking-across-america'] = 'photography'
extra_map['drawing-for-dollars'] = 'art'
extra_map['fictionary'] = 'games'
extra_map['games-for-the-people'] = 'games'
extra_map['get-your-holiday-cards-support-projects-in-afric'] = 'photography'
extra_map['hero-trade-paperback-volume-3'] = 'comics'
extra_map['home-base-an-independent-puzzle-game'] = 'games'
extra_map['laugh-riot-the-comedy-improv-card-game-11-2009'] = 'games'
extra_map['murality-artists-go-to-sderot-and-kiryat-gat-with-0'] = 'art'
extra_map['operation-zulu'] = 'technology'
extra_map['project-nihon-sustainable-art-travel-0'] = 'art'
extra_map['the-ian-c-anderson-embarassment-project'] = 'music'
extra_map['the-p3-project-people-places-and-patterns'] = 'photography'
extra_map['video-chat-at-35000-feet'] = 'film&video'

cate_dis.clear()
for line in fr:
    line = line.replace('\n','')
    project_id = line.split('\t')[0]
    if project_id == 'project-id':
        #header:
        newline = line +'\t' + 'category'
        newlines.append(newline)
        continue
    if project_id in key_set:
        newline = line + '\t' + key_map[project_id]
        newlines.append(newline)
        cate_dis.add(key_map[project_id])
    else:
        if project_id in extra_map.keys():
            newline = line + '\t' + extra_map[project_id]
            newlines.append(newline)
            cate_dis.add(extra_map[project_id])
        else:
            miss += 1
            print 'not existing category of project :', project_id
fr.close()

print sorted(cate_dis)
print 'total missing category:', miss
fw = open(NEWSOURCE, 'w')
for line in newlines:
    fw.write(line + '\n')
fw.flush()
fw.close()