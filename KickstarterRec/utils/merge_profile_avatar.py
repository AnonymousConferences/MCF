import string
data_file1   = 'data/profile-avatar/user_demographics_indiegogo_face++.csv'
output_file1 = 'data/profile-avatar/final_user_demographics_indiegogo.csv'
data_file2   = 'data/profile-avatar/user_demographics_kickstarter_ms.csv'
output_file2 = 'data/profile-avatar/final_user_demographics_kickstarter.csv'

class UserDemographicInfo:
	def __init__(self, link, gender_ms, gender_facepp, age_ms, age_facepp, age_facepp_range):
		self.link = link
		self.gender_ms = gender_ms
		self.gender_facepp = gender_facepp
		self.age_ms = age_ms
		self.age_facepp = age_facepp
		self.age_facepp_range = age_facepp_range
		self.gender_final = ''
		self.age_final = 0	
	def to_string(self):
		SEPARATOR  = ','
		text  = self.link			+ SEPARATOR 
		text +=	self.gender_ms  		+ SEPARATOR
		text +=	str(self.age_ms)		+ SEPARATOR
		text +=	self.gender_facepp 		+ SEPARATOR
		text +=	str(self.age_facepp)		+ SEPARATOR
		text +=	str(self.age_facepp_range)	+ SEPARATOR
		text +=	self.gender_final		+ SEPARATOR 
		text +=	str(self.age_final)
		return text

def read_data_file(input_file, source = 'indiegogo'):
	f = open(input_file, 'r')
	list_user_demographic_objs = []
	for line in f:
		tokens = line.replace('\n','').split(',')
		if len(tokens) != 7:
			continue
		link = tokens[0]
		gender_ms, gender_facepp, age_ms, age_facepp, age_facepp_range = '','',0,0,0
		if source == 'indiegogo':
			age_ms = float(tokens[1])
			gender_ms = tokens[2]
			age_facepp = int(tokens[3])
			age_facepp_range = int(tokens[4])
			gender_facepp = tokens[5]
		else:
			age_ms = float(tokens[2])
                        gender_ms = tokens[1]
                        age_facepp = int(tokens[3])
                        age_facepp_range = int(tokens[4])
                        gender_facepp = tokens[5]
		user_demographic_obj = UserDemographicInfo(link, gender_ms, gender_facepp, age_ms, age_facepp, age_facepp_range)
		list_user_demographic_objs.append(user_demographic_obj)
	f.close()
	return list_user_demographic_objs
def merge_data(list_user_demographic_objs):
	merged_user_obj_lst = []
	for user_obj in list_user_demographic_objs:
		if abs(user_obj.age_ms - user_obj.age_facepp) <= 5:
			user_obj.age_final = round((user_obj.age_ms + user_obj.age_facepp)/2)
		elif abs(user_obj.age_ms - (user_obj.age_facepp - user_obj.age_facepp_range)) <= 5:
			user_obj.age_final = round((user_obj.age_ms + user_obj.age_facepp - user_obj.age_facepp_range)/2)
		elif abs(user_obj.age_ms - (user_obj.age_facepp + user_obj.age_facepp_range)) <= 5:
			user_obj.age_final = round((user_obj.age_ms + user_obj.age_facepp + user_obj.age_facepp_range)/2)
		if user_obj.gender_ms.lower() == user_obj.gender_facepp.lower():
			user_obj.gender_final = user_obj.gender_facepp
		#filter:
		if user_obj.age_final != 0 and user_obj.gender_final != '':
			merged_user_obj_lst.append(user_obj)
	return merged_user_obj_lst
def flush_to_file(merged_user_obj_lst, output_file):
	print 'flushing %d users to file:'%(len(merged_user_obj_lst))
	f = open(output_file, 'w')
	for user_obj in merged_user_obj_lst:
		text = user_obj.to_string()
		f.write(text + '\n')
	f.flush()
	f.close()

source = 'indiegogo' # 'kickstarter'
list_user_demographic_objs 	= read_data_file(data_file1, source)
merged_user_obj_lst 		= merge_data(list_user_demographic_objs)
flush_to_file(merged_user_obj_lst, output_file1)

source = 'kickstarter'
list_user_demographic_objs     = read_data_file(data_file2, source)
merged_user_obj_lst            = merge_data(list_user_demographic_objs)
flush_to_file(merged_user_obj_lst, output_file2)

	
