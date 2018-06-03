import numpy as np
import os
import pickle
import time
import sys
import scipy
import random
import shutil

def convertToDictionary(dataset):
        return_dict = {}
        for value in dataset:
                user_id = value[0] + value[3]
                side = int(value[1])
                progress = int(value[2])
                return_array = [side, progress]
                dictionary_value = return_dict.get(user_id, None)
                if (dictionary_value == None):
                        return_dict[user_id] = return_array
                else:
                        return_dict[user_id] = [dictionary_value, return_array]
        return return_dict

def convertToDictionary1(dataset):
        return_dict = {}
        for value in dataset:
                user_id = value[0]
                side = value[1]
                side_string = value[2]
                return_array = [side, side_string]
                dictionary_value = return_dict.get(user_id, None)
                if (dictionary_value == None):
                        return_dict[user_id] = return_array
                else:
                        return_dict[user_id] = [dictionary_value, return_array]
        return return_dict

label_file = sys.argv[1]

filename = label_file
fileObject = open(filename,'rb')
label_data = pickle.load(fileObject)

label_dict = convertToDictionary(label_data)

count = 0
source_dir = '/home/npandit/project/kneeoa/kneeimage'
p_dest = '/home/npandit/project/kneeoa/progressimage'
np_dest = '/home/npandit/project/kneeoa/nonprogressimage'

for value in label_data:
	user_id = value[0] + value[3]
	line = source_dir + "/" + user_id + ".png"
        print line
	print label_dict[user_id][1]
	if (label_dict[user_id][1] == 1) :
		shutil.move(line, p_dest)
	else :
		shutil.move(line, np_dest)	
        count = count + 1

print ("Moved:" + str(count) + " files to train")




