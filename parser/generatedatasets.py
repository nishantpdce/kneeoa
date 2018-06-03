import numpy as np
import os
import pickle
import time
import sys
import scipy
import random

# Read in Total Data Set
def read_file(filename):
	file = open(filename, 'r')
	total_dataset = []
	for line in file:
		data = line
		data = data.split(',')
		total_dataset.append(data)
	return total_dataset

def shuffle_dataset(dataset):
	user_dataset = []
	for data in dataset:
		user_id = data[0]
		user_side = data[1]
		user_progress = data[2]
		user_side_string = data[len(data)-1][:-2]
		user_array = [user_id, user_side, user_progress, user_side_string]
		user_dataset.append(user_array)
	random.shuffle(user_dataset)
	return user_dataset
	

kl2ormore = read_file('KL2ormore.txt')
progressors = read_file('progressors.csv')
non_progressors = read_file('non_progressors.csv')
progressors_shuffled = shuffle_dataset(progressors)
non_progressors_shuffled = shuffle_dataset(non_progressors)


def splitData(shuffled_dataset, train_percentage, test_percentage, dev_percentage):
	total_prob = train_percentage + test_percentage + dev_percentage
	if (total_prob - 1 >= .01):
		print("Error")
		return
	total_num = len(shuffled_dataset)
	train_index = int(total_num * train_percentage)
	test_start_index = train_index
	test_end_index = test_start_index + int(total_num * test_percentage)
	dev_start_index = test_end_index
	dev_end_index = total_num
	train_set = shuffled_dataset[0: train_index]
	test_set = shuffled_dataset[test_start_index:test_end_index]
	dev_set = shuffled_dataset[dev_start_index:dev_end_index]
	return train_set, test_set, dev_set

progressors = os.listdir(os.getcwd()+"/"+sys.argv[1])

train_progressors, test_progressors, dev_progressors = splitData(progressors_shuffled, 0.8, 0.2, 0.0)
train_non_progressors, test_non_progressors, dev_non_progressors = splitData(non_progressors_shuffled, 0.8, 0.2, 0.0)

train_data = train_progressors + train_non_progressors
test_data = test_progressors + test_non_progressors

random.shuffle(train_data)
random.shuffle(test_data)

file_Name1 = os.getcwd()+"/"+sys.argv[1]
# open the file for writing
fileObject1 = open(file_Name1,'wb')

# this writes the object a to the
# file named 'testfile'
pickle.dump(train_data,fileObject1)

# here we close the fileObject                          
fileObject1.close()

file_Name2 = os.getcwd()+"/"+sys.argv[2]
# open the file for writing
fileObject2 = open(file_Name2,'wb')

# this writes the object a to the
# file named 'testfile'
pickle.dump(test_data,fileObject2)

# here we close the fileObject                          
fileObject2.close()

