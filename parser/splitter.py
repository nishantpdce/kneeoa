import numpy as np
import os
import pickle
import time
import sys
import scipy
import random
import shutil

train_file = sys.argv[1]
test_file = sys.argv[2]

filename = train_file
fileObject = open(filename,'r')
train_labels = pickle.load(fileObject)

filename = test_file
fileObject = open(filename,'r')
test_labels = pickle.load(fileObject)
print test_labels[0][0]
print "test_labels",len(test_labels)


count = 0
source_dir = '/home/npandit/project/kneeimagedata/kneeimagenew'
train_dest = '/home/npandit/project/kneeoa/DeepLearning/train_image'
test_dest = '/home/npandit/project/kneeoa/DeepLearning/test_image'

for value in train_labels:
	line = source_dir + "/" + value[0] + value[3] + ".png"
        print line
        shutil.copy(line, train_dest)
        count = count + 1
print ("Moved:" + str(count) + " files to train")

count = 0
for value in test_labels:
        line = source_dir + "/" + value[0] + value[3] + ".png"
        print line
        shutil.copy(line, test_dest)
        count = count + 1
print ("Moved:" + str(count) + " files to test")



