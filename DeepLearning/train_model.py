import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import pickle
import time
import sys

print sys.argv
training_folder = sys.argv[1]
train_labels = sys.argv[3]
mode_folder = sys.argv[4]
batch = 20
beta = 0.01
training_folder_len = len([name for name in os.listdir(os.getcwd()+"/"+training_folder)])


filename = train_labels
fileObject = open(filename,'r')
train_labels = pickle.load(fileObject)
print ("train_labels",len(train_labels))

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

train_labels_dict = convertToDictionary(train_labels)

arr_train = os.listdir(os.getcwd()+"/"+sys.argv[1])

n_input = 25088
# The number of classes which the ConvNet has to classify into .
n_classes = 1
# The number of neurons in the each Hidden Layer .
n_hidden1 = 512
n_hidden2 = 512

epsilon = 1e-3
g2 = tf.Graph()
with g2.as_default():

    # Tensorflow Graph input .
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    train_label = y
    with tf.name_scope('layer1'):
        W_1 = tf.get_variable(
                    name="W1",
                    shape=[n_input, n_hidden1],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())

        #b_1 = tf.get_variable(
        #    name='b1',
        #    shape=[n_hidden1],
        #    dtype=tf.float32,
        #    initializer=tf.constant_initializer(0.0))

        z1_BN = tf.matmul(x, W_1)
        batch_mean1, batch_var1 = tf.nn.moments(z1_BN,[0])
        scale1 = tf.Variable(tf.ones([n_hidden1]))
        beta1 = tf.Variable(tf.zeros([n_hidden1]))
        BN1 = tf.nn.batch_normalization(z1_BN,batch_mean1,batch_var1,beta1,scale1,epsilon)

        #h_1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, W_1),b_1))
        h_1 = tf.nn.relu(BN1)
        keep_prob = tf.placeholder("float")
        h_1d = tf.nn.dropout(h_1, keep_prob)
    with tf.name_scope('layer2'):
        W_2 = tf.get_variable(
                    name="W2",
                    shape=[n_hidden1,n_hidden2],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())

        #b_2 = tf.get_variable(
        #    name='b2',
        #    shape=[n_hidden2],
        #    dtype=tf.float32,
        #    initializer=tf.constant_initializer(0.0))

        z2_BN = tf.matmul(h_1d, W_2)
        batch_mean2, batch_var2 = tf.nn.moments(z2_BN,[0])
        scale2 = tf.Variable(tf.ones([n_hidden2]))
        beta2 = tf.Variable(tf.zeros([n_hidden2]))
        BN2 = tf.nn.batch_normalization(z2_BN,batch_mean2,batch_var2,beta2,scale2,epsilon)

        #h_2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_1, W_2),b_2))
        h_2 = tf.nn.relu(BN2)
        h_2d = tf.nn.dropout(h_2, keep_prob)
    with tf.name_scope('output'):
        W_3 = tf.get_variable(
                   name="W3",
                   shape=[n_hidden2,n_classes],
                   dtype=tf.float32,
                   initializer=tf.contrib.layers.xavier_initializer())

        b_3 = tf.get_variable(
           name='b3',
           shape=[n_classes],
           dtype=tf.float32,
           initializer=tf.constant_initializer(0.0))

        h_3 = tf.nn.bias_add(tf.matmul(h_2d, W_3),b_3)
        a_3 =  tf.nn.sigmoid(h_3)
    # h_3 = tf.nn.softmax(h_3)
    # h_3 = h_3

    Cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = h_3, labels = y))
    #regularizer = tf.nn.l2_loss(W_1) + tf.nn.l2_loss(W_2) + tf.nn.l2_loss(W_3)
    #Cost = tf.reduce_mean(Cost1 + beta * regularizer)
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(Cost)
    #optimizer = tf.train.AdamOptimizer(0.01).minimize(Cost)

    # saver = tf.train.Saver()
    #Monitor accuracy
    predicted_y = tf.nn.sigmoid(h_3)
    #predicted_y = tf.argmax(tf.nn.sigmoid(h_3), 1)
    actual_y = y 
    #tf.argmax(y, 1)
    print ("predicted",predicted_y)
    print ("actual", actual_y)

    correct_prediction = tf.equal(predicted_y, actual_y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


    # names = [op.name for op in g2.get_operations()]
    # print names
    saver = tf.train.Saver(max_to_keep=15)


r = (training_folder_len - (training_folder_len%batch))+1
print (r)

with tf.Session(graph=g2) as sess2:
# sess =  tf.Session(graph=g2)

    sess2.run(tf.global_variables_initializer())
    # saver.save(sess2, 'my-model')

    accuracy_list=[]
    cost=[]
    cost1 = np.empty(20, dtype=float)
    n_epochs = 20
    for epoch in range(n_epochs):

        start_time = time.time()

        for j in range(0,r,batch):

            file_Name =  os.getcwd()+"/"+sys.argv[2]+"/"+ str(j)
            fileObject = open(file_Name,'r')
            # load the object from the file into var b
            content_features = pickle.load(fileObject)

            # print type(content_features)
            content_features = content_features.reshape((content_features.shape[0],7*7*512))
            # content_features = content_features.reshape((content_features.shape[0],28*28*256))
            print (content_features.shape , "Feature Map Shape")

            print ("j=",j)

            label = np.zeros((batch,1))
            #print "label shape", label.shape
            idx = 0
            if j==r-1:
                #label = train_labels[j:]
                filename = arr_train[j:]
                for var in filename:
                    label[idx][0] = train_labels_dict[var[0:9]][1]
                    idx = idx + 1
                label = label[:idx]
                print (label.shape)
            else:
                #label = train_labels[j+0:j+20]
                filename = arr_train[j+0:j+batch]
                for var in filename:
                        label[idx][0] = train_labels_dict[var[0:9]][1]
                        idx = idx + 1
                print (label.shape)

            _,l,w1,cst,a3_out = sess2.run([optimizer,train_label,W_1,Cost,a_3], feed_dict={x: content_features, y:label, keep_prob : 0.5})


            if j % 100==0:
                print (" Epoch="+str(epoch),"j="+str(j))
            #     accuracy_list.append(acc)
                cost.append(cst)

                print ("COST",cost)
            if j == 100:
                cost1[epoch] = cst 

        print("--- %s seconds ---" % (time.time() - start_time))
        print ("Cost for some mini batch",cost1)
        j=0
        path_name = os.getcwd()+"/"+sys.argv[4]+"/"+"my-model-"+str(epoch)+".ckpt"
        save_path = saver.save(sess2, path_name)
        print (path_name,"saved")
        #

# python train_model.py <Training images folder> <Train images codes folder> <Training image labels file> <Folder to save models>
