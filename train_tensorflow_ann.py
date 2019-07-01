#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
import pandas as pd
import numpy as np
import pygridgen
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import random


%matplotlib inline

loss_limit = 2.00#?
adjacency_num = 9

os.chdir('./shp/train_data')
lst = os.listdir(os.getcwd())
filename = []

for c in lst:
    if os.path.isfile(c) and c.endswith('.csv'):# and c.find("test") == -1
        filename.append(c)
     
        

def judge_inner_point(nvert, vertx, verty, testx, testy):
    #PNPoly algorithm (judge whether a point is in a given polygon)
    #nvert : the number of the polygon's vertex
    #vertx(y) : coordinate of the polygon
    #testx(y) : coordinate of the test point
    
    i, j ,c = 0,nvert-1,False
    for i in range(nvert):
        P1 = ((verty[i]>testy) != (verty[j]>testy))
        P2 = (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) /(verty[j]-verty[i]+0.0000001) + vertx[i])
        if P1 & P2:
            c = not c
        j = i
        #print(P1,P2,c)
    return c
      
def adjacency(lat, lon, len_num):
    #return (train) length & angle
    length = []
    angle = []

    #length 
    for i in range(len_num-1):
        LAT = (lat[i]-lat[i+1])**2
        LON = (lon[i]-lon[i+1])**2
        length.append(np.sqrt(LAT + LON) )
    LAT = (lat[0]-lat[-1])**2
    LON = (lon[0]-lon[-1])**2
    length.append(np.sqrt(LAT + LON))
    
    length = (np.array(length)-np.mean(length)) / (np.std(length)+0.00000001)

    #angle
    for i in range(len_num-1):
        v1 = (lat[i+1] - lat[i] , lon[i+1] - lon[i])
        v2 = (lat[i-1] - lat[i] , lon[i-1] - lon[i])
        inner = v1[0]*v2[0] + v1[1]*v2[1]
        
        a1 = np.sqrt(v1[0]**2 + v1[1]**2)
        a2 = np.sqrt(v2[0]**2 + v2[1]**2)
        if judge_inner_point(len_num, lat, lon, (lat[i+1]+lat[i-1])/2, (lon[i+1]+lon[i-1])/2 ):
            angle.append(np.arccos(inner/(a1*a2))/np.pi*180 )
        else:
            angle.append(360-np.arccos(inner/(a1*a2))/np.pi*180)
            
    v1 = (lat[0] - lat[-1] , lon[0] - lon[-1])
    v2 = (lat[-2] - lat[-1] , lon[-2] - lon[-1])
    inner = v1[0]*v2[0] + v1[1]*v2[1]
    
    a1 = np.sqrt(v1[0]**2 + v1[1]**2)
    a2 = np.sqrt(v2[0]**2 + v2[1]**2)
    if judge_inner_point(len_num, lat, lon, (lat[0]+lat[-2])/2, (lon[0]+lon[-2])/2 ):
        angle.append(np.arccos(inner/(a1*a2))/np.pi*180 )
    else:
        angle.append(360-np.arccos(inner/(a1*a2))/np.pi*180)  
    
    angle = np.array(angle)/360
    
    return angle,length


Input = []
Output = []
num = -1
for file in filename:
    num += 1
    f = open(file)
    df = pd.read_csv(f)
    data = df.values[:,1:]
    
    len_num = int(data[0][-1])
    temp = data[0][:2*len_num].reshape(-1,2)
    lat = temp[:,0]
    lon = temp[:,1]       
    corner = data[0][2*len_num:-2]
    
    
    #length and angle
    file_angle ,file_length = adjacency(lat, lon, len_num)
    
    for i in range(len_num):
        #length + angle -->  input
        #corner --> output
        temp_input = []
        temp_output = []
        
        #input length
        for j in range(int(adjacency_num/2)):
            temp_input.append(file_length[(i+j)%len_num])
            temp_input.append(file_length[(i-j-1)%len_num])
            
        #input angle
        temp_input.append(file_angle[i])
        for j in range(int((adjacency_num-3)/2)):
            temp_input.append(file_angle[(i+j+1)%len_num])
            temp_input.append(file_angle[(i-j-1)%len_num])
        
        temp_output.append(corner[i])
        
        Input.append(temp_input)
        Output.append(temp_output)
            
    for i in range(len_num):
        if corner[i] == 1:
            #length + angle -->  input
            #corner --> output
            temp_input = []
            temp_output = []
            
            #input length -1
            for j in range(int(adjacency_num/2)):
                temp_input.append(file_length[(i+j)%len_num])
                temp_input.append(file_length[(i-j-1)%len_num])
                
                
            #input angle -1 
            temp_input.append(1-file_angle[i])
            for j in range(int((adjacency_num-3)/2)):
                temp_input.append(1-file_angle[(i+j+1)%len_num])
                temp_input.append(1-file_angle[(i-j-1)%len_num])
                
    
            temp_output.append(-1)
            
            Input.append(temp_input)
            Output.append(temp_output)
        else:
            continue
    
#ANN train________________________________________________________________________________________

temp_total = np.column_stack((np.array(Input),np.array(Output)))
total_1 = np.zeros((1,len(temp_total[0])))
total_0 = np.zeros((1,len(temp_total[0])))
total_f1 = np.zeros((1,len(temp_total[0])))

#for i in range(1,len(temp_total)):
#    if temp_total[i][-1] in [1]:
#        total_1 = np.row_stack((total_1,temp_total[i]))
#        continue
#    elif temp_total[i][-1] in [0]:
#        total_0 = np.row_stack((total_0,temp_total[i]))
#        continue
#    elif temp_total[i][-1] in [-1]:
#        total_f1 = np.row_stack((total_f1,temp_total[i]))
#        continue
#            
#total = np.row_stack((total_f1,total_1[:len(total_f1),:]))
##total = np.row_stack((total,total_0[:len(total_f1),:])) 为何不一样？

total = temp_total[0]
for i in range(1,len(temp_total)):#len(temp_total)
    if temp_total[i][-1] in [1,-1,0]:
        total = np.row_stack((total,temp_total[i]))
        continue
 
total = np.delete(total, 0, 0)
random.shuffle(total)
TEMP_X = total[4:,:-1]
TEMP_y = LabelBinarizer().fit_transform(total[4:,-1])

#
#TEMP_X = np.array(Input)
#TEMP_y = LabelBinarizer().fit_transform(np.array(Output))
rat = 0.8
rat_num = int(len(total)*rat)
X = TEMP_X[:rat_num,:]
y = TEMP_y[:rat_num,:]
TEST_X = TEMP_X[rat_num:,:]
TEST_y = TEMP_y[rat_num:,:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.01)


def add_layer(inputs, in_size, out_size, activation_function=None, ):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs  
    
    
    

# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)

input_size = int(adjacency_num-1 + adjacency_num-2)# length  +   angle
xs = tf.placeholder(tf.float32, [None, input_size])  
ys = tf.placeholder(tf.float32, [None, 3])

# add output layer
l1 = add_layer(xs, input_size, 8, activation_function=tf.nn.tanh)
#l2 = add_layer(l1, 10, 8, activation_function=tf.nn.tanh)
#l3 = add_layer(l2, 8, 6, activation_function=tf.nn.tanh)
prediction = add_layer(l1, 8, 3, activation_function=tf.nn.softmax)

# the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1])) # loss
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#AdamOptimizer


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    
    saver = tf.train.Saver()
    tf.add_to_collection('X', xs)
    tf.add_to_collection('keep_prob', keep_prob)
    tf.add_to_collection('pred', prediction)
    
    
    for i in range(50000):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.9})
        if i%1000==0:
            print(sess.run(cross_entropy,feed_dict={xs: X_test, ys: y_test, keep_prob: 1}))

    model_dir = "corner_ann_model"
    model_name = 'corner_model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # save model
    saver.save(sess, os.path.join(model_dir, model_name))
    print("successful！")   
    
    
    
    
    
    
    

#__________________________________________________________________-



#test
sess = tf.Session()

model_dir = "corner_ann_model"
model_name = 'corner_model'

new_saver = tf.train.import_meta_graph(model_dir+'/'+model_name+'.meta')
new_saver.restore(sess, model_dir+'/'+model_name)

X = tf.get_collection('X')[0]
keep_prob = tf.get_collection('keep_prob')[0]
pred = tf.get_collection('pred')[0]
print("恢复模型成功！")
    
def check2(predict):
    for i in range(len(predict)):
        if predict[i] == max(predict):
            break
    if i == 0:
        return True
    else:
        return False


def check(predict,true):
    for i in range(len(predict)):
        if predict[i] == max(predict):
            break
    for j in range(len(true)):
        if true[j] == max(true):
            break
    if i == j:
        return True
    else:
        return False

test_predict=[]
num = 0
hh_num = 0
for step in range(len(TEST_X)):
    prob=sess.run(pred,feed_dict={X:[TEST_X[step]],keep_prob:1})   
    predict=prob[0]
    if check2(predict):
        hh_num += 1
    if check(predict,TEST_y[step]):
        num += 1
print(num/len(TEST_X))
print(hh_num/len(TEST_X))




#
#
#
#
#
##
##
##
###_________________________________________________________________________-
##
#from sklearn.linear_model.logistic import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
#
#temp_total = np.column_stack((np.array(Input),np.array(Output)))
#total_1 = np.zeros((1,len(temp_total[0])))
#total_0 = np.zeros((1,len(temp_total[0])))
#total_f1 = np.zeros((1,len(temp_total[0])))
#
#for i in range(1,len(temp_total)):
#    if temp_total[i][-1] in [1]:
#        total_1 = np.row_stack((total_1,temp_total[i]))
#        continue
#    elif temp_total[i][-1] in [0]:
#        total_0 = np.row_stack((total_0,temp_total[i]))
#        continue
#    elif temp_total[i][-1] in [-1]:
#        total_f1 = np.row_stack((total_f1,temp_total[i]))
#        continue
#            
#total = np.row_stack((total_f1,total_1[:len(total_f1),:]))
#total = np.row_stack((total,total_0[:len(total_f1),:]))
#
#random.shuffle(total)
#TEMP_X = total[4:,:-1]
#TEMP_y = total[4:,-1].reshape(-1,1)#LabelBinarizer().fit_transform(total[:,-1])
#
##
##TEMP_X = np.array(Input)
##TEMP_y = LabelBinarizer().fit_transform(np.array(Output))
#rat = 0.8
#rat_num = int(len(total)*rat)
#X = TEMP_X[:rat_num,:]
#y = TEMP_y[:rat_num,:]
#TEST_X = TEMP_X[rat_num:,:]
#TEST_y = TEMP_y[rat_num:,:]
#
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
#
#
#
#
#classifier=LogisticRegression(class_weight='balanced')
#classifier.fit(X_train,y_train)
#predictions=classifier.predict(TEST_X)
#temp =TEST_y.reshape(-1,)
#
#print(np.sum((predictions -temp)==0)/len(temp))
#
