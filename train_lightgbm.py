#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# coding: utf-8
# pylint: disable = invalid-name, C0111
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

import os 
import pygridgen
import matplotlib.pyplot as plt
import random
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--adj_num",      nargs='?',   type=int,    help=" argument", default=20)
parser.add_argument("--From",        nargs='?',   type=int,    help=" argument", default=30)
parser.add_argument("--To",        nargs='?',   type=int,    help=" argument", default=40)

args = parser.parse_args()

loss_limit = 2.00
adjacency_num = args.adj_num
FROM = args.From
TO = args.To

os.chdir('./shp/train_data_'+str(FROM)+'_to_'+str(TO))
#os.chdir('./train_data')
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
        P2 = (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) /(verty[j]-verty[i]+0.0000000001) + vertx[i])
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
        
#        #input length
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


#
total = temp_total[0]
for i in range(1,len(temp_total)):#len(temp_total)
    if temp_total[i][-1] in [1,-1,0]:
        total = np.row_stack((total,temp_total[i]))
        continue
 
total = np.delete(total, 0, 0)
random.shuffle(total)



TEMP_X = total[:, :-1].astype(float)
TEMP_y = total[:, -1]+1

rat = 0.8
rat_num = int(len(total)*rat)
X = TEMP_X[:rat_num,:]
y = TEMP_y[:rat_num]
TEST_X = TEMP_X[rat_num:,:]
TEST_y = TEMP_y[rat_num:]




X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)





params = {
    'task' : 'train',
    'boosting_type' : 'gbdt',
    'objective' : 'multiclass',
    'metric' : {'multi_logloss'},
    'num_leaves' : 63,
    'learning_rate' : 0.01,
    'feature_fraction' : 0.9,
    'bagging_fraction' : 0.9,
    'bagging_freq': 0,
    'verbose' : 1,
    'num_class' : 3
}
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)


#
preds = gbm.predict(TEST_X)
predictions = []

for x in preds:
    predictions.append(np.argmax(x))

temp = np.array(predictions)-TEST_y
print(np.sum(temp == 0)/len(temp))


#
print('Save model...')
# save model to file
#gbm.save_model('model.txt')



# save model
joblib.dump(gbm, 'lgb_model_'+str(TO)+'corenr_'+str(adjacency_num)+'adj.pkl')
## load model
#gbm_pickle = joblib.load('lgb_model.pkl')
#
#print('Start predicting...')
## predict
#y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
## eval
#print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)











#
#import matplotlib.pyplot as plt
#
#for i in range(len(Input[0])):
#    temp = []
#    for inp in Input:
#        temp.append(inp[i])
#    plt.hist(temp,500,normed=True)
#    plt.savefig('./a/angle15/angle_15_'+str(i)+".png")
#    plt.show()
#
#    
#    
#for i in range(len(Input[0])):
#    temp = []
#    for inp in Input:
#        temp.append(inp[i])
#    plt.hist(temp,500)
#    plt.savefig('./a/length10/length_10_'+str(i)+".png")
#    plt.show()