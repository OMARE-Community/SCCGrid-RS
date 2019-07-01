#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


import os 
import pygridgen
import matplotlib.pyplot as plt
import random


%matplotlib inline

loss_limit = 2.00#?
adjacency_num = 5

os.chdir('./sjp/train_data')
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


#
total = temp_total[0]
for i in range(1,len(temp_total)):#len(temp_total)
    if temp_total[i][-1] in [1,-1,0]:
        total = np.row_stack((total,temp_total[i]))
        continue
 
total = np.delete(total, 0, 0)

#
#total_1 = np.zeros((1,len(temp_total[0])))
#total_0 = np.zeros((1,len(temp_total[0])))
#total_f1 = np.zeros((1,len(temp_total[0])))
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


X = total[:, :-1].astype(float)
Y = total[:, -1]

# encode class values as integers
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(Y)
# convert integers to dummy variables (one hot encoding)
dummy_y = np_utils.to_categorical(encoded_Y)



input_size = int(adjacency_num-1 + adjacency_num-2)
# define model structure
def baseline_model():
    model = Sequential()
#    model.add(Dropout(0.2, input_shape=(input_size,)))
    model.add(Dense(output_dim=8, input_dim=input_size, activation='relu'))
#    model.add(Dense(output_dim=8, input_dim=15, activation='relu'))
    model.add(Dense(output_dim=3, input_dim=8, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=800, batch_size=1024)
# splitting data into training set and test set. If random_state is set to an integer, the split datasets are fixed.
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.3, random_state=0)
estimator.fit(X_train, Y_train)

# make predictions
pred = estimator.predict(X_test)

# inverse numeric variables to initial categorical labels
init_lables = encoder.inverse_transform(pred)


# k-fold cross-validate
seed = 42
np.random.seed(seed)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print(results) 
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print('result-1:',np.sum(init_lables==-1)/len(init_lables))