#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

adjacency_num = 15

os.chdir('./data_rough')
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
    
#    length = (np.array(length)-np.mean(length)) / (np.std(length)+0.00000001)
    length = np.array(length)/sum(length)

    #angle
    for i in range(len_num-1):
        v1 = (lat[i+1] - lat[i] , lon[i+1] - lon[i])
        v2 = (lat[i-1] - lat[i] , lon[i-1] - lon[i])
        inner = v1[0]*v2[0] + v1[1]*v2[1]
#        print(i,inner)
        
        a1 = np.sqrt(v1[0]**2 + v1[1]**2)
        a2 = np.sqrt(v2[0]**2 + v2[1]**2)
        if judge_inner_point(len_num, lat, lon, (lat[i+1]+lat[i-1])/2, (lon[i+1]+lon[i-1])/2 ):
            angle.append(np.arccos(inner/(a1*a2+0.000000001))/np.pi*180 )
        else:
            angle.append(360-np.arccos(inner/(a1*a2+0.000000001))/np.pi*180)
            
    v1 = (lat[0] - lat[-1] , lon[0] - lon[-1])
    v2 = (lat[-2] - lat[-1] , lon[-2] - lon[-1])
    inner = v1[0]*v2[0] + v1[1]*v2[1]
    
    a1 = np.sqrt(v1[0]**2 + v1[1]**2)
    a2 = np.sqrt(v2[0]**2 + v2[1]**2)
    if judge_inner_point(len_num, lat, lon, (lat[0]+lat[-2])/2, (lon[0]+lon[-2])/2 ):
        angle.append(np.arccos(inner/(a1*a2+0.000000001))/np.pi*180 )
    else:
        angle.append(360-np.arccos(inner/(a1*a2+0.000000001))/np.pi*180)  
    
    angle = np.array(angle)/360
    
    return angle,length

Input = []
num = -1
for file in filename:
    num += 1
    f = open(file)
    df = pd.read_csv(f)
    data = df.values[:,1:]
    
    len_num = len(data)
    lat = data[:,0]
    lon = data[:,1]       
    
    
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
            

        
        Input.append(temp_input)

#ANN train________________________________________________________________________________________

temp_total = pd.DataFrame(np.array(Input))
temp_total.to_csv('/home/dedekinds/Single-connected-region-orthogonal-mesh-generation/shp/'+'adj'+str(adjacency_num)+'_len'+str(len_num)+'.csv')
#
#angle = []
#leng = []
#num = -1
#for file in filename:
#    num += 1
#    f = open(file)
#    df = pd.read_csv(f)
#    data = df.values[:,1:]
#    
#    lat = data[:,0]
#    lon = data[:,1]     
#    len_num = len(lat)
#    
#    file_angle ,file_length = adjacency(lat, lon, len_num)
#    angle = angle + list(file_angle)
##    leng = leng + list(file_length)
#
#plt.hist(angle,500,normed =1)   