#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.externals import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pygridgen
import matplotlib.pyplot as plt
from solver import solver_main, solve_main_number
import time
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--modelname",      nargs='?',   type=str,    help=" argument", default='lgb_model_10corenr_9adj.pkl')
parser.add_argument("--adj_num",        nargs='?',   type=int,    help=" argument", default=9)
parser.add_argument("--From",        nargs='?',   type=int,    help=" argument", default=30)
parser.add_argument("--To",        nargs='?',   type=int,    help=" argument", default=40)

args = parser.parse_args()


def judge_inner_point(nvert, vertx, verty, testx, testy):
    #PNPoly algorithm (judge whether a point is in a given polygon
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
            angle.append(np.arccos(inner/(a1*a2+0.00000001))/np.pi*180 )
        else:
            angle.append(360-np.arccos(inner/(a1*a2+0.00000001))/np.pi*180)
            
    v1 = (lat[0] - lat[-1] , lon[0] - lon[-1])
    v2 = (lat[-2] - lat[-1] , lon[-2] - lon[-1])
    inner = v1[0]*v2[0] + v1[1]*v2[1]
    
    a1 = np.sqrt(v1[0]**2 + v1[1]**2)
    a2 = np.sqrt(v2[0]**2 + v2[1]**2)
    if judge_inner_point(len_num, lat, lon, (lat[0]+lat[-2])/2, (lon[0]+lon[-2])/2 ):
        angle.append(np.arccos(inner/(a1*a2+0.00000001))/np.pi*180 )
    else:
        angle.append(360-np.arccos(inner/(a1*a2+0.00000001))/np.pi*180)  
    
    angle = np.array(angle)/360
    
    return angle,length

def input_data_process(lat, lon, len_num):
    file_angle ,file_length = adjacency(lat, lon, len_num)
    Input = []
    
    for i in range(len_num):
    #length + angle -->  input
    #corner --> output
        temp_input = []
        
        #input length
        for j in range(int(adjacency_num/2)):
            temp_input.append(file_length[(i+j)%len_num])
            temp_input.append(file_length[(i-j-1)%len_num])
            
            
        #input angle
        temp_input.append(file_angle[i])
        for j in range(int((adjacency_num-3)/2)):
            temp_input.append(file_angle[(i+j+1)%len_num])
            temp_input.append(file_angle[(i-j-1)%len_num])
            
        Input.append(temp_input)
    return np.array(Input)
'''
_______________________________________________________________________________________________________________________________________
______________________________________________________main_____________________________________________________________________________
_______________________________________________________________________________________________________________________________________
'''
gbm = joblib.load('./model/'+args.modelname)
os.chdir('./shp')
Fromto = str(args.From)+'_to_'+str(args.To)
if not os.path.exists('train_data_'+Fromto):
    os.mkdir('train_data_'+Fromto)
    
adjacency_num = args.adj_num       #depend on the last classification(e.g. 10_to_15,depend on 10's adjacency_num)
limit_rat_ini = 0.65    #ini
resolution_shape = (20,20)
ini_index = 0xf0000000000
lower_enumerate_num = 300
upper_enmuerate_num = 7000

# load model

os.chdir('./data_rough')
lst = os.listdir(os.getcwd())
filename = []

for c in lst:
    if os.path.isfile(c) and c.endswith('.csv'):# and c.find("test") == -1
        filename.append(c)


num = 0
for file in filename:
    num += 1
    print('loop:',num)

    df = pd.read_csv(file)#<-------------------------------------------------------------- filename
    data = df.values[:,1:]
    lat = data[:,0]
    lon = data[:,1]
    len_num = len(lat)
    
    #plt.plot(lat,lon,'r.')
    
    start = time.time()
    corner = []
    Input_data = input_data_process(lat, lon ,len_num)
    preds = gbm.predict(Input_data)
    temp_max_preds = []
    for temp in preds:
        temp_max_preds.append(max(temp))
    temp_max_preds = sorted(temp_max_preds)
    
    for i in range(len(temp_max_preds)):
        #find the min index of temp_max_preds which satisfy the condition(>limit_rat_ini)
        if temp_max_preds[i] > limit_rat_ini:
            ini_index = i
            break
    if ini_index == 0xf0000000000:
        print('All of preds are under limit_rat_ini! plz turn the limit_tar_ini down~')
    else:
        temp_check = False
        while not temp_check:
            limit_rat = temp_max_preds[ini_index]
            number = solve_main_number(preds, limit_rat, len_num)
            print('solve_main_number:',number)
            print('ini_index:',ini_index)
            print('limit_rat:',limit_rat)
            if lower_enumerate_num < number < upper_enmuerate_num:
                start = time.time()
                Corner, best_loss = solver_main(lat, lon, preds, limit_rat, len_num, resolution_shape)   
                end = time.time()
                
                temp_check = True
            elif number <= lower_enumerate_num:
                ini_index += 1
                if ini_index >= len(temp_max_preds):
                    print('GG, try to turn the parameters(lower/upper bound of the enumerate_number) plz _(:_」∠)_')
            else:
                print('________________too large number(>upper bound)_________________(~﹏~)')
                break
    end = time.time()
    print('______________________________running time:',abs(end-start))  

    if temp_check:
        temp_best_beta = list(Corner[0])         
        temp_best_beta.append(best_loss[0])
        temp_best_beta.append(len(Corner[0]))
        res = np.column_stack((data.reshape(1,-1) ,np.array(temp_best_beta ).reshape(1,-1)))


        res_dir = os.path.abspath(os.path.dirname(os.getcwd()))+"/train_data_"+Fromto
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)

        pd_data = pd.DataFrame(res)
        pd_data.to_csv(res_dir+'/'+Fromto+'_'+file)

        grid = pygridgen.grid.Gridgen(lat, lon, Corner[0], shape = resolution_shape)    
#plot
#        fig, ax = plt.subplots()
#        ax.plot(lat, lon, 'k-')
#        ax.plot(lat,lon,'r.')
#        ax.plot(grid.x, grid.y, 'b.')
#        plt.show() 
        print(best_loss[0])
        print(file)
        print('________________________________________________________________________')
    else:
        continue

    
                