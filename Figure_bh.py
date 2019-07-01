#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#cd ./Single-connected-region-orthogonal-mesh-generation/train_data
from sklearn.externals import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pygridgen
import matplotlib.pyplot as plt
from solver import solver_main, solve_main_number
import time
from mpl_toolkits.basemap import Basemap

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


adjacency_num = 39
limit_rat_ini = 0.65#ini
resolution_shape = (30,30)
ini_index = 0xf0000000000
lower_enumerate_num = 200
upper_enmuerate_num = 70000

# load model
#gbm = joblib.load('./train_data/lgb_model_10corenr_9adj.pkl')
gbm = joblib.load('./model/lgb_model_40corenr_39adj.pkl')
df = pd.read_csv('./model/bh_0.5d.csv')#<-------------------------------------------------------------- filename
data = df.values[:,1:]
lat = data[:,0]
lon = data[:,1]
len_num = len(lat)

#
#
#start = time.time()
#corner = []
#Input_data = input_data_process(lat, lon ,len_num)
#preds = gbm.predict(Input_data)
#temp_max_preds = []
#for temp in preds:
#    temp_max_preds.append(max(temp))
#temp_max_preds = sorted(temp_max_preds)
#
#for i in range(len(temp_max_preds)):
#    #find the min index of temp_max_preds which satisfy the condition(>limit_rat_ini)
#    if temp_max_preds[i] > limit_rat_ini: 
#        ini_index = i
#        break
#if ini_index == 0xf0000000000:
#    print('All of preds are under limit_rat_ini! plz turn the limit_tar_ini down~')
#else:
#    temp_check = False
#    while not temp_check:
#        limit_rat = temp_max_preds[ini_index]
#        number = solve_main_number(preds, limit_rat, len_num)
#        print('solve_main_number:',number)
#        print('ini_index:',ini_index)
#        print('limit_rat:',limit_rat)
#        if lower_enumerate_num < number < upper_enmuerate_num:
#            start = time.time()
#            Corner, best_loss = solver_main(lat, lon, preds, limit_rat, len_num, resolution_shape)   
#            end = time.time()
#            
#            temp_check = True
#        elif number <= lower_enumerate_num:
#            ini_index += 1
#            if ini_index >= len(temp_max_preds):
#                print('GG, try to turn the parameters(lower/upper bound of the enumerate_number) plz')
#        else:
#            print('________________too large number(>upper bound)______________________')
#            break
#end = time.time()
#print('running time:',abs(end-start))            
#        
#Corner =np.array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1., -1.,  0.,  0.,
#         0.,  0., -1., -1.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  1.,  0.,
#         1., -1.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  1.,  0.,
#        -1., -1.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  1.,  0., -1.,  1.,  0.,  0.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1., -1.,  0.,  0.,
#         0.,  0.,  0., -1., -1.,  0.,  1.,  0.,  0., -1.,  1.,  0.,  1.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#       [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1., -1.,  0.,  0.,
#         0.,  0., -1., -1.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  1.,  0.,
#         1., -1.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  1.,  0.,
#        -1., -1.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  1.,  0., -1.,  1.,  0.,  0.,  1.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0., -1.,  0.,  0.,
#         0.,  0.,  0., -1., -1.,  0.,  1.,  0.,  0., -1.,  1.,  0.,  1.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#       [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1., -1.,  0.,  0.,
#         0.,  0., -1., -1.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  1.,  0.,
#         1., -1.,  0.,  0.,  0.,  0.,  0.,  1., -1.,  0.,  0.,  1.,  0.,
#        -1., -1.,  1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  1.,  0., -1.,  1.,  0.,  0.,  1.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0., -1.,  0.,  0.,
#         0.,  0.,  0., -1., -1.,  0.,  1.,  0.,  0., -1.,  1.,  0.,  1.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])


#bh
Corner = np.array([[ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  1., -1.,  0., -1.,  0.,  1.,
        -1.,  1.,  0., -1., -1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
         0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,
         0.,  0.,  0.,  1.,  0., -1.,  1.,  0.,  1.,  0.,  0.,  0., -1.,
         1., -1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0., -1.,  0.,  0.,
         0.,  0.,  0.,  1.],
       [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  1., -1.,  0., -1.,  0.,  1.,
        -1.,  1.,  0., -1., -1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
         0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,
         0.,  0.,  0.,  1.,  0., -1.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,
         0., -1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0., -1.,  0.,  0.,
         0.,  0.,  0.,  1.],
       [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  1., -1.,  0., -1.,  0.,  1.,
        -1.,  1.,  0., -1., -1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
         0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,
         0.,  0.,  0.,  1.,  0., -1.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,
         0., -1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0., -1.,  0.,
         0.,  0.,  0.,  1.]])



#Corner = np.array([[ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  1., -1.,  0., -1.,  0.,  1.,
#        -1.,  1.,  0., -1., -1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
#         0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,
#         0.,  0.,  0.,  1.,  0., -1.,  1.,  0.,  1.,  0.,  0.,  0., -1.,
#         1., -1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0., -1.,  0.,  0.,
#         0.,  0.,  0.,  1.],
#       [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  1., -1.,  0., -1.,  0.,  1.,
#        -1.,  1.,  0., -1., -1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
#         0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,
#         0.,  0.,  0.,  1.,  0., -1.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,
#         0., -1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0., -1.,  0.,  0.,
#         0.,  0.,  0.,  1.],
#       [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  1., -1.,  0., -1.,  0.,  1.,
#        -1.,  1.,  0., -1., -1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
#         0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,
#         0.,  0.,  0.,  1.,  0., -1.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,
#         0., -1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0., -1.,  0.,
#         0.,  0.,  0.,  1.]])


resolution_shape = (60,60)
for i in range(1):      
    grid = pygridgen.grid.Gridgen(lon, lat, Corner[i], shape = resolution_shape)       
    fig, ax = plt.subplots()
    ax.plot(lon, lat, 'k-')
    ax.plot(lon , lat,'r.')
    ax.plot(grid.x, grid.y, 'b.')
    plt.show() 

#temp_x = grid.x.data
#temp_y = grid.y.data
#res_x = temp_x.reshape(-1,1)[::-1]
#res_y = temp_y.reshape(-1,1)[::-1]
#res = np.column_stack((res_x,res_y))
#df = pd.DataFrame(res)
#df.to_csv('tx_'+str(resolution_shape[0])+'_'+str(resolution_shape[1])+'.csv')


grid_x = grid.x
grid_y = grid.y
grid_mask = grid_y.mask


gap = 1
min_x = min(lon)-gap
max_x = max(lon)+gap
min_y = min(lat)-gap
max_y = max(lat)+gap



#left = -98.83
#right = -68.58
#down = 15.39
#up = 35.11

map = Basemap(llcrnrlon=min_x,
              llcrnrlat=min_y,
              urcrnrlon=max_x,
              urcrnrlat=max_y,
              resolution='i', projection='merc', lat_0 = 0, lon_0 = 0.)
map.shadedrelief()
map.drawcoastlines()

for i in range(np.shape(grid_mask)[0]):
    for j in range(len(grid_mask[0])-1):
        if (not grid_mask[i][j]) and (not grid_mask[i][j+1]):
            a = [grid_x[i][j],grid_x[i][j+1]]
            b = [grid_y[i][j],grid_y[i][j+1]]
            temp_x, temp_y = map(a,b)
            map.plot(temp_x,temp_y,color = 'k')

for i in range(len(grid_mask[0])-1):
    for j in range(np.shape(grid_mask)[0]):
        if (not grid_mask[i][j]) and (not grid_mask[i+1][j]):
            a = [grid_x[i][j],grid_x[i+1][j]]
            b = [grid_y[i][j],grid_y[i+1][j]]
            temp_x, temp_y = map(a,b)
            map.plot(temp_x,temp_y,color = 'k')
            
lon2,lat2 = map(lon,lat)
map.plot(lon2 , lat2,'r.')
plt.show()
#x = [-90,-70]
#y = [18,30]
#
#
#lons, lats = map(x, y)
#map.plot(lons, lats, marker=None,color='k')
#plt.show()

#
#
#
#
#
#Corner = [0]*np.shape(lon)[0]
#C1 = [42,48,93,96,110,113]
#C2 = [101,103]
#for t in C1:
#    Corner[t] = 1
#    
#for t in C2:
#    Corner[t] = -1
#
#Corner = np.array(Corner)
#resolution_shape = (30,200)
#grid = pygridgen.grid.Gridgen(lon, lat, Corner, shape = resolution_shape)       
#fig, ax = plt.subplots()
#ax.plot(lon, lat, 'k-')
#ax.plot(lon , lat,'r.')
#ax.plot(grid.x, grid.y, 'b.')
#plt.show() 
#
#
#temp_x = grid.x.data
#temp_y = grid.y.data
##res_x = temp_x.reshape(-1,1)[::-1]
##res_y = temp_y.reshape(-1,1)[::-1]
##res = np.column_stack((res_x,res_y))
##df = pd.DataFrame(res)
##df.to_csv('tx_'+str(resolution_shape[0])+'_'+str(resolution_shape[1])+'.csv')
#
#
#
#df = pd.DataFrame(temp_x)
#df.to_csv('x'+str(resolution_shape[0])+'_'+str(resolution_shape[1])+'.csv')
#
#df = pd.DataFrame(temp_y)
#df.to_csv('y'+str(resolution_shape[0])+'_'+str(resolution_shape[1])+'.csv')
