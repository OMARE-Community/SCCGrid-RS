#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#for China east Sea


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from netCDF4 import Dataset
#from train_data_process import judge_simple
import pandas as pd

'''    
          (i-1,j)
  (i,j-1)  (i,j) (i,j+1)
          (i+1,j)
'''
def get_next_location(now_location,direction):
    i = now_location[0]
    j = now_location[1]
    if direction == 'up':    return [(i,j-1),(i-1,j),(i,j+1),(i+1,j)]
    if direction == 'down':  return [(i,j+1),(i+1,j),(i,j-1),(i-1,j)]
    if direction == 'right': return [(i-1,j),(i,j+1),(i+1,j),(i,j-1)]
    if direction == 'left':  return [(i+1,j),(i,j-1),(i-1,j),(i,j+1)]

def get_next_direction(now_location,temp_location):
    if now_location[0] == temp_location[0]:
        if now_location[1] - temp_location[1] == 1:  return 'left'
        if now_location[1] - temp_location[1] == -1: return 'right'
        
    if now_location[1] == temp_location[1]:
        if now_location[0] - temp_location[0] == 1:  return 'up'
        if now_location[0] - temp_location[0] == -1: return 'down'

def seach_index_lon_lat(lonlat,num):
    for i in range(len(lonlat)):
        if lonlat[i] > num:
            return i
        
        
fh = Dataset('./shp/ETOPO1_bh.nc', mode='r')
lons = fh.variables['lon'][:]
lats = fh.variables['lat'][:]
band1 = fh.variables['Band1'][:]



######################################################################3
#remove some place we dont want to consider about


########################################################################


boundary = [] 
#check the four vectex of the region
#find two ocean points
#vectex_check = [[0,0],[0,np.shape(band1)[1]-1],[np.shape(band1)[0]-1,np.shape(band1)[1]-1],[np.shape(band1)[0]-1,0],[0,0]]
#origin_direction = ['right','down','left','up']
#
#for temp in range(len(vectex_check)-1):
#    point1_x = vectex_check[temp][0]
#    point1_y = vectex_check[temp][1]
#    
#    point2_x = vectex_check[temp+1][0]
#    point2_y = vectex_check[temp+1][1]
#    
#    if band1[point1_x][point1_y] <= 0 and band1[point2_x][point2_y] <= 0:
#        break

direction = 'down'

point1_x = 40
point1_y = np.shape(band1)[1]-1

now_location = (point1_x,point1_y)
lon,lat = np.meshgrid(lons,lats)   

num = 0
while now_location != (point1_x,point1_y) or num == 0:
    num += 1
    #print(now_location)
    for temp_location in get_next_location(now_location,direction):
        #print(temp_location)
        if  0 <= temp_location[0] < np.shape(band1)[0] and 0 <= temp_location[1] < np.shape(band1)[1]:
            
            temp_x = temp_location[0]
            temp_y = temp_location[1]
            
            if band1[temp_x][temp_y] <= 0:# on the ocean       
                direction = get_next_direction(now_location,temp_location)
                now_location = temp_location
            
                bound = [lat[temp_x][temp_y],lon[temp_x][temp_y]]
                if bound not in boundary:
                    boundary.append(bound)
                else:
                    temp_bound1 = boundary.pop()
                    temp_bound2 = boundary.pop()
                    boundary.append(temp_bound1)
                    boundary.append(temp_bound2)
                break
    
    
#X, Y = np.meshgrid(X, Y)    

boundary = np.array(boundary)
la = boundary[:,0]    
lo = boundary[:,1]

    #check if the polygon is a inner intersect polygon 
#if not judge_simple(la,lo):
#    plt.plot(lo,la,'o')
#else:
#    print('inner intersect!')

plt.plot(lo,la,'o')



#degree = 1/60
degree = 0.3
gap = degree * 60
temp_la = []
temp_lo = []


for i in range(len(la)):
    if i % gap == 0:
        temp_la.append(la[i])
        temp_lo.append(lo[i])
res = np.column_stack((np.array(temp_la).reshape(-1,1),np.array(temp_lo).reshape(-1,1)))
df = pd.DataFrame(res)
df.to_csv('./model/ETOPO1_mxg.nc233333.csv')

plt.plot(temp_lo,temp_la,'o--')
#if not judge_simple(temp_la,temp_lo):
#    plt.plot(temp_lo,temp_la,'o--')
#else:
#    print('inner intersect!')



#
#
#
#

#array([[ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  1., -1.,  0., -1.,  0.,  1.,
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




