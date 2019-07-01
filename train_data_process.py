# -*- coding: utf-8 -*-

'''
#pip install pyshp
'''
import shapefile
import matplotlib.pyplot as plt
import utils as ul
import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--node_num",      nargs='?',   type=int,    help=" argument", default=10)
args = parser.parse_args()




node_num = args.node_num

def judge_simple(Lon,Lat):
    if len(Lon) != node_num:
        return False
    
    
    #First, we judge special case  
    v1 = [Lon[0],Lat[0],Lon[-1],Lat[-1]]
    for k in range(len(Lon)-1):
        v2 = [Lon[k],Lat[k],Lon[k+1],Lat[k+1]]
        if ul.check_intersect(v1,v2):
            return False
    
    #general case
    for i in range(len(Lon)-1):
        for j in range(len(Lat)-1):
            if i == j:
                continue
            else:
                v1 = [Lon[i],Lat[i],Lon[i+1],Lat[i+1]]
                v2 = [Lon[j],Lat[j],Lon[j+1],Lat[j+1]]
                if ul.check_intersect(v1,v2):
                    return False
    return True





os.chdir('./shp')
if not os.path.exists('data_rough'):
    os.mkdir('data_rough')
lst = os.listdir(os.getcwd())
Filename = []

for c in lst:
    if os.path.isfile(c) and c.endswith('.dbf'):# and c.find("test") == -1
        Filename.append(c)
for filename in Filename:            
    #filename = 'gadm36_LAO_2'#"gadm36_CHN_2.shp"
    
    sf = shapefile.Reader(filename)
        #gadm36_CHN_0:China
        #gadm36_CHN_1:province of China
        #gadm36_CHN_2:major city in China
        
    shapes = sf.shapes()
    
    
    Sum = 0
    where = []
    for i in range(len(shapes)):
        boundary_data = shapes[i].points
        
        #first gap
        Longitude = []
        Latitude = []
        count = 1
        gap = int(len(boundary_data)/node_num)
        if gap == 0:
            break
        
        for temp in boundary_data:
            if count % gap ==0:
                Longitude.append(temp[0])
                Latitude.append(temp[1])   
            count += 1
            
        #judge simple polygon
        #print(judge_simple(Longitude,Latitude))
        #plt.plot(Longitude,Latitude)
        
        if judge_simple(Longitude,Latitude):
            col1 = np.array(Longitude).reshape(-1,1)
            col2 = np.array(Latitude).reshape(-1,1)
            res = np.column_stack((col1,col2))
            pd_data = pd.DataFrame(res)
            pd_data.to_csv('./data_rough/'+filename+'_'+str(i)+'_first'+'.csv')
        
    
        #second gap
        Longitude = []
        Latitude = []
        count = 1
        gap = int(len(boundary_data)/node_num)  
        
        for temp in boundary_data:
            if count % gap ==int(gap/2):
                Longitude.append(temp[0])
                Latitude.append(temp[1])   
            count += 1
            
        
        if judge_simple(Longitude,Latitude):
            col1 = np.array(Longitude).reshape(-1,1)
            col2 = np.array(Latitude).reshape(-1,1)
            res = np.column_stack((col1,col2))
            pd_data = pd.DataFrame(res)
            pd_data.to_csv('./data_rough/'+filename+'_'+str(i)+'_second'+'.csv')
    
