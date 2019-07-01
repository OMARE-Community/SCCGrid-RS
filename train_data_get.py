#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pygridgen
import numpy as np
import pandas as pd
import utils as ul
import time
from solver import solver_origin
import get_polygon_area as gpa


os.chdir('./shp/data_rough')
lst = os.listdir(os.getcwd())
filename = []
num_corner = 10
resolution_shape = (20,20)


for c in lst:
    if os.path.isfile(c) and c.endswith('.csv'):# and c.find("test") == -1
        filename.append(c)
        
num = len(filename)        
start = time.time()
for file in filename:
    print(num)
    num -= 1
    f = open(file)
    df = pd.read_csv(f)
    data = df.values[:,1:]
    best_beta, best_loss = solver_origin(data,num_corner,resolution_shape)
    x = data[:,0]
    y = data[:,1]



    temp_best_beta = list(best_beta)
    temp_best_beta.append(best_loss)
    temp_best_beta.append(num_corner)
    res = np.column_stack((data.reshape(1,-1) ,np.array(temp_best_beta ).reshape(1,-1)))
    
    res_dir = os.path.abspath(os.path.dirname(os.getcwd()))+"/train_data"
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    
    
    grid = pygridgen.grid.Gridgen(x,y, best_beta, shape=resolution_shape)
    
    fig, ax = plt.subplots()
    ax.plot(x, y, 'k-')
    ax.plot(x,y,'r.')
    ax.plot(grid.x, grid.y, 'b.')
    plt.show()
    
    
    #plt.savefig(res_dir+'/train_'+file+".jpg")
    
    pd_data = pd.DataFrame(res)
    pd_data.to_csv(res_dir+'/train_'+file)



end = time.time()
print("running time:")
print(end-start)


#
##for file in filename:
#file = filename[0]
#f = open(file)
#df =pd.read_csv(f)
#data = df.values[:,1:]
#
#num_corner = 10
#resolution_shape = (20,30)
#
#start = time.time()
#best_beta, best_loss = solver_origin(data,num_corner,resolution_shape)
#end = time.time()
#print(end-start)
#
#
#best_beta = (0, 0, 0, 0, 1, 0, 1, 1, 1, 0)
#
#
#x = data[:,0]
#y = data[:,1]
##best_beta = (-1, 1, 1, 1, 0, -1, 1, 1, 1, 0)
##best_beta = (1,1,0,0,1,0,0,1,1,-1)
#
##best_beta = (1, 0, 0, 0, 1, -1, 1, 1, 0, 1)
##best_beta = (1, 0, 0, 0, 1, 0, 0, 1, 0, 1)
#
#
##best_beta =(1, 0, 0, 0, 1, 0, 0, 1, 0, 1)
#grid = pygridgen.grid.Gridgen(x,y, best_beta, shape=resolution_shape)
#
#fig, ax = plt.subplots()
#ax.plot(x, y, 'k-')
#ax.plot(x,y,'r.')
#ax.plot(grid.x, grid.y, 'b.')
#plt.show()
#
#
#plt.savefig("examples.jpg")
#
##if isinstance(grid.x,np.ma.core.MaskedArray):
##    loss = ul.loss_for_mask(grid.x,grid.y,area_origin)
##else:
##    loss = ul.loss_for_no_mask(grid.x,grid.y,area_origin)
##    
##print(loss)


