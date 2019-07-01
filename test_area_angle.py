#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#先运行Figure_bh.py 或者Figure_mxg.py

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
def area(point1,point2,point3):
    #Calculate the triangle area
    x1,y1 = point1[0],point1[1]
    x2,y2 = point2[0],point2[1]
    x3,y3 = point3[0],point3[1]
    return 0.5*abs(x1*y2+x2*y3+x3*y1-x1*y3-x2*y1-x3*y2)  


grid_x = grid.x.data
grid_y = grid.y.data
grid_mask = grid.y.mask
area_res = np.zeros((np.shape(grid_mask)))

#area
for i in range(np.shape(grid_mask)[0]-1):
    for j in range(np.shape(grid_mask)[1]-1):
        c1 = not grid_mask[i][j]
        c2 = not grid_mask[i][j+1]
        c3 = not grid_mask[i+1][j+1]
        c4 = not grid_mask[i+1][j]
        
        if c1 and c2 and c3 and c4:
            p1 = [grid_x[i][j],grid_y[i][j]]
            p2 = [grid_x[i][j+1],grid_y[i][j+1]]
            p3 = [grid_x[i+1][j+1],grid_y[i+1][j+1]]
            p4 = [grid_x[i+1][j],grid_y[i+1][j]]
            area_res[i][j] = area(p1,p2,p3)+area(p1,p3,p4)
            
figure = plt.figure()
ax = Axes3D(figure)     
ax.view_init(elev=90, azim=-90)  
ax.grid(False)
plt.axis('off')
surf = ax.plot_surface(grid_x, grid_y, area_res, rstride=1, cstride=1, cmap='rainbow')        

#添加右侧的色卡条
figure.colorbar(surf, shrink=0.6, aspect=20) #shrink表示整体收缩比例，aspect仅对bar的宽度有影响，
# aspect值越大，bar越窄



#angle
angle_res = np.zeros((np.shape(grid_mask)))
for i in range(np.shape(grid_mask)[0]-1):
    for j in range(np.shape(grid_mask)[1]-1):
        c1 = not grid_mask[i][j]
        c2 = not grid_mask[i][j+1]
        c4 = not grid_mask[i+1][j]
        
        if c1 and c2 and c4:
            p1 = [grid_x[i][j],grid_y[i][j]]
            p2 = [grid_x[i][j+1],grid_y[i][j+1]]
            p4 = [grid_x[i+1][j],grid_y[i+1][j]]
            
            v1 = [p2[0]-p1[0],p2[1]-p1[1]]
            v2 = [p4[0]-p1[0],p4[1]-p1[1]]
            
            angle_res[i][j] = (v1[0]*v2[0]+v1[1]*v2[1])/(0.0000000001+np.sqrt((v1[0]**2+v1[1]**2)*(v2[0]**2+v2[1]**2)))
        
figure = plt.figure()
ax = Axes3D(figure)     
ax.view_init(elev=90, azim=-90)  
ax.grid(False)
plt.axis('off')
surf = ax.plot_surface(grid_x, grid_y, angle_res, rstride=1, cstride=1, cmap='rainbow', vmin=-0.25,vmax=0.25)        

#添加右侧的色卡条
figure.colorbar(surf, shrink=0.6, aspect=20) #shrink表示整体收缩比例，aspect仅对bar的宽度有影响，





#angle*1.8 + area
lambda1 = 1.8
lambda2 = 1.0
sum_res = angle_res * lambda1 + area_res * lambda2 
        
figure = plt.figure()
ax = Axes3D(figure)     
ax.view_init(elev=90, azim=-90)  
ax.grid(False)
plt.axis('off')
surf = ax.plot_surface(grid_x, grid_y, sum_res, rstride=1, cstride=1, cmap='rainbow', vmin=-1.2,vmax=1.2)        

#添加右侧的色卡条
figure.colorbar(surf, shrink=0.6, aspect=20) #shrink表示整体收缩比例，aspect仅对bar的宽度有影响，