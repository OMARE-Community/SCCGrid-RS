#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pygridgen
import numpy as np
import pandas as pd
import utils as ul
import itertools
import get_polygon_area as gpa

loss_limit = 0.05
def solver_origin(data, num_corner, resolution_shape):
    #return [-1 1 0 1 -1 0 1 1 1 0 1] as example
    #return the corner value(-1 1 0) of each vertex
    
    
    corner = [-1,0,1]
    best_beta = ()
    best_loss = 0xf00000000
    arg = []
    for i in range(num_corner):
        arg.append(corner)
    
    num = 0
    c = itertools.product(*arg)
    for beta in c:  
        if sum(beta) == 4:
            num += 1
#            if num % 200 ==0:
#                print(num)
            x = data[:,0]
            y = data[:,1]
            grid = pygridgen.grid.Gridgen(x, y, beta, shape=resolution_shape)
            area_origin = gpa.get_area(x,y)
            
            if isinstance(grid.x,np.ma.core.MaskedArray):
                loss = ul.loss_for_mask(grid.x,grid.y,area_origin)
            else:
                loss = ul.loss_for_no_mask(grid.x,grid.y,area_origin)
            
            if loss < best_loss:
                best_loss = loss
                best_beta = beta
        
    return best_beta, best_loss

            

def solver_main(lat, lon, preds, limit_rat, len_num, resolution_shape):
    #after classification-solver
    best_beta = np.zeros((1,len_num))
    pending_index = []
    pending_corner = []

    for i in range(len_num):
        if max(preds[i]) > limit_rat:
            best_beta[0][i] = np.argmax(preds[i]) - 1
        else:
            temp_corner = [-1,0,1]
            temp_corner.pop(np.argmin(preds[i]))
            
            pending_corner.append(temp_corner)
            pending_index.append(i)
    
    num = 0
    c = itertools.product(*pending_corner)
    
    best_loss = 0xf00000000
    sec_best_loss = 0xf000000000
    thir_best_loss = 0xf0000000000
    
    Corner = []
    sec_Corner = []
    thir_Corner = []
    Sum = solve_main_number(preds, limit_rat, len_num)

    
    for beta in c:
        for i in range(len(beta)):
            best_beta[0][pending_index[i]] = beta[i]
            
        
        if sum(best_beta[0]) == 4:
            num += 1
            if num % (int(Sum/10)+1) == 0 :
                print('-----------------------------------',num,'/',Sum)
            x = lat
            y = lon
            grid = pygridgen.grid.Gridgen(x, y, best_beta[0], shape = resolution_shape)
            area_origin = gpa.get_area(x,y)

            if isinstance(grid.x,np.ma.core.MaskedArray):
                loss = ul.loss_for_mask(grid.x,grid.y,area_origin)
            else:
                loss = ul.loss_for_no_mask(grid.x,grid.y,area_origin)
#            print(num,loss ,num % int(Sum/10)+1 == 0)
                
            if loss < best_loss:
                best_loss = loss
                Corner = list(best_beta[0])[:]
            
            if best_loss < loss < sec_best_loss:
                sec_best_loss = loss
                sec_Corner = list(best_beta[0])[:]

            if sec_best_loss < loss < thir_best_loss:
                thir_best_loss = loss
                thir_Corner = list(best_beta[0])[:]

                
    return np.array([Corner,sec_Corner,thir_Corner]), [best_loss,sec_best_loss,thir_best_loss]
    
def solve_main_number(preds, limit_rat, len_num):
    #after classification-solver
    best_beta = np.zeros((1,len_num))
    pending_index = []
    pending_corner = []

    for i in range(len_num):
        if max(preds[i]) > limit_rat:
            best_beta[0][i] = np.argmax(preds[i]) - 1
        else:
            temp_corner = [-1,0,1]
            temp_corner.pop(np.argmin(preds[i]))
            
            pending_corner.append(temp_corner)
            pending_index.append(i)
    
    num = 0
    c = itertools.product(*pending_corner)

    
    for beta in c:
        for i in range(len(beta)):
            best_beta[0][pending_index[i]] = beta[i]
            
        
        if sum(best_beta[0]) == 4:
            num += 1
    return num