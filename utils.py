# -*- coding: utf-8 -*-

'''
#pip install pyshp
'''
import numpy as np
import math
import get_polygon_area as gpa
import time

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
        print(P1,P2,c)
    return c
    
def vector_cross(vec1,vec2):
    #caculate the cross between two vector
    #data format:
    #vec1:list: [vec1.x , vec1.y]
    return vec1[0]*vec2[1]-vec1[1]*vec2[0]
    


def check_intersect(v1,v2):
    #check if two line segments intersect 
    #reference:
    #https://martin-thoma.com/how-to-check-if-two-line-segments-intersect/
    #https://www.cnblogs.com/sytu/articles/3876585.html
    #Of course, there is a O(nlogn) algorithm : http://www.twinklingstar.cn/2016/2994/6-3-polygon-check/
    
    #data format:
    #v1:list: [v1.left_endpoint_x ,v1.left_endpoint_y, v1.right_endpoint_x, v1.right_endpoint_y]
    
    
    #if v1 and v2 have the same endpoint, we think they don't intersect
    if (v1[0] in v2 and v1[1] in v2) or (v1[2] in v2 and v1[3] in v2):
        return False
    
    
    #box test
    P1 = max(v1[0],v1[2]) < min(v2[0],v2[2])
    P2 = max(v1[1],v1[3]) < min(v2[1],v2[3])
    P3 = max(v2[0],v2[2]) < min(v1[0],v1[2])
    P4 = max(v2[1],v2[3]) < min(v1[1],v1[3])
    if P1 or P2 or P3 or P4:
        return False
    
    else:
    #vector crossing
        a1 = [v1[0]-v2[2],v2[1]-v2[3]]
        b1 = [v2[0]-v2[2],v2[1]-v2[3]]
        c1 = [v1[2]-v2[2],v1[3]-v2[3]]
        
        a2 = [v1[0]-v2[0],v1[1]-v2[1]]
        b2 = [-v2[0]+v2[2],-v2[1]+v2[3]]#-b1
        c2 = [v1[2]-v2[0],v1[3]-v2[1]]
        
        if vector_cross(a1,b1)*vector_cross(c1,b1) <= 0 and  vector_cross(a2,b2)*vector_cross(c2,b2) <= 0:
            return True
        else:
            return False
        
def area(point1,point2,point3):
    #Calculate the triangle area
    x1,y1 = point1[0],point1[1]
    x2,y2 = point2[0],point2[1]
    x3,y3 = point3[0],point3[1]
    return 0.5*abs(x1*y2+x2*y3+x3*y1-x1*y3-x2*y1-x3*y2)     
        
        
def loss_for_no_mask(grid_x, grid_y,area_origin):
    # the loss of our optimal problem
    # loss = lambda1*(max-area-mean_area)/max-area + lambda2*0.5* (abs(max_angle)+abs(min_angle)) + lambada3 * area rate
    # In order to reduce the amount of calculation, we only calculate the points in boundary  :)
    #(the max and min usually appear in the corner(vertex) of the polygon)
    max_angle = -0xf000000
    min_angle = 0xf000000
    max_area = -0xf000000
    min_area = 0xf000000
    lambda1 = 1.0
    lambda2 = 1.8
    lambda3 = 3.0
    size = np.shape(grid_x)
    area_x = []
    area_y = []
    
    
    area_num = 0
    mean_area = 0
    
    for i in range(size[0]-1):
        area_x.append(grid_x[i][0])
        area_y.append(grid_y[i][0])
    #left
    #  *----
    #  *   \
    #  *   \
    #  *   \
    #  -----

        
        # angle
        v1 = [grid_x[i+1][0]-grid_x[i][0] , grid_y[i+1][0]-grid_y[i][0]]
        v2 = [grid_x[i][1]-grid_x[i][0] , grid_y[i][1]-grid_y[i][0]]
        inner_product = v1[0]*v2[0] + v1[1]*v2[1]
        norm = np.sqrt( (v1[0]**2+v1[1]**2)*(v2[0]**2+v2[1]**2) )
        value_cos = inner_product/(norm+0.0000001)
        
        if value_cos > max_angle:
            max_angle = value_cos
        if value_cos < min_angle:
            min_angle = value_cos
            
        
        #area
        #p0---p1
        #|  /  |
        #| /   |
        #p3---p2
        p0 = [grid_x[i][0],grid_y[i][0]]
        p1 = [grid_x[i][1],grid_y[i][1]]
        p2 = [grid_x[i+1][1],grid_y[i][1]]
        p3 = [grid_x[i+1][0],grid_y[i][0]]
        
        areas = area(p0,p1,p3) + area(p1,p2,p3)
        #print('area=',areas)
        mean_area += areas
        area_num += 1
        
        if areas > max_area:
            max_area = areas
        if areas < min_area:
            min_area = areas
  

        
    for j in range(size[1]-1):
        area_x.append(grid_x[size[0]-1][j])
        area_y.append(grid_y[size[0]-1][j])
    #down
    #  -----
    #  \   \
    #  \   \
    #  \   \
    #  ****-
        
        #angle
        v1 = [grid_x[size[0]-1][j]-grid_x[size[0]-2][j], grid_y[size[0]-1][j]-grid_y[size[0]-2][j]]
        v2 = [grid_x[size[0]-1][j+1]-grid_x[size[0]-1][j],grid_y[size[0]-1][j+1]-grid_y[size[0]-1][j]]
        inner_product = v1[0]*v2[0] + v1[1]*v2[1]
        norm = np.sqrt( (v1[0]**2+v1[1]**2)*(v2[0]**2+v2[1]**2) )
        value_cos = inner_product/(norm+0.0000001)
        
        if value_cos > max_angle:
            max_angle = value_cos
        if value_cos < min_angle:
            min_angle = value_cos
            
        #area
        p0 = [grid_x[-2][j],grid_y[-2][j]]
        p1 = [grid_x[-2][j+1],grid_y[-2][j+1]]
        p2 = [grid_x[-1][j+1],grid_y[-1][j+1]]
        p3 = [grid_x[-1][j],grid_y[-1][j]]
        
        areas = area(p0,p1,p3) + area(p1,p2,p3)
        mean_area += areas
        area_num += 1
        
        if areas > max_area:
            max_area = areas
        if areas < min_area:
            min_area = areas
 
    
    
    #special case
    #  -----
    #  \   \
    #  \   \
    #  \   \
    #  ----*
    area_x.append(grid_x[-1][-1])
    area_y.append(grid_y[-1][-1])
    #angle
    v1 = [grid_x[-1][-1]-grid_x[-2][-1],grid_y[-1][-1]-grid_y[-2][-1]]
    v2 = [grid_x[-1][-1]-grid_x[-1][-2],grid_y[-1][-1]-grid_y[-1][-2]]
    inner_product = v1[0]*v2[0] + v1[1]*v2[1]
    norm = np.sqrt( (v1[0]**2+v1[1]**2)*(v2[0]**2+v2[1]**2) )
    value_cos = inner_product/(norm+0.0000001)
    
    if value_cos > max_angle:
        max_angle = value_cos
    if value_cos < min_angle:
        min_angle = value_cos
        
        
    temp_x = []
    temp_y = []     
    for i in range(size[0]-1):
        temp_x.append(grid_x[i][size[1]-1])
        temp_y.append(grid_y[i][size[1]-1])
        
    #right
    #  ----*
    #  \   *
    #  \   *
    #  \   *
    #  -----
        
        #angle
        v1 = [grid_x[i+1][size[1]-1]-grid_x[i][size[1]-1],grid_y[i+1][size[1]-1]-grid_y[i][size[1]-1]]
        v2 = [grid_x[i][size[1]-1]-grid_x[i][size[1]-2],grid_y[i][size[1]-1]-grid_y[i][size[1]-2] ]
        inner_product = v1[0]*v2[0] + v1[1]*v2[1]
        norm = np.sqrt( (v1[0]**2+v1[1]**2)*(v2[0]**2+v2[1]**2) )
        value_cos = inner_product/(norm+0.0000001)
        
        if value_cos > max_angle:
            max_angle = value_cos
        if value_cos < min_angle:
            min_angle = value_cos
            
           
        #area
        p0 = [grid_x[i][-2],grid_y[i][-2]]
        p1 = [grid_x[i][-1],grid_y[i][-1]]
        p2 = [grid_x[i+1][-1],grid_y[i][-1]]
        p3 = [grid_x[i+1][-2],grid_y[i][-2]]
        
        areas = area(p0,p1,p3) + area(p1,p2,p3)
        mean_area += areas
        area_num += 1
        
        if areas > max_area:
            max_area = areas
        if areas < min_area:
            min_area = areas
    area_x = area_x + temp_x[::-1]
    area_y = area_y + temp_y[::-1]

    temp_x = []
    temp_y = []
    for j in range(size[1]-1):
        temp_x.append(grid_x[0][j])
        temp_y.append(grid_y[0][j])
    #up
    #  ****-
    #  \   \
    #  \   \
    #  \   \
    #  -----   
        
        #angle
        v1 = [grid_x[1][j]-grid_x[0][j], grid_y[1][j]-grid_y[0][j]]
        v2 = [grid_x[0][j+1]-grid_x[0][j],grid_y[0][j+1]-grid_y[0][j] ]
        inner_product = v1[0]*v2[0] + v1[1]*v2[1]
        norm = np.sqrt( (v1[0]**2+v1[1]**2)*(v2[0]**2+v2[1]**2) )
        value_cos = inner_product/(norm+0.0000001)
        
        if value_cos > max_angle:
            max_angle = value_cos
        if value_cos < min_angle:
            min_angle = value_cos
            
            
        #area
        p0 = [grid_x[0][j],grid_y[0][j]]
        p1 = [grid_x[0][j+1],grid_y[0][j+1]]
        p2 = [grid_x[1][j+1],grid_y[1][j+1]]
        p3 = [grid_x[1][j],grid_y[1][j]]
        
        areas = area(p0,p1,p3) + area(p1,p2,p3)
        mean_area += areas
        area_num += 1
        
        if areas > max_area:
            max_area = areas
        if areas < min_area:
            min_area = areas

    area_x = area_x + temp_x[::-1]
    area_y = area_y + temp_y[::-1]
           
    
    loss_angle = 0.5*(abs(max_angle)+abs(min_angle))
    loss_area = (max_area - mean_area/area_num )/(max_area+0.000000001)
    loss_area_rate = abs(area_origin - gpa.get_area(area_x,area_y)  )/(area_origin+0.0000001)
#    print(loss_angle)
#    print(loss_area)
#    print(loss_area_rate)
    
    return lambda1*loss_angle + lambda2*loss_area + lambda3*loss_area_rate
        
        
        
def loss_for_mask(grid_x,grid_y,area_origin):
    # the loss of our optimal problem
    # loss = lambda1*(max-area-mean_area)/max-area + lambda2*0.5* (abs(max_angle)+abs(min_angle)) + lambada3 * area rate
    # In order to reduce the amount of calculation, we only calculate the points in boundary  :)
    #(the max and min usually appear in the corner(vertex) of the polygon)
    
    #there are some details for this algorithm. Actually, it is not a normal algorithm for every situation(but in most situation)
    #first, we should look for a bounary point(search it through the middle line of the X axis)
    #then we search next bounary point(Four adjacent point  +  Anti-clockwise)
    
    
    size = np.shape(grid_x)
    direction = 'down'
    grid_mask = grid_x.mask
    next_point = [0,0]
    lambda1 = 1.0
    lambda2 = 1.8
    lambda3 = 3.0
    limit_time = 3
    max_angle = -0xf000000
    min_angle = 0xf000000
    max_area = -0xf000000
    min_area = 0xf000000
    
    mean_area = []
    
    visited = np.array([[0]*size[1] for i in range(size[0])])
    #look for a bounary point
    judge_if_adjacent_point = False
    ini_location = int(size[1]/2)
    while not judge_if_adjacent_point:
    
        for i in range(size[0]):
            if not grid_mask[i][ini_location]:
                next_point[0] = i
                next_point[1] = ini_location
                break
        
        origin_point = next_point[:]
        if i == size[0]-1:
            break
        
        loc = [[origin_point[0]-1,origin_point[1]],
              [origin_point[0]+1,origin_point[1]],
              [origin_point[0],origin_point[1]-1],
              [origin_point[0],origin_point[1]+1]]
#        print(loc)
        
        if grid_mask[loc[0][0],loc[0][1]] and grid_mask[loc[1][0],loc[1][1]] and grid_mask[loc[2][0],loc[2][1]] and grid_mask[loc[3][0],loc[3][1]]:
            ini_location += 1
            if ini_location > size[1]-2:
                print('GG,loss_for_mask(grid_x,grid_y,area_origin),ini_location')
        else:
            judge_if_adjacent_point = True
    
    #search next bounary point
    chance = 1
    check_area = [-10,-5,-6]
    area_x = [grid_x[next_point[0]][next_point[1]] ]
    area_y = [grid_y[next_point[0]][next_point[1]] ]
    
    
    start = time.time()
    while next_point != origin_point or chance != 0 or len(mean_area) < 0xf0000000:
        #check_area && 3rd condition--->deal with the case like(Return to the origin prematurely):
        visited[next_point[0]][next_point[1]] = 1
        #print(direction)
        check_area.append(len(mean_area))
#        print(check_area)
        if check_area[-1] == check_area[-2] == check_area[-3] and check_area[-1] > max(size[0],size[1]):
            break
        
        end = time.time()
        if abs(end - start) > limit_time:
            return 0xf0000000#loss
#        print(len(mean_area))
#        print(next_point)
        if next_point == origin_point:
            chance -= 1
            
        cnt = 0
        for temp in loss_location(next_point, direction):
            cnt += 1
            if  -1<temp[0]<size[0] and -1<temp[1]<size[1] and not grid_mask[temp[0]][temp[1]]:
                next_point[0] = temp[0]
                next_point[1] = temp[1]
                
                direction = loss_direction(direction, cnt)
                break
            
        #angle_loss
        res_angle_loss = loss_angle(next_point,size,grid_x,grid_y)
        if res_angle_loss > max_angle:
            max_angle = res_angle_loss
        if res_angle_loss < min_angle:
            min_angle = res_angle_loss
        
        
        #area_loss
        if not visited[next_point[0]][next_point[1]]:
            area_x.append(grid_x[next_point[0]][next_point[1]])
            area_y.append(grid_y[next_point[0]][next_point[1]])
            
            res_area_loss = loss_area(next_point,size,grid_x,grid_y)
            mean_area.append(res_area_loss)
            if res_area_loss > max_area:
                max_area = res_area_loss
            if res_area_loss < min_area:
                min_area = res_area_loss
                
            #masked
    mean_area_last = []
    for temp in mean_area:
        if isinstance(temp,float):
            mean_area_last.append(temp)
    
    #total loss
    Loss_angle = 0.5*(abs(max_angle)+abs(min_angle))
    Loss_area = (max_area-np.mean(mean_area_last))/(max_area+0.000000001)
    loss_area_rate = abs(area_origin- gpa.get_area(area_x,area_y)  )/(area_origin+0.0000001)
#    print(Loss_angle)
#    print(Loss_area)
#    print(loss_area_rate)
    
    
    return lambda1*Loss_angle + lambda2*Loss_area + lambda3*loss_area_rate
    
    
      
def loss_angle(Next_point,Size,Grid_x,Grid_y):
    i = Next_point[0]
    j = Next_point[1]
    grid_mask = Grid_x.mask

    if 0 < i+1 <Size[0] and not grid_mask[i+1][j]:
        v1 = [Grid_x[i+1][j]-Grid_x[i][j],Grid_y[i+1][j]-Grid_y[i][j]]
    else:
        v1 = [Grid_x[i][j]-Grid_x[i-1][j],Grid_y[i][j]-Grid_y[i-1][j]]
        
    if 0 < j+1 <Size[1] and not grid_mask[i][j+1]:
        v2 = [Grid_x[i][j+1]-Grid_x[i][j],Grid_y[i][j+1]-Grid_y[i][j]]
    else:
        v2 = [Grid_x[i][j]-Grid_x[i][j-1],Grid_y[i][j]-Grid_y[i][j-1]]
        
    inner_product = v1[0]*v2[0] + v1[1]*v2[1]
    norm = np.sqrt( (v1[0]**2+v1[1]**2)*(v2[0]**2+v2[1]**2) )
    value_cos = inner_product/(norm+0.0000001)
    
    return value_cos    



def loss_area(next_point,size,grid_x,grid_y):
    i = next_point[0]
    j = next_point[1]
    grid_mask = grid_x.mask
    
    p0 = [grid_x[i][j],grid_y[i][j]]
    
    if 0 < j+1 <size[1] and not grid_mask[i][j+1]:
        p1 = [grid_x[i][j+1],grid_y[i][j+1]]
    else:
        p1 = [grid_x[i][j-1],grid_y[i][j-1]]
        
    if 0 < i+1 <size[0] and not grid_mask[i+1][j]:
        p3 = [grid_x[i+1][j],grid_y[i+1][j]]
    else:
        p3 = [grid_x[i-1][j],grid_y[i-1][j]]
        
    p2 = [p3[0],p1[1]]
    
    areas = area(p0,p1,p3) + area(p1,p2,p3)
    return areas
    

def loss_location(next_point, direction):
    #return the next direction(location)
    '''
          (i-1,j)
  (i,j-1)  (i,j)   (i,j+1)
          (i+1,j)
    '''
    #next_point : (i,j)
    #direction: 'up','down','left','right'
    i = next_point[0]
    j = next_point[1]
    
    if direction == 'up':    return [(i,j+1),(i-1,j),(i,j-1),(i+1,j)]
    if direction == 'down':  return [(i,j-1),(i+1,j),(i,j+1),(i-1,j)]
    if direction == 'right': return [(i+1,j),(i,j+1),(i-1,j),(i,j-1)]
    if direction == 'left':  return [(i-1,j),(i,j-1),(i+1,j),(i,j+1)]

        
def loss_direction(direction, cnt):
    #return the next direction
    def convert(DIRE):
        if DIRE == 'right':  return 'left'
        if DIRE == 'left':   return 'right'
        if DIRE == 'down':   return 'up'
        if DIRE == 'up':     return 'down'
        
    temp = ['up','left','down','right']
    
    if direction == 'up':    return convert(temp[(0+cnt)%4])
    if direction == 'right': return convert(temp[(3+cnt)%4])
    if direction == 'down':  return convert(temp[(2+cnt)%4])
    if direction == 'left':  return convert(temp[(1+cnt)%4])       
        
        
        
    
    