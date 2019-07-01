#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

left = -98.83
right = -68.58
down = 15.39
up = 35.11

map = Basemap(llcrnrlon=left,llcrnrlat=down,urcrnrlon=right,urcrnrlat=up,
             resolution='i', projection='merc', lat_0 = 0, lon_0 = 0.)
map.shadedrelief()
map.drawcoastlines()
plt.show()

x = [-90,-70]
y = [18,30]


lons, lats = map(x, y)
map.plot(lons, lats, marker='D',color='k')
plt.show()

#parallels = np.arange(33.,46.,10.)
## labels = [left,right,top,bottom]
#m.drawparallels(parallels,labels=[False,True,True,False])
#meridians = np.arange(10.,351.,20.)    






#
#from mpl_toolkits.basemap import Basemap
#import matplotlib.pyplot as plt
#import numpy as np
## setup Lambert Conformal basemap.
#m = Basemap(projection='merc', 
#              lat_0=0, lon_0=0,
#              llcrnrlon=-20.,llcrnrlat=0.,urcrnrlon=180.,urcrnrlat=80.)
## draw coastlines.
#m.drawcoastlines()
## draw a boundary around the map, fill the background.
## this background will end up being the ocean color, since
## the continents will be drawn on top.
#m.drawmapboundary(fill_color='aqua')
## fill continents, set lake color same as ocean color.
#m.fillcontinents(color='coral',lake_color='aqua')
## draw parallels and meridians.
## label parallels on right and top
## meridians on bottom and left
#parallels = np.arange(0.,81,10.)
## labels = [left,right,top,bottom]
#m.drawparallels(parallels,labels=[False,True,True,False])
#meridians = np.arange(10.,360.0,20.)
#m.drawmeridians(meridians,labels=[True,False,False,True])
#plt.show()
#
#
