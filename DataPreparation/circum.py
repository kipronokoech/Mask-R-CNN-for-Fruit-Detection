#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
from math import pi, cos, sin


def xy_values(xx,yy,radius,width,height,n=98):
    def point(x, y, r,angle):
        theta = angle * 2 * pi
        xxx = int(x + cos(theta) * r)
        if xxx<0:
            xxx = 0
        if xxx>=width:
            xxx = width-1
        yyy = int(y + sin(theta) * r)
        if yyy<0:
            yyy = 0
        if yyy>=height:
            yyy = height-1
        return xxx, yyy
    angle = np.linspace(0,1,n)
    angle = np.append(angle,angle[0])
    xy = [point(xx,yy,radius,angle[i]) for i in range(n)]
    x,y = [],[]
    for i,j in xy:
        x.append(i)
        y.append(j)
    return x,y

