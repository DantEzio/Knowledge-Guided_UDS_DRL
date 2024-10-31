# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 15:52:53 2022

@author: chong
"""

import numpy as np
import matplotlib.pyplot as plt


# This hyetograph is correct in a continous function, but incorrect with 5-min block.
#It start from 9:00.
def Chicago_Hyetographs(para_tuple):
    A,C,n,b,r,P,delta,dura = para_tuple
    a = A*(1+C*np.log10(P))
    tsd = []
    for i in range(dura//delta):
        t = i*delta
        key = str(9+t//60).zfill(2)+':'+str(t % 60).zfill(2)
        if t <= r*dura:
            tsd.append((key,(a*((1-n)*(r*dura-t)/r+b)/((r*dura-t)/r+b)**(1+n))*60))
        else:
            tsd.append((key,(a*((1-n)*(t-r*dura)/(1-r)+b)/((t-r*dura)/(1-r)+b)**(1+n))*60))
    # tsd = TimeseriesData(Name = name,data = ts)
    return tsd

# Generate a rainfall intensity file from a cumulative values in ICM
#It start from 9:00.
def Chicago_icm(para_tuple):
    A,C,n,b,r,P,delta,dura = para_tuple
    a = A*(1+C*np.log10(P))
    HT = a*dura/(dura+b)**n
    Hs = []
    for i in range(dura//delta+1):
        t = i*delta
        if t <= r*dura:
            H = HT*(r-(r-t/dura)*(1-t/(r*(dura+b)))**(-n))
        else:
            H = HT*(r+(t/dura-r)*(1+(t-dura)/((1-r)*(dura+b)))**(-n))
        Hs.append(H)
    tsd = np.diff(15*np.array(Hs))*60/delta
    ts = []
    for i in range(dura//delta):
        t = i*delta
        key = '08/28/2015 '+str(9+t//60).zfill(2)+':'+str(t % 60).zfill(2)+':'+'00'
        ts.append([key,tsd[i]])
    return ts

if __name__=='__main__':
    
    A,C,n,b,r,P,delta,dura = 23,4,0.03,0.2,0.5,10,1,120
    para_tuple = (A,C,n,b,r,P,delta,dura)
    ts=Chicago_Hyetographs(para_tuple)
    #print(t1[:][1])
    t2 = Chicago_icm(para_tuple)
    print(t2)