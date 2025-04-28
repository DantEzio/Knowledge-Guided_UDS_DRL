# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 22:41:39 2022

@author: chong

SWMM environment
can be used for any inp file
established based pyswmm
"""
import os
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE']="1"
import numpy as np
#import pyswmm.toolkitapi as tkai

from swmm_api.input_file import read_inp_file
from pyswmm import Simulation,Links,Nodes,RainGages,SystemStats
from swmm_api.input_file.sections.others import TimeseriesData
from swmm_api.input_file.section_labels import TIMESERIES

import matplotlib.pyplot as plt
import datetime
import yaml
import shutil 


class SWMM_ENV:
    #can be used for every SWMM inp
    def __init__(self,params):
        '''
        params: a dictionary with input
        orf: original file of swmm inp
        control_asset: list of contorl objective, pumps' name
        advance_seconds: simulation time interval
        flood_nodes: selected node for flooding checking
        '''
        self.params = params
        self.config = yaml.load(open(self.params['parm']+".yaml"), yaml.FullLoader)
        #self.t=[]
    
    def reset(self,rain,i,trainlog):
        if trainlog:
            root='_teminp'
        else:
            root='_temtestinp'
        shutil.copyfile(self.params['orf']+'.inp', './'+root+'/'+self.params['orf']+str(i)+'.inp')
        inp = read_inp_file('./'+root+'/'+self.params['orf']+str(i)+'.inp')
        inp[TIMESERIES]['rainfall']=TimeseriesData('rainfall',rain)
        inp.write_file('./'+root+'/'+self.params['orf']+str(i)+'_rain.inp')
        self.sim=Simulation('./'+root+'/'+self.params['orf']+str(i)+'_rain.inp')
        self.sim.start()
        
        #模拟一步
        if self.params['advance_seconds'] is None:
            self.sim._model.swmm_step()
        else:
            self.sim._model.swmm_stride(self.params['advance_seconds'])
        
        #记录总体cso和flooding
        self.CSO,self.flooding=0,0
        
        #obtain states and reward term by yaml (config)
        nodes = Nodes(self.sim)
        links = Links(self.sim)
        rgs = RainGages(self.sim)
        states = []
        for _temp in self.config["states"]:
            if _temp[1] == 'depthN':
                states.append(nodes[_temp[0]].depth)
            elif _temp[1] == 'flow':
                states.append(links[_temp[0]].flow)
            elif _temp[1] == 'inflow':
                states.append(nodes[_temp[0]].total_inflow)
            else:
                states.append(rgs[_temp[0]].rainfall)
        return states
        
    def step(self,action):
        #获取模拟结果
        nodes = Nodes(self.sim)
        links = Links(self.sim)
        rgs = RainGages(self.sim)
        sys = SystemStats(self.sim)
        #obtain states and reward term by yaml (config)
        states = []
        for _temp in self.config["states"]:
            if _temp[1] == 'depthN':
                states.append(nodes[_temp[0]].depth)
            elif _temp[1] == 'flow':
                states.append(links[_temp[0]].flow)
            elif _temp[1] == 'inflow':
                states.append(nodes[_temp[0]].total_inflow)
            else:
                states.append(rgs[_temp[0]].rainfall)
            
        
        #设置控制
        for item,a in zip(self.config['action_assets'],action):
            links[item].target_setting = a
        
        
        #模拟一步
        if self.params['advance_seconds'] is None:
            time = self.sim._model.swmm_step()
        else:
            time = self.sim._model.swmm_stride(self.params['advance_seconds'])
        #self.t.append(self.sim._model.getCurrentSimulationTime())
        done = False if time > 0 else True
        
        #获取reward
        flooding,CSO,CSOtem,inflow=0,0,0,0
        for _temp in self.config['reward_targets']:
            if _temp[1] == 'flooding':
                if _temp[0] == 'system':
                    flooding += sys.routing_stats[_temp[1]]-self.flooding
                else:
                    flooding += nodes[_temp[0]].statistics['flooding_volume']
                
                self.flooding = sys.routing_stats[_temp[1]]
                #flooding += (cum_flooding - self.data_log[_temp[1]][_temp[0]][-1]) *_temp[2]
                # log the cumulative value
                #self.data_log[_temp[1]][_temp[0]].append(cum_flooding)
            else:
                
                '''
                if _temp[0] == 'system':
                    cum_cso = sys.routing_stats['outflow']
                else:
                    cum_cso = nodes[_temp[0]].volume
                '''
                #cum_cso = sys.routing_stats['outflow']
                CSOtem += nodes[_temp[0]].cumulative_inflow
                # log the cumulative value
                #self.data_log[_temp[1]][_temp[0]].append(cum_cso)
                
        inflow = (sys.routing_stats['dry_weather_inflow']
                  +sys.routing_stats['wet_weather_inflow']
                  +sys.routing_stats['groundwater_inflow']
                  +sys.routing_stats['II_inflow'])
        
        
        CSO = CSOtem - self.CSO
        self.CSO = CSOtem
        rewards = -(flooding+CSO)/inflow
        #rewards = np.exp(-(flooding/inflow)**2/0.01) + np.exp(-(CSO/inflow)**2/0.01)
        
        #降雨结束检测
        if done:
            self.sim._model.swmm_end()
            self.sim._model.swmm_close()
        return states,rewards,self.flooding,self.CSO,done
        
            
if __name__=='__main__':
    params={
            'orf':'chaohu',
            'advance_seconds':300
           }
    env=SWMM_ENV(params)
    
    #prepare rainfall
    data=[]
    for t in range(120):
        if t//60==0:
            data.append((datetime.datetime(2015, 8, 28, 9, np.mod(t,60)),0.1))
        elif t//60==1:
            data.append((datetime.datetime(2015, 8, 28, 10, np.mod(t,60)),0.1))
            
    env.reset(data)
    
    done = False
    states,actions,rewards=[],[],[]
    t=[]
    while not done:
        action = [0.5 for _ in range(len(env.config['action_assets']))]
        s,r,done = env.step(action)
        states.append(s)
        actions.append(action)
        rewards.append(r)
        
    plt.plot(states)
