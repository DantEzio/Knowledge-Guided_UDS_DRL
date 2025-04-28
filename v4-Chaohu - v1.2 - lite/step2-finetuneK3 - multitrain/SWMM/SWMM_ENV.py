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


def get_step_results(results,nodes,links,rgs,sys,config,params):
    
    #获取reward
    delt_flooding,delt_CSO,CSOtem,delt_Qtw=0,0,0,0
    for _temp in config['reward_targets']:
        if _temp[1] == 'flooding':
            if _temp[0] == 'system':
                delt_flooding += sys.routing_stats[_temp[1]] - results['flooding'][-1]
            else:
                delt_flooding += nodes[_temp[0]].statistics['flooding_volume']
            results['flooding'].append(sys.routing_stats[_temp[1]])
        else:
            #cum_cso = sys.routing_stats['outflow']
            CSOtem += nodes[_temp[0]].cumulative_inflow
            # log the cumulative value
            #self.data_log[_temp[1]][_temp[0]].append(cum_cso)
    delt_CSO = CSOtem - results['CSO'][-1]
    results['CSO'].append(CSOtem)
            
    Qtw = (sys.routing_stats['dry_weather_inflow']
                +sys.routing_stats['wet_weather_inflow']
                +sys.routing_stats['groundwater_inflow']
                +sys.routing_stats['II_inflow'])
    delt_Qtw = Qtw - results['inflow'][-1]
    results['inflow'].append(Qtw)
    
    #flooding time and cso time
    floodingt, delt_flooding_time = 0,0
    for n in nodes:
        if n.statistics['flooding_duration'] >= floodingt:
            floodingt = n.statistics['flooding_duration']
    delt_flooding_time = floodingt - results['total_flooding_time'][-1]
    results['total_flooding_time'].append(floodingt)

    delt_cso_time = 0
    if delt_CSO > 0:
        delt_cso_time = params['advance_seconds']/3600
    results['total_CSO_time'].append(results['total_CSO_time'][-1]+delt_cso_time)

    time = len(results['total_CSO_time']) * 1

    # 3 types of reward
    if params['reward_type'] == '1':
        #if results['inflow'][-1] == 0:
        #    reward = 0
        #else:
        #    reward = - delt_flooding_time / params['advance_seconds'] - delt_cso_time / params['advance_seconds']
        if results['inflow'][-1] == 0:
            reward = 0
        else:
            reward = - delt_flooding / delt_Qtw - delt_CSO / delt_Qtw
    elif params['reward_type'] == '2':
        if results['inflow'][-1] == 0:
            reward = 0
        else:
            reward = - (1/(results['inflow'][-1])) * ((delt_flooding * results['total_flooding_time'][-1] + delt_flooding_time * results['flooding'][-1]) 
                                        + (delt_CSO * results['total_CSO_time'][-1] + delt_cso_time * results['CSO'][-1]))
    elif params['reward_type'] == '3':
        if results['inflow'][-1] == 0 or delt_Qtw == 0:
            reward = 0
        else:
            reward = 1 / (1 + delt_flooding/delt_Qtw + delt_CSO/delt_Qtw) - 1
    
    else:
        # maybe the real derivate
        if results['inflow'][-1] == 0:
            reward = 0
        else:
            df_flooding = delt_flooding * results['total_flooding_time'][-1] + delt_flooding_time * results['flooding'][-1]
            f_flooding = results['total_flooding_time'][-1] * results['flooding'][-1]
            dm_flooding = Qtw + delt_Qtw * time
            m_flooding = Qtw * time
            dsev_flooding = (df_flooding * m_flooding - dm_flooding * f_flooding)/(m_flooding * m_flooding)

            df_cso = delt_CSO * results['total_CSO_time'][-1] + delt_cso_time * results['CSO'][-1]
            f_cso = results['total_CSO_time'][-1] * results['CSO'][-1]
            dm_cso = Qtw + delt_Qtw * time
            m_cso = Qtw * time
            dsev_cso = (df_cso * m_cso - dm_cso * f_cso)/(m_cso * m_cso)

            reward = -dsev_flooding - dsev_cso
            # 所有flow差值越低越好
    
    return results, reward


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
    
    def reset(self,rain,rainid,trainlog,root):
        
        if trainlog:
            root += '/SWMM/_teminp'
        else:
            root += '/SWMM/_temtestinp'
        inp = read_inp_file(self.params['orf']+'.inp')
        inp[TIMESERIES]['rainfall']=TimeseriesData('rainfall',rain[rainid])
        inp.write_file(root+'/'+self.params['orf_save']+str(rainid)+'_rain.inp')
        self.sim=Simulation(root+'/'+self.params['orf_save']+str(rainid)+'_rain.inp')
        self.sim.start()
        
        #模拟一步
        if self.params['advance_seconds'] is None:
            self.sim._model.swmm_step()
        else:
            self.sim._model.swmm_stride(self.params['advance_seconds'])
                
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

        # 获取结果，记录cso、flooding
        self.results = {}
        
        #记录总体cso, flooding, res
        self.results['CSO'], self.results['flooding'], self.results['inflow'] = [0], [0], [0]
        self.results['total_flooding_time'], self.results['total_CSO_time'] = [0], [0]
        self.results['res'] = [0]
        self.results['state'], self.results['action'], self.results['rewards'] = [], [], []
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

        #获取reward和结果
        self.results, reward = get_step_results(self.results,nodes,links,rgs,sys,self.config,self.params)

        self.results['state'].append(states)
        self.results['action'].append(action)
        self.results['rewards'].append(reward)
            
        #降雨结束检测
        if done:
            self.sim._model.swmm_end()
            self.sim._model.swmm_close()
        return states,reward,self.results,done
        
            
