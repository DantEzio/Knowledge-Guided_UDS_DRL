# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:39:01 2022

@author: chong
"""
import numpy as np
import SWMM_ENV
import PPO as PPO
import Rainfall_data as RD

import datetime
import matplotlib.pyplot as plt


env_params={
        'orf':'chaohu',
        'advance_seconds':300
    }
env=SWMM_ENV.SWMM_ENV(env_params)

raindata = np.load('training_raindata.npy').tolist()

agent_params={
    'state_dim':len(env.config['states']),
    'action_dim':int(2**len(env.config['action_assets'])),
    'actornet_layer':[30,30,30],
    'criticnet_layer':[30,30,30],
    
    'bound_low':0,
    'bound_high':1,
    
    'clip_ratio':0.01,
    'target_kl':0.03,
    'lam':0.01,
    
    'policy_learning_rate':0.001,
    'value_learning_rate':0.001,
    'train_policy_iterations':20,
    'train_value_iterations':20,
    
    'num_rain':50,
    
    'training_step':1800,
    'gamma':0.3,
    'epsilon':1e-40,
    'ep_min':1e-50,
    'ep_decay':0.1
}
model = PPO.PPO(agent_params,env)
model.load_model()

history = PPO.train(model,raindata)
np.save('./Results/Train.npy',history)

raindata = np.load('test_raindata.npy').tolist()
model.load_model()

for i in range(len(raindata)):
    test_his = PPO.test(model,raindata[i])
    np.save('./Results/'+str(i)+'.npy',test_his)
