# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:39:01 2022

@author: chong
"""
import sys 
sys.path.append("..") 

import numpy as np
import pandas as pd
from SWMM import SWMM_ENV
from PPO import Buffer
from PPO import PPO as PPO
import tensorflow as tf
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import os


Train=True
init_train=True

tf.compat.v1.reset_default_graph()
env_params = {
    'orf':os.path.dirname(os.getcwd())+'/SWMM/Astlingen_SWMM',
    'orf_save':'Astlingen_RTC',# opt中使用不同的inp计算，上两个参数没有用到
    'parm':os.path.dirname(os.getcwd())+'/states_yaml/Astlingen',
    'advance_seconds':300,
    'kf':1,
    'kc':1,
    'reward_type':'3',
}
env=SWMM_ENV.SWMM_ENV(env_params)

rain1 = np.load(os.path.dirname(os.getcwd())+'/rainfall/real_rainfall/rain1.npy').tolist()
rain2 = np.load(os.path.dirname(os.getcwd())+'/rainfall/real_rainfall/rain2.npy').tolist()
rain3 = np.load(os.path.dirname(os.getcwd())+'/rainfall/real_rainfall/rain3.npy').tolist()
rain4 = np.load(os.path.dirname(os.getcwd())+'/rainfall/real_rainfall/rain4.npy').tolist()
raindata={'1':rain1, '2':rain2, '3':rain3, '4':rain4}

agent_params={
    'state_dim':len(env.config['states']),
    'action_dim':pd.read_csv(os.path.dirname(os.getcwd())+'/SWMM/action_table.csv').values[:,1:].shape[0],
    'actornet_layer':[50,50],
    'criticnet_layer':[50,50],
    
    'bound_low':0,
    'bound_high':1,
    
    'clip_ratio':0.01,
    'target_kl':0.03,
    'lam':0.01,
    
    'policy_learning_rate':0.01,
    'value_learning_rate':0.01,
    'train_policy_iterations':20,
    'train_value_iterations':20,
    
    'num_rain':50,
    
    'training_step':1010,
    'gamma':0.3,
    'epsilon':1,
    'ep_min':1e-50,
    'ep_decay':0.05,

    'action_table':pd.read_csv(os.path.dirname(os.getcwd())+'/SWMM/action_table.csv').values[:,1:],
}
model = PPO.PPO(agent_params)
if init_train:
    model.critic.save_weights('./model/PPOcritic.h5')
    model.actor.save_weights('./model/PPOactor.h5')
model.load_model('./model/')

###############################################################################
# Train
###############################################################################
    
def interact(i,ep):   
    env=SWMM_ENV.SWMM_ENV(env_params)
    tem_model = PPO.PPO(agent_params)
    tem_model.load_model('./model/')
    tem_model.params['epsilon']=ep
    s,a,r,vt,lo = [],[],[],[],[]
    observation, episode_return, episode_length = env.reset(raindata,i,True,os.path.dirname(os.getcwd())), 0, 0
    #optdata = np.load(os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'\\OPT_results\\opt_results_rain'+str(i)+'.npy',allow_pickle=True).tolist()
    #o=optdata['state'][1:-2]
    #a=optdata['action'][1:-1]
    #optr=optdata['rewards'][:]
    
    done = False
    #step = 0
    while not done:
        # Get the logits, action, and take one step in the environment
        observation = np.array(observation).reshape(1, -1)
        logits, action = PPO.sample_action(observation,tem_model,True)
        at = tem_model.action_table[int(action[0].numpy())].tolist()
        observation_new, reward, results, done = env.step(at)

        #Knowledge informed reward
        #reward = reward - optr[step]
        #step += 1

        episode_return += reward
        episode_length += 1

        # Get the value and log-probability of the action
        value_t = tem_model.critic(observation)
        logprobability_t = PPO.logprobabilities(logits, action, tem_model.params['action_dim'])
        
        # Store obs, act, rew, v_t, logp_pi_t
        # buffer.store(observation, action, reward, value_t, logprobability_t)
        s.append(observation)
        a.append(action)
        r.append(reward)
        vt.append(value_t)
        lo.append(logprobability_t)
        
        # Update the observation
        observation = observation_new
    # Finish trajectory if reached to a terminal state
    last_value = 0 if done else tem_model.critic(observation.reshape(1, -1))
    return s,a,r,vt,lo,last_value,episode_return,episode_length

if Train:
    tf.config.experimental_run_functions_eagerly(True)

    # main training process   
    history = {'episode': [], 'Batch_reward': [], 'Episode_reward': [], 'Loss': []}
    
    # Iterate over the number of epochs
    for epoch in range(model.params['training_step']):
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        
        # Initialize the buffer
        buffer = Buffer.Buffer(model.params['state_dim'], int(8*60/5)*model.params['num_rain'])
        
        # Iterate over the steps of each epoch
        # Parallel method in joblib
        res = Parallel(n_jobs=20)(delayed(interact)(i,model.params['epsilon']) for i in range(model.params['num_rain'])) 
        
        for i in range(model.params['num_rain']):
            #s, a, r, vt, lo, lastvalue in buffer
            for o,a,r,vt,lo in zip(res[i][0],res[i][1],res[i][2],res[i][3],res[i][4]):
                buffer.store(o,a,r,vt,lo)
            buffer.finish_trajectory(res[i][5])
            sum_return += res[i][6]
            sum_length += res[i][7]
            num_episodes += 1
        
        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()
    
        # Update the policy and implement early stopping using KL divergence
        for _ in range(model.params['train_policy_iterations']):
            kl = PPO.train_policy(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer, model)
            #if kl > 1.5 * target_kl:
                ## Early Stopping
                #break
    
        # Update the value function
        for _ in range(model.params['train_value_iterations']):
            PPO.train_value_function(observation_buffer, return_buffer, model)
        
        
        model.critic.save_weights('./model/PPOcritic.h5')
        model.actor.save_weights('./model/PPOactor.h5')
        # log training results
        history['episode'].append(epoch)
        history['Episode_reward'].append(sum_return)
        # reduce the epsilon egreedy and save training log
        if model.params['epsilon'] >= model.params['ep_min'] and epoch % 3 == 0:
            model.params['epsilon'] *= model.params['ep_decay']
        
        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Return: {sum_return}. Mean Length: {sum_length / num_episodes}. Epsilon: {model.params['epsilon']}"
        )
        
        np.save('./Results/Train.npy',history)
    
    # plot
    plt.figure()
    plt.plot(history['Episode_reward'])
    plt.savefig('./Results/Train.tif')

    
###############################################################################
# end Train
###############################################################################