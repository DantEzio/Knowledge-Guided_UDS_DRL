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
import shutil
from tensorflow.keras import backend as K
import time

Train=True
init_train=True

tf.compat.v1.reset_default_graph()
env_params = {
    'orf':os.path.dirname(os.getcwd())+'/SWMM/chaohu',
    'orf_save':'Astlingen_RTC',# opt中使用不同的inp计算，上两个参数没有用到
    'parm':os.path.dirname(os.getcwd())+'/states_yaml/chaohu',
    'advance_seconds':300,
    'kf':1,
    'kc':1,
    'reward_type':'3',
}
env=SWMM_ENV.SWMM_ENV(env_params)

raindata = np.load(os.path.dirname(os.getcwd())+'\\rainfall\\design_rainfall\\training_raindata.npy').tolist()

agent_params={
    'state_dim':len(env.config['states']),
    'action_dim':pd.read_csv(os.path.dirname(os.getcwd())+'/SWMM/action_table.csv').values[:,1:].shape[0],
    'actornet_layer':[50,50,50],
    'criticnet_layer':[50,50,50],
    
    'bound_low':0,
    'bound_high':1,
    
    'clip_ratio':0.01,
    'target_kl':0.03,
    'lam':0.01,
    
    'policy_learning_rate':0.001,
    'value_learning_rate':0.001,
    'train_policy_iterations':5,
    'train_value_iterations':5,
    
    'num_rain':50,
    
    'training_step':501,
    'gamma':0.01,
    'epsilon':1,
    'ep_min':1e-50,
    'ep_decay':0.1,

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
    observation, episode_return, episode_length = env.reset(raindata[i],i,True,os.path.dirname(os.getcwd())), 0, 0
    
    done = False
    while not done:
        # Get the logits, action, and take one step in the environment
        observation = np.array(observation).reshape(1, -1)
        logits, action = PPO.sample_action(observation,tem_model,True)
        at = tem_model.action_table[int(action[0].numpy())].tolist()
        observation_new, reward, results, done = env.step(at)
        episode_return += reward
        episode_length += 1

        # Get the value and log-probability of the action
        value_t = tem_model.critic(observation).numpy()[0]
        logprobability_t = PPO.logprobabilities(logits, action, tem_model.params['action_dim']).numpy()[0]
        
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

    # plasticity input X
    #Xdataobs=np.load('../pretraindata/observation_buffer.npy',allow_pickle=True)
    #Xdataact=np.load('../pretraindata/action_buffer.npy',allow_pickle=True)
    #Xdataadv=np.load('../pretraindata/advantage_buffer.npy',allow_pickle=True)
    #Xdatalog=np.load('../pretraindata/logprobability_buffer.npy',allow_pickle=True)

    t1 = time.perf_counter()
    t2=[]

    # main training process   
    history = {'episode': [], 'Batch_reward': [], 'Episode_reward': [], 'Loss': [], 'Dormant Neuron number':[], 'b':[]}
    
    # Iterate over the number of epochs
    for epoch in range(model.params['training_step']):
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        
        # Initialize the buffer
        buffer = Buffer.Buffer(model.params['state_dim'], int(len(raindata[0])*model.params['num_rain']))
        
        # Iterate over the steps of each epoch
        # Parallel method in joblib
        #res = Parallel(n_jobs=20)(delayed(interact)(i,model.params['epsilon']) for i in range(model.params['num_rain']))

        # use prepared data buffer
        res = []
        tem_model = PPO.PPO(agent_params)
        tem_model.load_model('./model/')
        tem_model.params['epsilon']=model.params['epsilon']
        for rainid in range(50):
            episode_return, episode_length = 0, 0

            os.chdir(os.path.pardir)
            os.chdir(os.path.pardir)
            #print(os.getcwd())
            os.chdir('./step0-knowledge/')
            optdata = np.load('./DNv2_results/dnv2_results_rain'+str(rainid)+'.npy',allow_pickle=True).tolist()
            os.chdir(os.path.pardir)
            #print(os.getcwd())
            os.chdir('./step1-pretrainK2/Results_PPO_reward3')
            #print(os.getcwd())

            o=optdata['state'][1:-2]
            a=optdata['action'][1:-1]
            r=optdata['rewards'][1:-2]
            o_=optdata['state'][2:-1]
            last_value=0
            vt, lo = [], []
            for k in range(len(o)):
                observation = o[k]
                value_t = tem_model.critic(np.array(observation).reshape(1, -1))
                #gammas = np.array([np.power(agent_params['gamma'],t) for t in range(len(r[k:]))])
                #value_t = np.sum(np.array(r[k:])*gammas)
                logits, action = PPO.sample_action(np.array(observation).reshape(1, -1),tem_model,True)
                logprobability_t = PPO.logprobabilities(logits, action, tem_model.params['action_dim'])
                vt.append(value_t)
                lo.append(logprobability_t)

            episode_return += np.sum(r)
            episode_length += len(r)
            res.append([o,a,r,vt,lo,last_value,episode_return,episode_length])
        
        for i in range(model.params['num_rain']):
            #s, a, r, vt, lo, lastvalue in buffer
            for o,a,r,vt,lo in zip(res[i][0],res[i][1],res[i][2],res[i][3],res[i][4]):
                #buffer.store(o,a,r,vt,lo) # For OPT data
                #For EFD data
                atem = 0
                for it in range(agent_params['action_table'].shape[0]):
                    if (agent_params['action_table'][it] == np.array(a)).all():
                        atem = it
                buffer.store(o,atem,r,vt,lo)
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
            kl, policy_loss = PPO.train_policy(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer, model)
            value_loss = PPO.train_value_function(observation_buffer, return_buffer, model)
            history['Loss'].append(policy_loss+value_loss)

        # Get dormant unit number
        mid_output = []
        for k in range(1,len(model.params['actornet_layer'])):
            rp = K.function(inputs=[model.actor.layers[0].input],outputs=[model.actor.layers[k].output])
            mid_results = rp(observation_buffer)[0]
            domarnt = np.mean(np.abs(mid_results),axis=0)
            mid_output += domarnt.tolist()
        b = np.sum(mid_output)
        b1 = np.array(mid_output)/b
        count = np.sum(b1<0.01)
        domarnt_rate = count/np.array(mid_output).shape[0]
        print(b,count,domarnt_rate)
        history['Dormant Neuron number'].append(domarnt_rate)
        history['b'].append(b)
        
        model.critic.save_weights('./model/PPOcritic.h5')
        model.actor.save_weights('./model/PPOactor.h5')
        if epoch % 10 == 0:
            model.critic.save_weights('./model/PPOcritic'+str(epoch)+'.h5')
            model.actor.save_weights('./model/PPOactor'+str(epoch)+'.h5')

        # log training results
        history['episode'].append(epoch)
        history['Episode_reward'].append(sum_return)
        # reduce the epsilon egreedy and save training log
        if model.params['epsilon'] >= model.params['ep_min'] and epoch % 3 == 0:
            model.params['epsilon'] *= model.params['ep_decay']
        
        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Loss: {history['Loss'][-1]}. Return: {sum_return}. Mean Length: {sum_length / num_episodes}. Epsilon: {model.params['epsilon']}"
        )
        
        np.save('./Results/Train.npy',history)
        if epoch % 10 == 0:
            t2.append(time.perf_counter())

    timeR=[ti-t1 for ti in t2]
    print('Running time: ',timeR)
    np.save('./time.npy',timeR)

    
    # plot
    plt.figure()
    plt.plot(history['Loss'])
    plt.savefig('./Results/Train.tif')

    
###############################################################################
# end Train
###############################################################################
def delete_folder_contents(folder_path):
    try:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                shutil.rmtree(dir_path)
        print(f"文件夹 {folder_path} 内的所有文件和子文件夹已成功删除。")
    except FileNotFoundError:
        print(f"错误: 文件夹 {folder_path} 未找到。")
    except PermissionError:
        print(f"错误: 没有权限删除文件夹 {folder_path} 内的文件或子文件夹。")
    except Exception as e:
        print(f"发生未知错误: {e}")

folder_to_clean = ['../SWMM/_teminp','../SWMM/_temopt',
                   '../SWMM/_temopt_original/BC','../SWMM/_temopt_original/DN','../SWMM/_temopt_original/DNv2',
                   '../SWMM/_temopt_original/EFD']
for f in folder_to_clean:
    delete_folder_contents(f)