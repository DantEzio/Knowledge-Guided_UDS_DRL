# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:39:01 2022

@author: chong
"""
import numpy as np
import SWMM_ENV
import Buffer
import PPO as PPO
import Rainfall_data as RD
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

#state_list=['state_1_1','state_2_1','state_2_2','state_3_1']
state_list=['state_3_2','state_4_1','state_4_2']
inptype_list=['flwl']#['wl','fl','flwl']

Train=True
init_train=True

for sl in state_list:
    for il in inptype_list:
        print(sl,il)

        tf.compat.v1.reset_default_graph()
        env_params={
                'orf':'chaohu',
                'parm':'./states_yaml/'+il+'/'+sl,
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
            
            'num_rain':10,
            
            'training_step':1000,
            'gamma':0.3,
            'epsilon':1,
            'ep_min':1e-50,
            'ep_decay':0.1
        }
        model = PPO.PPO(agent_params)
        if init_train:
            model.critic.save_weights('./model/'+il+'/'+sl+'/PPOcritic.h5')
            model.actor.save_weights('./model/'+il+'/'+sl+'/PPOactor.h5')
        model.load_model('./model/'+il+'/'+sl)


        ###############################################################################
        # Train
        ###############################################################################
            
        def interact(i,ep):   
            env=SWMM_ENV.SWMM_ENV(env_params)
            tem_model = PPO.PPO(agent_params)
            tem_model.load_model('./model/'+il+'/'+sl)
            tem_model.params['epsilon']=ep
            s,a,r,vt,lo = [],[],[],[],[]
            observation, episode_return, episode_length = env.reset(raindata[i],i,True), 0, 0
            
            done = False
            while not done:
                # Get the logits, action, and take one step in the environment
                observation = np.array(observation).reshape(1, -1)
                logits, action = PPO.sample_action(observation,tem_model,True)
                at = tem_model.action_table[int(action[0].numpy())].tolist()
                observation_new, reward, flooding,CSO,done = env.step(at)
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
                buffer = Buffer.Buffer(model.params['state_dim'], int(len(raindata[0])*model.params['num_rain']))
                
                # Iterate over the steps of each epoch
                # Parallel method in joblib
                res = Parallel(n_jobs=10)(delayed(interact)(i,model.params['epsilon']) for i in range(model.params['num_rain'])) 
                
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
                
                
                #model.critic.save_weights('./model/PPOcritic_'+str(epoch)+'.h5')
                #model.actor.save_weights('./model/PPOactor_'+str(epoch)+'.h5')
                model.critic.save_weights('./model/'+il+'/'+sl+'/PPOcritic.h5')
                model.actor.save_weights('./model/'+il+'/'+sl+'/PPOactor.h5')
                # log training results
                history['episode'].append(epoch)
                history['Episode_reward'].append(sum_return)
                # reduce the epsilon egreedy and save training log
                if model.params['epsilon'] >= model.params['ep_min'] and epoch % 3 == 0:
                    model.params['epsilon'] *= model.params['ep_decay']
                    
                    
                
                # Print mean return and length for each epoch
                print(
                    f" Epoch: {epoch + 1}. Return: {sum_return}. Mean Length: {sum_length / num_episodes}"
                )
                
                np.save('./Results/'+il+'/'+sl+'/Train.npy',history)
            
            # plot
            plt.figure()
            plt.plot(history['Episode_reward'])
            plt.savefig('./Results/'+il+'/'+sl+'/Train.tif')

            
        ###############################################################################
        # end Train
        ###############################################################################

            
        # test PPO agent
        def test(model,rain,i):
            # simulation on given rainfall
            env=SWMM_ENV.SWMM_ENV(env_params)
            test_history = {'time':[] ,'state': [], 'action': [], 'reward': [], 'F':[], 'C':[]}
            observation = env.reset(rain,i,False)
            done, t= False, 0
            test_history['time'].append(t)
            test_history['state'].append(observation)
            while not done:
                observation = np.array(observation).reshape(1, -1)
                logits, action = PPO.sample_action(observation,model,False)
                at=model.action_table[int(action[0].numpy())].tolist()
                observation_new,reward,F,C,done = env.step(at)
                observation = observation_new
                t +=1
                
                test_history['time'].append(t)
                test_history['state'].append(observation)
                test_history['action'].append(action)
                test_history['reward'].append(reward)
                test_history['F'].append(F)
                test_history['C'].append(C)
            
            return test_history


        raindata = np.load('test_raindata.npy').tolist()
        model.load_model('./model/'+il+'/'+sl)
        print(sl,il)
        for i in range(len(raindata)):
            test_his = test(model,raindata[i],i)
            np.save('./Results/'+il+'/'+sl+'/'+str(i)+'.npy',test_his)
