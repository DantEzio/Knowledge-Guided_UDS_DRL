# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 20:10:48 2022

@author: chong
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)

class DQN:
    def __init__(self,params):
        self.params=params
        self.action_table=params['action_table']
        
        # Initialize models
        self.optimizer = Adam(learning_rate=self.params['learning_rate'])

        self.observation_input = keras.Input(shape=(self.params['state_dim'],), dtype=tf.float32, name='sc_input')
        self.Q = mlp(self.observation_input, self.params['encoding_layer']+self.params['value_layer']+[self.params['action_dim']], tf.tanh, None)
        self.model = keras.Model(inputs=self.observation_input, outputs=self.Q)
        self.target_model = keras.Model(inputs=self.observation_input, outputs=self.Q)

        '''
        self.encoding = mlp(self.observation_input, self.params['encoding_layer'], tf.tanh, None)
        self.V = mlp(self.encoding, self.params['value_layer']+[1], tf.tanh, None)
        self.A = mlp(self.encoding, self.params['advantage_layer']+[self.params['action_dim']], tf.tanh, None)
        self.Q = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keepdims=True))
        
        self.model = keras.Model(inputs=self.observation_input, outputs=self.Q)
        self.target_model = keras.Model(inputs=self.observation_input, outputs=self.Q)
        '''
                              
    def load_model(self,file):
        self.model.load_weights(file+'/dqn.h5')
        self.target_model.load_weights(file+'/target_dqn.h5')

# 
def sample_action(observation,model,train_log):
    #input state, output action
    if train_log:
        #epsilon greedy
        pa = np.random.uniform()
        if model.params['epsilon'] < pa:
            action_value = model.model(observation)
            action = tf.argmax(action_value,axis=1)
            #action = tf.squeeze(tf.random.categorical(action_value, 1), axis=1)
        else:
            logits = tf.compat.v1.random_normal([1, model.action_table.shape[0]], mean=0, stddev=1)
            action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    else:
        action_value = model.model(observation)
        action = tf.argmax(action_value,axis=1)
        #action = tf.squeeze(tf.random.categorical(action_value, 1), axis=1)
    return action

def train_value(observation_buffer, action_buffer, reward_buffer, observation_next_buffer, model):
    with tf.GradientTape() as tape:
        y = model.model(observation_buffer).numpy()
        q = model.target_model(observation_next_buffer).numpy()
        for i in range(observation_buffer.shape[0]):
            target = reward_buffer[i] + model.params['gamma'] * np.amax(q[i])
            y[i][action_buffer[i]] = target
        
        loss = tf.reduce_mean((model.model(observation_buffer)-y)**2)
        grads = tape.gradient(loss, model.model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.model.trainable_variables))
    return loss
    