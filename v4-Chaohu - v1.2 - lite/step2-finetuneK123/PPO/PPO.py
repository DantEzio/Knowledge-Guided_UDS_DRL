# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 21:10:39 2023

@author: chongtm
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


def logprobabilities(logits, a, num_actions):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability

class PPO:
    def __init__(self,params):
        #tf.compat.v1.disable_eager_execution()
        self.params=params
        self.action_table=params['action_table']      
        
        # Initialize the policy and the value function optimizers
        self.policy_optimizer = Adam(learning_rate=self.params['policy_learning_rate'])
        self.value_optimizer = Adam(learning_rate=self.params['value_learning_rate'])
        
        self.observation_dimensions = self.params['state_dim']
        self.num_actions = self.params['action_dim']

        # Initialize the actor and the critic as keras models
        self.observation_input = keras.Input(shape=(self.params['state_dim'],), dtype=tf.float32, name='sc_input')
        self.logits = mlp(self.observation_input, self.params['actornet_layer']+[self.num_actions], tf.tanh, None)
        self.actor = keras.Model(inputs=self.observation_input, outputs=self.logits)
        self.value = tf.squeeze(mlp(self.observation_input, self.params['criticnet_layer']+[1], tf.tanh, None), axis=1)
        self.critic = keras.Model(inputs=self.observation_input, outputs=self.value)
    
    def load_model(self,file,finetuneID):
        self.critic.load_weights(file+'/PPOcritic'+finetuneID+'.h5')
        self.actor.load_weights(file+'/PPOactor'+finetuneID+'.h5')
        
        
# Sample action from actor
@tf.function
def sample_action(observation,model,train_log):
    if train_log:
        #epsilon greedy
        pa = np.random.uniform()
        if pa > model.params['epsilon']:
            logits = model.actor(observation)
            action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        else:
            logits = tf.compat.v1.random_normal([1, model.action_table.shape[0]], mean=0, stddev=1)
            action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
            
    else:    
        logits = model.actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action


# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(observation_buffer, action_buffer, logprobability_buffer, advantage_buffer, model):

    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            logprobabilities(model.actor(observation_buffer), action_buffer, model.params['action_dim'])
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + model.params['clip_ratio']) * advantage_buffer,
            (1 - model.params['clip_ratio']) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, model.actor.trainable_variables)
    model.policy_optimizer.apply_gradients(zip(policy_grads, model.actor.trainable_variables))

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(model.actor(observation_buffer), action_buffer, model.params['action_dim'])
    )
    kl = tf.reduce_sum(kl)
    return kl


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer,model):
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        value_loss = tf.reduce_mean((return_buffer - model.critic(observation_buffer)) ** 2)
    value_grads = tape.gradient(value_loss, model.critic.trainable_variables)
    model.value_optimizer.apply_gradients(zip(value_grads, model.critic.trainable_variables))
    
    
        
        
        
        
        