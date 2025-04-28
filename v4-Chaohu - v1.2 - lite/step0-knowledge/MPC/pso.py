 # -*- coding: utf-8 -*-
"""
Created on Thu May  5 07:57:43 2022

@author: chong
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt


# 0.0 coding:utf-8 0.0
def best(pop, fit_value):
    px = len(pop)
    best_individual = pop[0]
    best_fit = fit_value[0]
    for i in range(1, px):
        if(fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_individual = pop[i]
    return [best_individual, best_fit]

def calfitValue(obj_value):
    fit_value = []
    c_min = 0
    for i in range(len(obj_value)):
        if(obj_value[i] + c_min > 0):
            temp = c_min + obj_value[i]
        else:
            temp = 0.0
        fit_value.append(temp)
    return fit_value


def crossover(pop, pc):
    pop_len = len(pop)
    for i in range(pop_len - 1):
        if(random.random() < pc):
            cpoint = random.randint(0,len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0:cpoint])
            temp1.extend(pop[i+1][cpoint:len(pop[i])])
            temp2.extend(pop[i+1][0:cpoint])
            temp2.extend(pop[i][cpoint:len(pop[i])])
            pop[i] = temp1
            pop[i+1] = temp2


def geneEncoding(pop_size, chrom_length):
    pop = [[]]
    for i in range(pop_size):
        temp = []
        for j in range(chrom_length):
            temp.append(random.random())
        pop.append(temp)

    return pop[1:]


def mutation(pop, pm):
    px = len(pop)
    py = len(pop[0])
    
    for i in range(px):
        if(random.random() < pm):
            mpoint = random.randint(0, py-1)
            if(pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1


def sum(fit_value):
    total = 0
    for i in range(len(fit_value)):
        total += fit_value[i]
    return total


def cumsum(fit_value):
    for i in range(len(fit_value)-2, -1, -1):
        t = 0
        j = 0
        while(j <= i):
            t += fit_value[j]
            j += 1
        fit_value[i] = t
        fit_value[len(fit_value)-1] = 1


def selection(pop, fit_value):
    newfit_value = []
    # 适应度总和
    total_fit = sum(fit_value)
    for i in range(len(fit_value)):
        newfit_value.append(fit_value[i] / total_fit)
    # 计算累计概率
    cumsum(newfit_value)
    ms = []
    pop_len = len(pop)
    for i in range(pop_len):
        ms.append(random.random())
    ms.sort()
    fitin = 0
    newin = 0
    newpop = pop
    # 转轮盘选择法
    while newin < pop_len:
        if(ms[newin] < newfit_value[fitin]):
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    pop = newpop


def decodechrom(pop, chrom_length):
    temp = []
    for i in range(len(pop)):
        t = 0
        for j in range(chrom_length):
            t += pop[i][j] * (math.pow(2, j))
        temp.append(t)
    return temp


class PSO(object):
    def __init__(self, population_size, 
                       max_steps, 
                       calculate_fitness,
                       control_d, w, c,
                       x_bound,
                       step,
                       rainid,rainname,ta_original):
        self.w = w  # 惯性权重
        self.c1 = self.c2 = c  # 加速常数
        self.population_size = population_size  # 粒子群数量
        self.dim = control_d  # 搜索空间的维度
        self.max_steps = max_steps  # 迭代次数
        #self.x_bound = [-1, 1]  # 解空间范围
        # 使用上一时刻数据限定解空间范围
        self.x_bound = x_bound
        self.x = np.random.uniform(self.x_bound[0], self.x_bound[1],
                                   (self.population_size, self.dim))  # 初始化粒子群位置
        self.v = np.random.rand(self.population_size, self.dim)  # 初始化粒子群速度

        self.calculate_fitness=calculate_fitness
        self.step = step
        self.rainid, self.rainname = rainid, rainname
        self.ta_original = ta_original
        
        fitness = self.calculate_fitness(self.x,
                                         self.step,
                                         self.rainid,
                                         self.rainname,
                                         self.ta_original)
        self.p = self.x  # 个体的最佳位置
        self.pg = self.x[np.argmax(fitness)]  # 全局最佳位置
        self.individual_best_fitness = fitness  # 个体的最优适应度
        self.global_best_fitness = np.max(fitness)  # 全局最佳适应度

    def evolve(self):
        self.best_fit_curv=[]
        for step in range(self.max_steps):
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            # 更新速度和权重
            self.v = self.w * self.v + self.c1 * r1 * (self.p - self.x) + self.c2 * r2 * (self.pg - self.x)
            self.x = 0.01*self.v + self.x
            #fitness = self.calculate_fitness(self.x)
            fitness = self.calculate_fitness(self.x,
                                             self.step,
                                             self.rainid,
                                             self.rainname,
                                             self.ta_original)
            # 需要更新的个体
            update_id = np.greater(fitness,self.individual_best_fitness)
            self.p[update_id] = self.x[update_id]
            self.individual_best_fitness[update_id] = fitness[update_id]
            # 新一代出现了更大的fitness，所以更新全局最优fitness和位置
            if np.max(fitness)> self.global_best_fitness:
                self.pg = self.x[np.argmax(fitness)]    
                self.global_best_fitness = np.max(fitness)
            self.best_fit_curv.append(self.global_best_fitness)
            
            #渐进终止条件
            if step>21:
                if np.sum(self.best_fit_curv[-10:])-np.sum(self.best_fit_curv[-20:-10])<0.01:
                    break


if __name__=='__main__':
    pso = PSO(100, 100)
    pso.evolve()
    plt.show()