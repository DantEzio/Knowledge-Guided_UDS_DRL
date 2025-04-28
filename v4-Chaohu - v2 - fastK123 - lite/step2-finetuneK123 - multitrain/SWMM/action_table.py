# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 15:17:12 2022

@author: chong
"""

import numpy as np
import pandas as pd

table=[]
for i1 in [0,1]:
    for i2 in [0,1]:
        for i3 in [0,1]:
            for i4 in [0,1]:
                for i5 in [0,1]:
                    for i6 in [0,1]:
                        for i7 in [0,1]:
                            tem = [i1,i2,i3,i4,i5,i6,i7]
                            table.append(tem)
                            
table = np.array(table)
pd.DataFrame(table).to_csv('DQN_action_table.csv')