# -*- coding: utf-8 -*-
"""
Created on Sun May 12 10:10:29 2019

@author: user
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib qt
import glob

pathname = "D:\\CM_BDA\\dataset"
filepath = pathname + "\\*\\*.csv"

dataset = pd.DataFrame()

for file in glob.glob(filepath):
    data = pd.read_csv(file)
    data = data.drop(columns = ['Unnamed: 0']) 
    dataset = pd.concat([dataset, data])

#dataset = dataset.drop(columns = ['Unnamed: 0'])    
dataset = dataset.drop_duplicates()
dataset = dataset.sort_values(by = 'timestamp')

dataset.to_csv('MSFT_INTRA_DAY.csv')