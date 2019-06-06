# -*- coding: utf-8 -*-
"""
Created on Sun May  5 21:04:53 2019

@author: user
"""

import glob
import numpy as np
import pandas as pd
import requests
import datetime, os, sys
from time import sleep
from threading import Thread

api_key = 'IHZZRCFX1UTLCOO4'

now = datetime.datetime.now()
dirnow = now.strftime("%Y-%m-%d")
path = '.\\dataset\\{}'.format(str(dirnow))
os.mkdir(path)

'''
data=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&datatype=csv&symbol=MSFT&apikey={}'.format(api_key))
data=data.json()

now = datetime.datetime.now()
dirnow = now.strftime("%Y-%m-%d")
path = '.\\dataset\\{}'.format(str(dirnow))
os.mkdir(path)

filenow = now.strftime("%Y-%m-%d %H:%M:%S")

data=data['Time Series (1min)']
df=pd.DataFrame(columns=['date','open','high','low','close','volume'])
for d,p in data.items():
    date=datetime.datetime.strptime(d,'%Y-%m-%d %H:%M:%S')
    data_row=[date,float(p['1. open']),float(p['2. high']),float(p['3. low']),float(p['4. close']),int(p['5. volume'])]
    df.loc[-1,:]=data_row
    df.index=df.index+1
X=df.sort_values('date')
X['close']=X['close'].astype(float)

file = 'msft_{}'.format(filenow)
outpath = '.\\dataset\\2019-05-03\\{}'.format(file)
X.to_csv(outpath)
'''

def data_task():
    #while True:
    #api_key = 'IHZZRCFX1UTLCOO4'
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&datatype=csv&symbol=MSFT&apikey={}'.format(api_key)
    #data=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&datatype=csv&interval=1min&symbol=MSFT&apikey={}'.format(api_key))
    data=pd.read_csv(url)
    
    now = datetime.datetime.now()
    datenow = now.strftime("%Y-%m-%d")
    filenow = now.strftime("%Y%m%d_%H%M%S")
    '''
    data=data['Time Series (1min)']
    df=pd.DataFrame(columns=['date','open','high','low','close','volume'])
    for d,p in data.items():
        date=datetime.datetime.strptime(d,'%Y-%m-%d %H:%M:%S')
        data_row=[date,float(p['1. open']),float(p['2. high']),float(p['3. low']),float(p['4. close']),int(p['5. volume'])]
        df.loc[-1,:]=data_row
        df.index=df.index+1
    X=df.sort_values('date')
    X['close']=X['close'].astype(float)
    '''
    file = 'msft_{}.csv'.format(filenow)
    outpath = '.\\dataset\\{}'.format(datenow)
    data.to_csv(os.path.join(outpath, file))

data_task()

#Use when Intra_day
t = Thread(target = data_task)  
                             
t.daemon = True               
                              
t.start()

snooziness = 6*60*60 - 30*60
sleep(snooziness)