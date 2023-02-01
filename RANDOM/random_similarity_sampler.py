import numpy as np 
import pandas as pd 
import json 
import ccxt 
import seaborn as sns
import os 
import pandas_ta as ta 
import time
from datetime import datetime, timedelta
import math
from tqdm.auto import tqdm 
import matplotlib.pyplot as plt 
from transformers import * 
import torch 
from torch import Tensor 
from torch.utils.data import * 
import torch.nn as nn 
import torch.nn.functional as F 
from sklearn.utils.class_weight import compute_class_weight 
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler
from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import CosineSimilarity
from scipy.spatial.distance import cdist 
import pickle
from torchmetrics import MeanAbsolutePercentageError 
import random

with open("BTC_USDT-1h-12.json") as f: 
        d = json.load(f) 

chart_df = pd.DataFrame(d) 
chart_df = chart_df.rename(columns={0:"timestamp", 1:"open", 2:"high", 3:"lowdatetime(2017, 10, 11, 19, 0),", 4:"close", 5:"volume"})

def process(df): 
        binance = ccxt.binance() 
        dates = df["timestamp"].values 
        timestamp = [] 
        for i in range(len(dates)):
                date_string = binance.iso8601(int(dates[i])) 
                date_string = date_string[:10] + " " + date_string[11:-5] 
                timestamp.append(date_string) 
        df["datetime"] = timestamp
        df = df.drop(columns={"timestamp"}) 
        return df 

chart_df = process(chart_df) 

hours, days, months, years = [],[],[],[] 
for dt in tqdm(chart_df["datetime"]):
        dtobj = pd.to_datetime(dt) 
        hour = dtobj.hour 
        day = dtobj.day 
        month = dtobj.month 
        year = dtobj.year 
        hours.append(hour) 
        days.append(day) 
        months.append(month) 
        years.append(year) 

chart_df["hours"] = hours 
chart_df["days"] = days  
chart_df["months"] = months 
chart_df["years"] = years 


start_index = 1000 

datetimes = chart_df["datetime"].values 

seq_len = 24
forecast_horizon = 6
date_chart = {} # datetime object : close prices  

for i in tqdm(range(len(datetimes) - seq_len - forecast_horizon), position=0, leave=True): 
    dt_obj = datetime.strptime(str(datetimes[i]), "%Y-%m-%d %H:%M:%S")
    date_chart[dt_obj] = (chart_df["close"].values[i:i+seq_len], chart_df["close"].values[i+seq_len:i+seq_len+forecast_horizon])
    

random_aug = {} 

for i in tqdm(range(1000, len(datetimes) - seq_len), desc="initializing..."): 
    dtobj = datetime.strptime(str(datetimes[i]), "%Y-%m-%d %H:%M:%S")
    random_aug[dtobj] = [] 

for index in tqdm(range(1000, len(datetimes) - seq_len - forecast_horizon), position=0, leave=True, desc="sampling random data"): 
    query_date = datetimes[index] 
    query_dt_obj = datetime.strptime(str(query_date), "%Y-%m-%d %H:%M:%S")  
    
    # candidate_close_prices, candidate_dates, candidates_future  = [], [], []   
    candidate_dates = [] 
    for key, _ in date_chart.items(): 
        if key + timedelta(hours=30) <= query_dt_obj: # 24 hours + next 6 hours 
            candidate_dates.append(key)      
    sampled_index = random.sample(range(0, len(candidate_dates)), 3) 
    
    for i in range(len(sampled_index)): 
        random_aug[query_dt_obj].append(candidate_dates[sampled_index[i]])  
        
        
print("saving random augmentation dictionary") 

with open("random_aug.pkl", "wb") as handle: 
    pickle.dump(random_aug, handle) 