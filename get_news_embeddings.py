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
import random 
import warnings 
warnings.filterwarnings("ignore") 
import pickle
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier  
import dtw 
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from datetime import datetime, timedelta 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
tokenizer = AlbertTokenizer.from_pretrained("totoro4007/cryptodeberta-base-all-finetuned")
model = AutoModel.from_pretrained("totoro4007/cryptodeberta-base-all-finetuned") 
model.to(device) 
model.eval() 

news = pd.read_csv("full_news_22_01_16-Copy1.csv")
titles = news["titles"].values 
contents = news["contents"].values 
years = news["year"].values 
months = news["month"].values 
days = news["day"].values 
hours = news["hour"].values 

news_dt_objects = [] 
for i in tqdm(range(len(years)), position=0, leave=True): 
    date_str = str(years[i]) + "-" + str(months[i]) + "-" + str(days[i]) + " " + str(hours[i]) + ":00:00"  
    dtobj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S") 
    news_dt_objects.append(dtobj)

news_emb = {}  
    
for i in tqdm(range(len(news_dt_objects)), position=0, leave=True): 
    encoded_input = tokenizer(str(titles[i]), str(contents[i]), max_length=512, truncation=True, padding="max_length", return_tensors="pt").to(device) 
    with torch.no_grad():
        outputs = model(**encoded_input) 
        outputs = outputs[0][:,0,:] 
        outputs = outputs.detach().cpu().numpy()
        news_emb[news_dt_objects[i]] = outputs 
        
        
with open("news_emb.pkl", "wb") as f: 
    pickle.dump(news_emb, f) 
    
print(news_emb) 
print("done!")