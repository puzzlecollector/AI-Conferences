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
import umap 

with open("all_ts2vec_embeddings.pkl", "rb") as f: 
    ts2vec_embeddings_dict = pickle.load(f) # length is 6 timesteps shorter due to lookback 
    
with open("news_emb_chart_dates.pkl", "rb") as f: 
    news_embeddings_dict = pickle.load(f) 
    
news_date_keys = [] 
news_embeddings = [] 
for key, value in tqdm(news_embeddings_dict.items()): 
    vv = value.detach().cpu() 
    vv = torch.reshape(vv, (-1, 768))  
    news_date_keys.append(key) 
    news_embeddings.append(vv) 
    
news_embeddings = torch.cat(news_embeddings, dim=0) 
news_embeddings = np.array(news_embeddings) 

print("dimensionality reduction of news embeddings") 
reducer = umap.UMAP(n_components=128)
news_embeddings = reducer.fit_transform(news_embeddings)
print("done reducing dimension!") 

reduced_news_embeddings = {} 
for i in range(len(news_embeddings)): 
    reduced_news_embeddings[news_date_keys[i]] = news_embeddings[i]

multimodal_embeddings = {} 

for key, value in tqdm(ts2vec_embeddings_dict.items(), position=0, leave=True): 
    multimodal_embeddings[key] = ts2vec_embeddings_dict[key] + reduced_news_embeddings[key] 
    
with open("multimodal_dict.pkl", "wb") as f: 
    pickle.dump(multimodal_embeddings, f) 
    
print("done saving multimodal embeddings!") 