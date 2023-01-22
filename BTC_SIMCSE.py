import numpy as np 
import pandas as pd 
import json 
import ccxt 
from tqdm.auto import tqdm 
import pandas_ta as ta 
import seaborn as sns 
from sklearn.utils.class_weight import compute_class_weight 
from sklearn.metrics import f1_score 
import random 
import torch 
from torch import Tensor 
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler 
from transformers import * 
import matplotlib.pyplot as plt 
import time 
import math
import os 
from pytorch_metric_learning import miners, losses # pip install pytoch-metric-learning 
from pytorch_metric_learning.distances import CosineSimilarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

### define model ### 
class PositionalEncoding(nn.Module):
        def __init__(self, d_model, dropout=0.1, max_len=5000):
                super(PositionalEncoding, self).__init__() 
                self.dropout = nn.Dropout(p=dropout) 
                pe = torch.zeros(max_len, d_model) 
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) 
                pe[:, 0::2] = torch.sin(position * div_term) 
                pe[:, 1::2] = torch.cos(position * div_term) 
                pe = pe.unsqueeze(0).transpose(0, 1) 
                self.register_buffer("pe", pe) 
        def forward(self, x):
                x = x + self.pe[:x.size(0), :] 
                return self.dropout(x) 

class AttentivePooling(nn.Module):
        def __init__(self, input_dim):
                super(AttentivePooling, self).__init__() 
                self.W = nn.Linear(input_dim, 1) 
        def forward(self, x):
                softmax = F.softmax
                att_w = softmax(self.W(x).squeeze(-1)).unsqueeze(-1) 
                x = torch.sum(x * att_w, dim=1) 
                return x 

# returns chart embedding 
class BTCSimCSE(nn.Module): 
        def __init__(self, chart_features, sequence_length, d_model, n_heads, num_encoders):
                super(BTCSimCSE, self).__init__() 
                self.chart_features = chart_features
                self.sequence_length = sequence_length 
                self.d_model = d_model 
                self.n_heads = n_heads 
                self.num_encoders = num_encoders 
                self.batchnorm = nn.BatchNorm1d(sequence_length) 
                self.chart_embedder = nn.Sequential(
                        nn.Linear(self.chart_features, d_model//2),
                        nn.ReLU(), 
                        nn.Linear(d_model//2, d_model)
                )
                self.pos_encoder = PositionalEncoding(d_model=self.d_model) 
                self.encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=self.num_encoders) 
                self.attentive_pooling = AttentivePooling(input_dim=self.d_model) 
        def forward(self, x): 
                x = self.batchnorm(x) 
                x = self.chart_embedder(x) 
                x = self.pos_encoder(x) 
                x = self.transformer_encoder(x) 
                x = self.attentive_pooling(x) 
                return x 
            

### get data and preprocess ### 
with open("BTC_USDT-1h-12.json") as f: 
        d = json.load(f) 

chart_df = pd.DataFrame(d) 
chart_df = chart_df.rename(columns={0:"timestamp", 1:"open", 2:"high", 3:"low", 4:"close", 5:"volume"})

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

chart_df["hours"] = hour 
chart_df["days"] = day 
chart_df["months"] = months 
chart_df["years"] = years 

seq_len = 168 # 168 hours = one week worth of time 

X_seq = [] 

for i in range(chart_df.shape[0] - seq_len):
        X_seq.append(chart_df.iloc[i:i+seq_len, [0,1,2,3,4]].values) 

X_seq = np.array(X_seq) 

train_size = int(0.9 * len(X_seq)) 

X_train = X_seq[:train_size]

X_val = X_seq[train_size:] 

X_train = torch.tensor(X_train).float() 
X_val = torch.tensor(X_val).float() 

def custom_collate(batch):
    sequences, labels = [], []  
    ids = 0 
    for seq in batch: 
        sequences.append(torch.reshape(seq[0], (-1, seq_len, 5))) 
        sequences.append(torch.reshape(seq[0], (-1, seq_len, 5))) 
        labels.append(ids) 
        labels.append(ids) 
        ids += 1 
    sequences = torch.cat(sequences, dim=0)  
    labels = torch.tensor(labels, dtype=int) 
    return sequences, labels 

batch_size = 128
train_data = TensorDataset(X_train) 
train_sampler = RandomSampler(train_data) 
train_dataloader = DataLoader(train_data, sampler=train_sampler, collate_fn = custom_collate, batch_size=batch_size)  

val_data = TensorDataset(X_val) 
val_sampler = SequentialSampler(val_data) 
val_dataloader = DataLoader(val_data, sampler=val_sampler, collate_fn = custom_collate, batch_size=batch_size)  

miner = miners.MultiSimilarityMiner(distance = CosineSimilarity()) # default is cosine distance anyways 
loss_func = losses.MultiSimilarityLoss(distance = CosineSimilarity())

model = BTCSimCSE(chart_features=5, sequence_length=168, d_model=128, n_heads=8, num_encoders=6)
model.to(device) 
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8) 
epochs = 100  
total_steps = len(train_dataloader) * epochs 
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps) 

best_val_loss = 123456789

model.zero_grad() 
for epoch_i in tqdm(range(epochs), desc="Epochs", position=0, leave=True, total=epochs):
    train_loss = 0
    model.train() 
    with tqdm(train_dataloader, unit="batch") as tepoch: 
        for step, batch in enumerate(tepoch): 
            batch = (t.to(device) for t in batch) 
            b_seqs, b_labels = batch 
            embeddings = model(b_seqs)
            hard_pairs = miner(embeddings, b_labels)
            loss = loss_func(embeddings, b_labels, hard_pairs) 
            train_loss += loss.item() 
            loss.backward()
            optimizer.step() 
            scheduler.step() 
            model.zero_grad() 
            tepoch.set_postfix(loss=train_loss / (step+1)) 
            time.sleep(0.1) 
    avg_train_loss = train_loss / len(train_dataloader) 
    print(f"average train loss : {avg_train_loss}") 
    val_loss = 0 
    model.eval() 
    for step, batch in tqdm(enumerate(val_dataloader), position=0, leave=True, total=len(val_dataloader)):
        batch = (t.to(device) for t in batch) 
        b_seqs, b_labels = batch 
        embeddings = model(b_seqs) 
        loss = loss_func(embeddings, b_labels) 
        val_loss += loss.item() 
    avg_val_loss = val_loss / len(val_dataloader) 
    print(f"average validation loss : {avg_val_loss}") 
    if avg_val_loss < best_val_loss: 
        best_val_loss = avg_val_loss 
        torch.save(model.state_dict(), "BTC_SIMCSE.pt") 

print("done training!") 
print(f"best val loss : {best_val_loss}") 
os.rename("BTC_SIMCSE.pt", f"BTC_SIMCSE_val_loss_{best_val_loss}.pt")  
