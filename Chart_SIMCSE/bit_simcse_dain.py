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

chart_df["hours"] = hours 
chart_df["days"] = days  
chart_df["months"] = months 
chart_df["years"] = years 

# predict the next 24 hours given thet past 168 hours 

all_seqs = [] 
all_datetimes = [] 

datetimes = chart_df["datetime"].values 

seq_len = 168 

for i in tqdm(range(0, chart_df.shape[0] - seq_len), position=0, leave=True): 
    all_seqs.append(chart_df.iloc[i:i+seq_len, [3]].values) 
    all_datetimes.append(datetimes[i+seq_len-1]) 

all_seqs = torch.tensor(all_seqs).float() 

### define model ### 
class DAIN_Layer(nn.Module):
    def __init__(self, mode, mean_lr, gate_lr, scale_lr, input_dim):
        super(DAIN_Layer, self).__init__()
        print("Mode = ", mode)
        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr
        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))
        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))
        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(input_dim, input_dim)
        self.eps = 1e-8
    def forward(self, x):
        # Expecting  (n_samples, dim,  n_feature_vectors)
        # Nothing to normalize
        if self.mode == None:
            pass
        # Do simple average normalization
        elif self.mode == 'avg':
            avg = torch.mean(x, 2)
            avg = avg.resize(avg.size(0), avg.size(1), 1)
            x = x - avg
        # Perform only the first step (adaptive averaging)
        elif self.mode == 'adaptive_avg':
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg
        # Perform the first + second step (adaptive averaging + adaptive scaling )
        elif self.mode == 'adaptive_scale':
            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg
            # Step 2:
            std = torch.mean(x ** 2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1
            adaptive_std = adaptive_std.resize(adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / (adaptive_std)
        elif self.mode == 'full':
            # Step 1:
            avg = torch.mean(x, 2)
            adaptive_avg = self.mean_layer(avg)
            adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
            x = x - adaptive_avg
            # # Step 2:
            std = torch.mean(x ** 2, 2)
            std = torch.sqrt(std + self.eps)
            adaptive_std = self.scaling_layer(std)
            adaptive_std[adaptive_std <= self.eps] = 1
            adaptive_std = adaptive_std.resize(adaptive_std.size(0), adaptive_std.size(1), 1)
            x = x / adaptive_std
            # Step 3: 
            avg = torch.mean(x, 2)
            gate = F.sigmoid(self.gating_layer(avg))
            gate = gate.resize(gate.size(0), gate.size(1), 1)
            x = x * gate
        else:
            assert False
        return x

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

# returns chart embedding 
class BTCSimCSE(nn.Module): 
        def __init__(self, chart_features, sequence_length, d_model, n_heads, num_encoders):
                super(BTCSimCSE, self).__init__() 
                self.chart_features = chart_features
                self.sequence_length = sequence_length 
                self.d_model = d_model 
                self.n_heads = n_heads 
                self.num_encoders = num_encoders 
                self.dain = DAIN_Layer(mode="full", mean_lr=1e-06, gate_lr=10, scale_lr=0.001, input_dim=self.sequence_length)  
                self.chart_embedder = nn.Sequential(
                        nn.Linear(self.chart_features, d_model//2),
                        nn.ReLU(), 
                        nn.Linear(d_model//2, d_model)
                )
                self.pos_encoder = PositionalEncoding(d_model=self.d_model) 
                self.encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=self.num_encoders) 
        def forward(self, x): 
                x = self.dain(x) 
                x = self.chart_embedder(x) 
                x = self.pos_encoder(x) 
                x = self.transformer_encoder(x) 
                x = torch.mean(x, dim=1) 
                return x 

seq_len = 168 
def custom_collate(batch):
    sequences, labels = [], []  
    ids = 0 
    for seq in batch: 
        sequences.append(torch.reshape(seq, (-1, seq_len, 1))) 
        sequences.append(torch.reshape(seq, (-1, seq_len, 1))) 
        labels.append(ids) 
        labels.append(ids) 
        ids += 1 
    sequences = torch.cat(sequences, dim=0)  
    labels = torch.tensor(labels, dtype=int) 
    return sequences, labels 

train_size = int(all_seqs.shape[0] * 0.9) 

train_seqs = all_seqs[:train_size] 
val_seqs = all_seqs[train_size:] 

batch_size = 128
train_data = TensorDataset(train_seqs) 
train_sampler = RandomSampler(train_seqs) 
train_dataloader = DataLoader(train_seqs, sampler=train_sampler, collate_fn = custom_collate, batch_size=batch_size)  

val_data = TensorDataset(val_seqs) 
val_sampler = SequentialSampler(val_seqs) 
val_dataloader = DataLoader(val_seqs, sampler=val_sampler, collate_fn = custom_collate, batch_size=batch_size) 

miner = miners.MultiSimilarityMiner(distance = CosineSimilarity()) # default is cosine distance anyways 
loss_func = losses.MultiSimilarityLoss(distance = CosineSimilarity())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

model = BTCSimCSE(chart_features=1, sequence_length=168, d_model=128, n_heads=8, num_encoders=6)
model.to(device) 
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8) 
epochs = 30
total_steps = len(train_dataloader) * epochs 
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps) 

best_val_loss = 9999999999

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
        
os.rename("BTC_SIMCSE.pt", f"BTC_SIMCSE_val_loss_{best_val_loss}.pt") 
