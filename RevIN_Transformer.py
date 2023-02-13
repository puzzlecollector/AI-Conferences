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
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay

with open("BTC_USDT-4h-12.json") as f: 
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


def preprocess_seq_data(chart_df, threshold=0.0075): 
    targets = [] 
    openv = chart_df["open"].values 
    close = chart_df["close"].values 
    high = chart_df["high"].values 
    low = chart_df["low"].values  
    volume = chart_df["volume"].values 
    
    for i in range(close.shape[0]-1):
        high_vol = (high[i+1] - close[i]) / close[i] 
        low_vol = (low[i+1] - close[i]) / close[i] 
        if high_vol >= threshold: 
            targets.append(0) 
        elif low_vol <= -threshold:
            targets.append(1) 
        else:
            targets.append(2) 
        
    targets.append(None) 
    chart_df["Targets"] = targets 
    
    chart_df.set_index(pd.DatetimeIndex(chart_df["datetime"]), inplace=True)
    chart_df["bop"] = chart_df.ta.bop(lookahead=False) 
    chart_df["ebsw"] = chart_df.ta.ebsw(lookahead=False) 
    chart_df["cmf"] = chart_df.ta.cmf(lookahead=False) 
    chart_df["rsi/100"] = chart_df.ta.rsi(lookahead=False) / 100
    chart_df["high/low"] = chart_df["high"] / chart_df["low"] 
    chart_df["high/open"] = chart_df["high"] / chart_df["open"] 
    chart_df["low/open"] = chart_df["low"] / chart_df["open"] 
    chart_df["close/open"] = chart_df["close"] / chart_df["open"] 
    chart_df["high/close"] = chart_df["high"] / chart_df["close"] 
    chart_df["low/close"] = chart_df["low"] / chart_df["close"]     
    for l in tqdm(range(1, 12), position=0, leave=True): 
        for col in ["open", "high", "low", "close", "volume"]:
            val = chart_df[col].values 
            val_ret = [None for _ in range(l)]
            for i in range(l, len(val)):
                if val[i-l] == 0: 
                    ret = 1 
                else:
                    ret = val[i] / val[i-l]  
                val_ret.append(ret) 
            chart_df["{}_change_{}".format(col, l)] = val_ret 

    chart_df.dropna(inplace=True) 
    chart_df.drop(columns={"open", "high", "low", "close", "volume"}, inplace=True) 
    return chart_df

chart_df = preprocess_seq_data(chart_df) 

train_columns = []
for col in chart_df.columns:
    if col not in ["Targets", "datetime", "years"]:
        train_columns.append(col)  

X = chart_df[train_columns] 
Y = chart_df["Targets"] 

train_size = int(chart_df.shape[0] * 0.8) 
val_size = int(chart_df.shape[0] * 0.1) 

X_train = X.iloc[:train_size] 
Y_train = Y.iloc[:train_size] 

X_val = X.iloc[train_size:train_size+val_size] 
Y_val = Y.iloc[train_size:train_size+val_size] 

X_test = X.iloc[train_size+val_size:] 
Y_test = Y.iloc[train_size+val_size:] 

X_train_seq = [] 
Y_train_seq = [] 

X_val_seq = [] 
Y_val_seq = [] 

X_test_seq = [] 
Y_test_seq = [] 

lookback = 42

for i in tqdm(range(lookback, X_train.shape[0]), position=0, leave=True):
    X_train_seq.append(X_train.iloc[i-lookback:i, 3:].values)  
    Y_train_seq.append(Y_train[i-1]) 
    
for i in tqdm(range(lookback, X_val.shape[0]), position=0, leave=True): 
    X_val_seq.append(X_val.iloc[i-lookback:i, 3:].values) 
    Y_val_seq.append(Y_val[i-1]) 

for i in tqdm(range(lookback, X_test.shape[0]), position=0, leave=True): 
    X_test_seq.append(X_test.iloc[i-lookback:i, 3:].values)  
    Y_test_seq.append(Y_test[i-1])   
    
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(Y_train), y=np.array(Y_train)) 

X_train_seq = torch.tensor(X_train_seq).float() 
Y_train_seq = torch.tensor(Y_train_seq, dtype=int) 

X_val_seq = torch.tensor(X_val_seq).float() 
Y_val_seq = torch.tensor(Y_val_seq, dtype=int) 

X_test_seq = torch.tensor(X_test_seq).float() 
Y_test_seq = torch.tensor(Y_test_seq, dtype=int) 

batch_size = 64 

train_data = TensorDataset(X_train_seq, Y_train_seq) 
train_sampler = RandomSampler(train_data) 
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size) 

val_data = TensorDataset(X_val_seq, Y_val_seq) 
val_sampler = SequentialSampler(val_data) 
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size) 

test_data = TensorDataset(X_test_seq, Y_test_seq) 
test_sampler = SequentialSampler(test_data) 
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size) 

class PositionalEncoding(nn.Module): 
    def __init__(self, d_model, dropout=0.1, max_len=5000): 
        super(PositionalEncoding, self).__init__() 
        self.dropout = nn.Dropout(p=dropout) 
        pe = torch.zeros(max_len, d_model) 
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0).transpose(0,1) 
        self.register_buffer("pe", pe) 
    def forward(self, x): 
        x = x + self.pe[:x.size(0), :] 
        return self.dropout(x) 

class MultiSampleDropout(nn.Module): 
    def __init__(self, max_dropout_rate, num_samples, classifier): 
        super(MultiSampleDropout, self).__init__() 
        self.dropout = nn.Dropout 
        self.classifier = classifier 
        self.max_dropout_rate = max_dropout_rate 
        self.num_samples = num_samples 
    def forward(self, out): 
        return torch.mean(torch.stack([self.classifier(self.dropout(p=rate)(out)) for _, rate in enumerate(np.linspace(0, self.max_dropout_rate, self.num_samples))], dim=0), dim=0)

class AttentivePooling(torch.nn.Module): 
    def __init__(self, input_dim): 
        super(AttentivePooling, self).__init__() 
        self.W = nn.Linear(input_dim, 1) 
    def forward(self, x): 
        softmax = F.softmax 
        att_w = softmax(self.W(x).squeeze(-1)).unsqueeze(-1) 
        x = torch.sum(x * att_w, dim=1) 
        return x 
    
def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res

def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
    def forward(self, x):
        return self.net(x)

class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        mask &= nan_mask
        x[~mask] = 0
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        return x
    
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class Encoder(nn.Module): 
    def __init__(self, 
                 chart_features, 
                 sequence_length, 
                 d_model, 
                 num_classes, 
                 n_heads, 
                 num_encoders,
                 dropout_r,
                 dropout_samples): 
        super(Encoder, self).__init__() 
        self.chart_features = chart_features 
        self.sequence_length = sequence_length 
        self.d_model = d_model 
        self.num_classes = num_classes  
        self.n_heads = n_heads 
        self.num_encoders = num_encoders  
        self.dropout_r = dropout_r 
        self.dropout_samples = dropout_samples 
        self.batchnorm = nn.BatchNorm1d(self.sequence_length) 
        self.revin_layer = RevIN(self.chart_features)
        self.ts_encoder = TSEncoder(input_dims=self.chart_features, output_dims=self.d_model)
        self.pos_encoder = PositionalEncoding(d_model=self.d_model) 
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads, batch_first=True) 
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=self.num_encoders) 
        self.attentive_pooling = AttentivePooling(input_dim=self.d_model)   
        self.fc = nn.Linear(self.d_model, self.num_classes) 
        self.multi_dropout = MultiSampleDropout(self.dropout_r, self.dropout_samples, self.fc) 
    def forward(self, x): 
        x = self.revin_layer(x, "norm") 
        x = self.ts_encoder(x) 
        x = self.pos_encoder(x) 
        x = self.transformer_encoder(x)  
        x = self.attentive_pooling(x) 
        x = self.multi_dropout(x) 
        return x 

    
def flat_accuracy(preds, labels): 
    pred_flat = np.argmax(preds, axis=1).flatten() 
    labels_flat = labels.flatten() 
    return np.sum(pred_flat == labels_flat) / len(labels_flat) 

device = torch.device("cuda:0") 

best_val_accuracy = 0 

train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], [] 

model = Encoder(chart_features = X_train_seq.shape[2],
                sequence_length = X_train_seq.shape[1], 
                d_model = 128,
                num_classes = 3,
                n_heads = 8,
                num_encoders = 6, 
                dropout_r = 0.2, 
                dropout_samples = 8) 

model.to(device) 

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8) 
epochs = 50
total_steps = len(train_dataloader) * epochs 
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*total_steps), num_training_steps=total_steps) 
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)   
loss_func = nn.CrossEntropyLoss(weight=class_weights) 
model.zero_grad() 
for epoch_i in tqdm(range(epochs), desc="Epochs", position=0, leave=True, total=epochs): 
    train_loss, train_accuracy = 0, 0 
    model.train() 
    with tqdm(train_dataloader, unit="batch") as tepoch: 
        for step, batch in enumerate(tepoch): 
            batch = tuple(t.to(device) for t in batch) 
            b_seqs, b_labels = batch 
            outputs = model(b_seqs) 
            loss = loss_func(outputs, b_labels) 
            train_loss += loss.item() 
            logits_cpu, labels_cpu = outputs.detach().cpu().numpy(), b_labels.detach().cpu().numpy() 
            train_accuracy += flat_accuracy(logits_cpu, labels_cpu)  
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step() 
            scheduler.step() 
            model.zero_grad() 
            tepoch.set_postfix(loss=train_loss / (step+1), accuracy=train_accuracy / (step+1) * 100.0) 
            time.sleep(0.1) 
        avg_train_loss = train_loss / len(train_dataloader) 
        avg_train_accuracy = train_accuracy / len(train_dataloader) 
        train_losses.append(avg_train_loss) 
        train_accuracies.append(avg_train_accuracy) 
        print(f"average train loss : {avg_train_loss}") 
        print(f"average train accuracy : {avg_train_accuracy}") 
    
    val_loss, val_accuracy = 0, 0 
    model.eval() 
    for step, batch in tqdm(enumerate(val_dataloader), position=0, leave=True, total=len(val_dataloader)): 
        batch = tuple(t.to(device) for t in batch) 
        b_seqs, b_labels = batch 
        with torch.no_grad(): 
            outputs = model(b_seqs) 
        loss = loss_func(outputs, b_labels) 
        val_loss += loss.item() 
        logits_cpu, labels_cpu = outputs.detach().cpu().numpy(), b_labels.detach().cpu().numpy() 
        val_accuracy += flat_accuracy(logits_cpu, labels_cpu) 
    avg_val_loss = val_loss / len(val_dataloader) 
    avg_val_accuracy = val_accuracy / len(val_dataloader) 
    val_losses.append(avg_val_loss) 
    val_accuracies.append(avg_val_accuracy) 
    print(f"average val loss : {avg_val_loss}") 
    print(f"average val accuracy : {avg_val_accuracy}") 
    if avg_val_accuracy > best_val_accuracy: 
        best_val_accuracy = avg_val_accuracy
        torch.save(model.state_dict(), f"NN_Encoder.pt")  
        
os.rename("NN_Encoder.pt", f"NN_Encoder_{best_val_accuracy}.pt") 
