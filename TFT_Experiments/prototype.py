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

# for processing discrete features 
def one_hot(x, dims, gpu = True):
    # print(x)
    out = []
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    
    if(not gpu):
        dtype = torch.FloatTensor
    else:
        dtype = torch.cuda.FloatTensor
    
    # print("Converting to one hot vector")
    for i in range(0, x.shape[-1]): # get rid of tqdm for training 
        x_ = x[:,:,i].byte().cpu().long().unsqueeze(-1)
        o = torch.zeros([batch_size, seq_len, dims[i]]).long()
        o.scatter_(-1, x_,  1) 
        out.append(o.type(dtype))
    return out

### addition of DAIN layer 
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

def a_norm(Q, K): 
    m = torch.matmul(Q, K.transpose(2,1).float()) 
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float()) 
    return torch.softmax(m, -1) 

def attention(Q, K, V): 
    a = a_norm(Q, K) # (batch_size, dim_attn, seq_length) 
    return torch.matmul(a, V) # (batch_size, seq_length, seq_length) 

class AttentionBlock(torch.nn.Module): 
    def __init__(self, dim_val, dim_attn): 
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val) 
        self.key = Key(dim_val, dim_attn) 
        self.query = Query(dim_val, dim_attn) 
    
    def forward(self, x, kv = None):
        if kv is None:
            # Attention with x connected to Q,K and V (For encoder)
            return attention(self.query(x), self.key(x), self.value(x))
        # Attention with x as Q, external vector kv as K and V (For decoder)
        return attention(self.query(x), self.key(kv), self.value(kv))
    
class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn))
        
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias = False)
                      
    def forward(self, x, kv = None):
        a = []
        for h in self.heads:
            a.append(h(x, kv = kv))
            
        a = torch.stack(a, dim = -1) #combine heads
        a = a.flatten(start_dim = 2) #flatten all head outputs
        
        x = self.fc(a)
        return x
    
class Value(torch.nn.Module):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_val, bias = False).cuda()
    
    def forward(self, x):
        return self.fc1(x)

class Key(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = False).cuda()
    
    def forward(self, x):
        return self.fc1(x)

class Query(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = False).cuda()
    
    def forward(self, x):
        return self.fc1(x)

def QuantileLoss(net_out, Y, q):
    return (q * F.relu(net_out - Y)) + ((1 - q) * F.relu(Y - net_out))

class GLU(torch.nn.Module): 
    def __init__(self, dim_input): 
        super(GLU, self).__init__() 
        self.fc1 = nn.Linear(dim_input, dim_input) 
        self.fc2 = nn.Linear(dim_input, dim_input) 
    def forward(self, x): 
        return torch.sigmoid(self.fc1(x)) * self.fc2(x) 

class GRN(torch.nn.Module): 
    def __init__(self, dim_input, dim_out=None, n_hidden=10, dropout_r=0.1): 
        super(GRN, self).__init__() 
        if dim_out != None: 
            self.skip = nn.Linear(dim_input, dim_out) 
        else: 
            self.skip = None 
            dim_out = dim_input 
        self.fc1 = nn.Linear(dim_input, n_hidden) 
        self.fc2 = nn.Linear(n_hidden, dim_out) 
        self.dropout = nn.Dropout(dropout_r) 
        self.gate = GLU(dim_out) 
        self.norm = nn.LayerNorm(dim_out) 
    def forward(self, x): 
        a = F.elu(self.fc1(x)) 
        a = self.dropout(self.fc2(a)) 
        a = self.gate(a) 
        if self.skip != None: 
            return self.norm(self.skip(x) + a) 
        return self.norm(x+a) 

class VSN(torch.nn.Module): 
    def __init__(self, n_var_cont, n_var_disc, dim_model, dropout_r=0.1): 
        super(VSN, self).__init__()
        n_var_total = n_var_cont + len(n_var_disc)
        # linear transformation of inputs into dmodel vectors 
        self.linearise = []
        for i in range(n_var_cont): 
            self.linearise.append(nn.Linear(1, dim_model, bias=False).cuda()) 
        
        # self.fc = nn.Linear(1, dim_model, bias=False).cuda()     
        # entity embeddings for discrete inputs 
        self.entity_embed = [] 
        for i in n_var_disc: 
            self.entity_embed.append(nn.Linear(i, dim_model, bias=False).cuda())  
        
        self.input_grn = GRN(dim_model, dropout_r = dropout_r) 
        self.vs_grn = GRN(n_var_total * dim_model, dim_out=n_var_total, dropout_r = dropout_r)
    
    # input (batch_size, seq_len, n_variables, input_size)
    def forward(self, x_cont, x_disc): 
        # linearise continuous inputs 
        linearised = [] 
        for idx, fc in enumerate(self.linearise): 
            linearised.append(fc(x_cont[:,:,idx])) 
        # entity embeddings for discrete inputs 
        embedded = []
        
        for x, fc in zip(x_disc, self.entity_embed): 
            embedded.append(fc(x)) 
            
        if len(self.linearise) != 0 and len(self.entity_embed) != 0: 
            linearised = torch.stack(linearised, dim=-2) 
            embedded = torch.stack(embedded, dim=-2)    
            vectorised_vars = torch.cat((linearised, embedded), dim=-2) # (batch_size, seq_len, dim_model, n_vars_total)
        elif len(self.linearise) != 0 and len(self.entity_embed) == 0: 
            vectorised_vars = torch.stack(linearised, dim=-2) # (batch_size, seq_len, n_var_cont, dim_model)
        elif len(self.entity_embed) != 0 and len(self.linearise) == 0: 
            vectorised_vars = torch.stack(embedded, dim=-2)
        
        # flatten everything except accross batch for variable selection weights 
        vs_weights = self.vs_grn(vectorised_vars.flatten(start_dim=2)) # (batch_size, seq_len, n_variables)
        vs_weights = torch.softmax(vs_weights, dim=-1).unsqueeze(-1) # (batch_size, seq_len, n_variables, 1) 
        
        # input_grn applied to every input separately 
        input_weights = self.input_grn(vectorised_vars) # (batch_size, seq_len, n_variables, dim_model)
        
        x = torch.sum((vs_weights * input_weights), dim = 2) 
        return x, vs_weights # returns (batch_size, seq_len, dim_model)
    
    
class LSTMLayer(torch.nn.Module): 
    def __init__(self, dim_model, n_layers=1, dropout_r=0.1): 
        super(LSTMLayer, self).__init__() 
        self.n_layers = n_layers 
        self.dim_model = dim_model 
        self.lstm = nn.LSTM(dim_model, dim_model, num_layers=n_layers, batch_first=True) 
        self.hidden = None
        self.dropout = nn.Dropout(dropout_r) 
    def forward(self, x): 
        if self.hidden == None: 
            raise Exception("call reset() to initialize LSTM Layer") 
        x, self.hidden = self.lstm(x, self.hidden) 
        x = self.dropout(x) 
        return x, self.hidden 
    def reset(self, batch_size, gpu=True): 
        if not gpu: 
            dtype = torch.FloatTensor 
        else: 
            dtype = torch.cuda.FloatTensor 
        self.hidden = (torch.zeros([self.n_layers, batch_size, self.dim_model]).type(dtype),  
                       torch.zeros([self.n_layers, batch_size, self.dim_model]).type(dtype)) 
        
class TFN(torch.nn.Module): 
    def __init__(self, 
                 n_var_past_cont, 
                 n_var_future_cont, 
                 n_var_past_disc, 
                 n_var_future_disc, 
                 dim_model, 
                 n_quantiles = 1,
                 dropout_r = 0.1, 
                 n_lstm_layers = 1, 
                 n_attention_layers = 1, 
                 n_heads = 4, 
                 sequence_length = 24, 
                 num_features = 4, 
                 forecast_horizon = 6, 
                 future_num_features = 3): 
        super(TFN, self).__init__() 
        self.sequence_length = sequence_length 
        self.num_features = num_features  
        self.forecast_horizon = forecast_horizon 
        self.future_num_features = future_num_features 
        
        self.past_cont_dain = DAIN_Layer(mode="full", mean_lr=1e-06, gate_lr=10, scale_lr=0.001, input_dim=sequence_length) # past forecast length = 24 
        self.future_cont_dain = DAIN_Layer(mode="full", mean_lr=1e-06, gate_lr=10, scale_lr=0.001, input_dim=forecast_horizon) # future forecast length = 6
        
        self.vs_past = VSN(n_var_past_cont, n_var_past_disc, dim_model, dropout_r = dropout_r) 
        self.vs_future = VSN(n_var_future_cont, n_var_future_disc, dim_model, dropout_r = dropout_r) 
        
        self.enc = LSTMLayer(dim_model, dropout_r = dropout_r, n_layers = n_lstm_layers) 
        self.dec = LSTMLayer(dim_model, dropout_r = dropout_r, n_layers = n_lstm_layers) 
        
        self.gate1 = GLU(dim_model) 
        self.norm1 = nn.LayerNorm(dim_model) 
        
        self.static_enrich_grn = GRN(dim_model, dropout_r = dropout_r) 
        
        self.attention = [] 
        for i in range(n_attention_layers):
            self.attention.append([MultiHeadAttentionBlock(dim_model, dim_model, n_heads=n_heads).cuda(), 
                                   nn.LayerNorm(dim_model).cuda()]) 
        self.norm2 = nn.LayerNorm(dim_model) 
        self.positionwise_grn = GRN(dim_model, dropout_r = dropout_r) 
        self.norm3 = nn.LayerNorm(dim_model) 
        self.dropout = nn.Dropout(dropout_r)  
        self.fc_out = nn.Linear(dim_model, n_quantiles) 
        
    def forward(self, x_past_cont, x_past_disc, x_future_cont, x_future_disc): 
        x_past_cont = self.past_cont_dain(x_past_cont) 
        x_future_cont = self.future_cont_dain(x_future_cont) 
        x_past_cont = torch.reshape(x_past_cont, (-1, self.sequence_length, self.num_features, 1)) 
        x_future_cont = torch.reshape(x_future_cont, (-1, self.forecast_horizon, self.future_num_features, 1))  
        
        # Encoder 
        x_past, vs_weights = self.vs_past(x_past_cont, x_past_disc) 
        e, e_hidden = self.enc(x_past) 
        self.dec_hidden = e_hidden 
        e = self.dropout(e) 
        x_past = self.norm1(self.gate1(e) + x_past) 

        # Decoder
        x_future, _ = self.vs_future(x_future_cont, x_future_disc) 
        d, _ = self.dec(x_future) 
        d = self.dropout(d) 
        x_future = self.norm1(self.gate1(d) + x_future) 

        # static enrichment
        x = torch.cat((x_past, x_future), dim=1) # (batch_size, past_seq_len + future_seq_len, dim_model)
        attention_res = x_future
        x = self.static_enrich_grn(x) 

        # attention layer 
        a = self.attention[0][1](self.attention[0][0](x) + x) 
        for at in self.attention[1:]:
            a = at[1](at[0](a) + a) 
        x_future = self.norm2(a[:, x_past.shape[1]:] + x_future) 
        a = self.positionwise_grn(x_future) 
        x_future = self.norm3(a + x_future + attention_res) 
        net_out = self.fc_out(x_future)  
        return net_out, vs_weights 

    def reset(self, batch_size, gpu=True): 
        self.enc.reset(batch_size, gpu) 
        self.dec.reset(batch_size, gpu) 
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    
model = TFN(n_var_past_cont = 4, 
            n_var_future_cont = 3, 
            n_var_past_disc = [13, 32, 24], 
            n_var_future_disc = [13, 32, 24], 
            dim_model = 160) 

model.to(device) 

# load data 
past_cont_inputs = np.load("aug_top3_simcse_past_cont_inputs.npy")
past_cont_inputs = torch.tensor(past_cont_inputs).float() 

past_dates = np.load("aug_top3_simcse_past_dates.npy")
past_dates = torch.tensor(past_dates).float() 

future_cont_inputs = np.load("aug_top3_simcse_future_cont_inputs.npy") 
future_cont_inputs = torch.tensor(future_cont_inputs).float() 

future_dates = np.load("aug_top3_simcse_future_dates.npy") 
future_dates = torch.tensor(future_dates).float() 

targets = np.load("aug_top3_simcse_targets.npy") 
targets = torch.tensor(targets).float() 

batch_size = 128 

class CustomDataset(Dataset): 
    def __init__(self, past_cont, past_disc, future_cont, future_disc, target_seq): 
        self.past_cont = past_cont 
        self.past_disc = past_disc 
        self.future_cont = future_cont 
        self.future_disc = future_disc 
        self.target_seq = target_seq
    def __len__(self): 
        return len(self.past_cont) 
    def __getitem__(self, i): 
        return {
            "past_cont": torch.tensor(self.past_cont[i], dtype=torch.float32), 
            "past_disc": torch.tensor(self.past_disc[i], dtype=torch.float32), 
            "future_cont": torch.tensor(self.future_cont[i], dtype=torch.float32), 
            "future_disc": torch.tensor(self.future_disc[i], dtype=torch.float32), 
            "target_seq": torch.tensor(self.target_seq[i], dtype=torch.float32), 
        }


train_dataset = CustomDataset(past_cont_inputs, past_dates, future_cont_inputs, future_dates, targets) 
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 


epochs = 50
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5) 
train_losses, val_losses = [], []  
criterion = nn.L1Loss()  
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

model.zero_grad() 
for epcoh_i in tqdm(range(0, epochs), desc="Epochs", position=0, leave=True, total=epochs):
    train_loss = 0 
    model.train() 
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for step, batch in enumerate(tepoch): 
            past_cont = batch["past_cont"].to(device) 
            past_disc = batch["past_disc"].to(device) 
            future_cont = batch["future_cont"].to(device) 
            future_disc = batch["future_disc"].to(device) 
            target_seq = batch["target_seq"].to(device) 
            model.reset(batch_size=past_cont.shape[0], gpu=True)  
            past_disc = one_hot(past_disc, [13, 32, 24]) 
            future_disc = one_hot(future_disc, [13, 32, 24]) 
            net_out, vs_weights  = model(past_cont, past_disc, future_cont, future_disc) 
            net_out = torch.reshape(net_out, (-1, 6)) 
            loss = criterion(net_out, target_seq) 
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            tepoch.set_postfix(loss=train_loss / (step+1))
            time.sleep(0.1)
    avg_train_loss = train_loss / len(train_dataloader) 
    train_losses.append(avg_train_loss) 
    print(f"average train loss: {avg_train_loss}") 
    
    val_loss = 0 
    model.eval() 
    

