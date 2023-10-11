# https://charlieoneill11.github.io/charlieoneill/python/lstm/pytorch/2022/01/14/lstm2.html

# MULTIVARIANT CHARLIE VERSION

import os

from sched import scheduler

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators

print("All libraries loaded")

aggiorna_dati = 0 # 0 no
disegna = 0 # i grafici rallentano il programma, con questa var si possono disabilitare (0)
data_example = 1 # 0 no
salva_su_file = 1 # 0 no

config = {
    "alpha_vantage": {
        "key": "demo",
        "mykey": "URXBRZFHAJXF1NSB",
        "symbol": "IBM",
        "outputsize": "full",
        "key_adjusted_close": "5. adjusted close", # It is considered an industry best practice to use split/dividend-adjusted prices instead of raw prices to model stock price movements. 
    },
    "data": {
        "window_size": 20,
        "days_predicted": 1,
        "train_split_size": 0.9,
    },
    "plots": {
        "xticks_interval": 180, # show a date every xx days
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 5, # MOD dcp, rsi, stochrsi, willr, roc 
        "output_size": 1,
        "num_lstm_layers": 2,
        "lstm_size": 32, # MOD erano 32
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 64, # +++ MOD erano 64
        "num_epoch": 5, # +++ MOD erano 100
        "learning_rate": 0.01,# +++ MOD era 0.01
        "scheduler_step_size": 40,
    }
}


# se esistono i file coi dati li apre, altrimenti scarica i dati da AlphaVantage
if(aggiorna_dati == 0 &
   os.path.isfile("date_file.npy") & 
   os.path.isfile("dcp_file.npy") & 
   os.path.isfile("rsi_file.npy") & 
   os.path.isfile("stochrsi_file.npy") & 
   os.path.isfile("willr_file.npy") & 
   os.path.isfile("roc_file.npy")):

    data_date = np.load("date_file.npy")
    dcp = np.load("dcp_file.npy")
    rsi = np.load("rsi_file.npy")
    stochrsi = np.load("stochrsi_file.npy")
    willr = np.load("willr_file.npy")
    roc = np.load("roc_file.npy")  
else:
    print("Scaricando i dati...")
    def download_data(config):
        ts = TimeSeries(key=config["alpha_vantage"]["key"])
        ti = TechIndicators(key=config["alpha_vantage"]["mykey"])

        def elaborate_data(name, data):
            data_date = [date for date in data.keys()] # keys() restituisce la lista degli arg dell'ogg stesso
            data_date.reverse() # contiene lista date
            data_= [float(data[date][name]) for date in data.keys()]
            data_.reverse() # inverto l'array
            data_ = np.array(data_) # trasformo in array di numpy
            
            num_data_points = len(data_date)
            display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
            print("Number data points", num_data_points, display_date_range)

            return data_, data_date

        name = config["alpha_vantage"]["key_adjusted_close"]
        data, _ = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])
        dcp, data_date = elaborate_data(name, data)
        
        name = "RSI"
        data, _ = ti.get_rsi(config["alpha_vantage"]["symbol"], 'daily', '14', 'close')
        rsi, _ = elaborate_data(name, data)
        
        name = "FastK"
        data, _ = ti.get_stochrsi(config["alpha_vantage"]["symbol"], 'daily', '14', 'close')
        stochrsi, _ = elaborate_data(name, data)
        
        name = "WILLR"
        data, _ = ti.get_willr(config["alpha_vantage"]["symbol"], 'daily', '14')
        willr, _ = elaborate_data(name, data)
        
        name = "ROC"
        data, _ = ti.get_roc(config["alpha_vantage"]["symbol"], 'daily', '14')
        roc, _ = elaborate_data(name, data)
        
        return data_date, dcp, rsi, stochrsi, willr, roc
    
    
    data_date, dcp, rsi, stochrsi, willr, roc = download_data(config)
    # come � fatto dcp: Mat(1,len(dcp)) cio� un vettore coi valori dal primo a ieri, in questo ordine per via del reverse

    np.save("date_file", data_date)
    np.save("dcp_file", dcp)
    np.save("rsi_file", rsi)
    np.save("stochrsi_file", stochrsi)
    np.save("willr_file", willr)
    np.save("roc_file", roc)

print("Dati caricati")

# Rendo i vettori della stessa lunghezza, gettando i valori pi� vecchi nel caso differiscano,
# fondamentale perch� DataFrame non accetta vettori con lunghezze differenti fra loro
len_dcp, len_rsi, len_stochrsi, len_willr, len_roc = len(dcp), len(rsi), len(stochrsi), len(willr), len(roc)
min_len = min(len_dcp, len_rsi, len_stochrsi, len_willr, len_roc)

data_date = data_date[len_dcp-min_len:]
dcp = dcp[len_dcp-min_len:]
rsi = rsi[len_rsi-min_len:]
stochrsi = stochrsi[len_stochrsi-min_len:]
willr = willr[len_willr-min_len:]
roc = roc[len_roc-min_len:]

# creo DataFrame
data = {'dcp': dcp,
        'rsi': rsi,
        'stochrsi': stochrsi,
        'willr': willr,
        'roc': roc}
df = pd.DataFrame(data, index=data_date)
print("DataFrame creato")
if salva_su_file == 1: 
    df.to_csv("panda.csv")
    print("e salvato in csv")


if(disegna == 1):
    df_to_plot = df.reset_index() # stampare coll'indice e' lento
    plt.plot(df_to_plot.dcp)
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.title("IBM dcp")
    plt.savefig("initial_plot.png", dpi=250)
    plt.show();

print("Ora normalizzo")
class Normalizer():
    # una gaussiana ha 2 parametri: mu (indica la media, il centro, x del picco) e la variazione std
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x): 
        self.mu = np.mean(x, axis=(0)) # mean() fa la media dei dati sull'asse specificato. keepdims=True tolto perch� non accettato da DataFrame
        self.sd = np.std(x, axis=(0)) # deviazione std.  
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu

scaler = Normalizer()
df = scaler.fit_transform(df)

X, y = df, df.dcp.values
if(data_example == 1): 
    print("X.shape X=df: ", X.shape, "y.shape y=df.dcp: ", y.shape)

def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in # = i + 20
        # out_end_ix = end_ix + n_steps_out # i + 20 + 1 = end_ix + 1 MOD: c'era -1, vd sotto per spiegaz
        # check if we are beyond the dataset
        if end_ix + n_steps_out > len(input_sequences): break
        # gather input and output of the pattern
        seq_x = input_sequences[i:end_ix] # seq_x = inp[0,20]; seq_x = inp[1,21]; seq_x = inp[2,22] etc
        seq_y = output_sequence[end_ix:end_ix+1] # seq_y = out[21]; seq_y = out[22]; seq_y = out[23] etc

        X.append(seq_x), y.append(seq_y)      

        # originale: mod perch� cos� inutile, per lui funziona perch� usa tutti gli indicatori per prevedere il prezzo di oggi, mentre io uso anche il prezzo per prevedere il prezzo di domani
        # seq_y = output_sequence[end_ix-1:out_end_ix] # seq_y = out[19, 20]; seq_y = out[20, 21]; seq_y = out[21, 22] etc
        
    return np.array(X), np.array(y)
# the function boils down to getting 20 samples from X, then looking at the 1 next indices in y, and patching these together.
# Note that because of this we'll throw out the first 1 values of y

X_ss, y_mm = split_sequences(X, y, config["data"]["window_size"], config["data"]["days_predicted"])


train_test_cutoff = round(config["data"]["train_split_size"] * len(X))

X_train = X_ss[:train_test_cutoff]
X_test = X_ss[train_test_cutoff:]
y_train = y_mm[:train_test_cutoff]
y_test = y_mm[train_test_cutoff:]

X_train_tensors = torch.tensor(X_train, requires_grad=True)
X_test_tensors = torch.tensor(X_test, requires_grad=True)
y_train_tensors = torch.tensor(y_train, requires_grad=True)
y_test_tensors = torch.tensor(y_test, requires_grad=True)


if(data_example == 1):
    print("X_ss.shape: ", X_ss.shape, "y_mm.shape: ", y_mm.shape)
    print("ultimo X: X_test[-1] ", X_test[-1])
    print("ultimo X: X_test[-2] ", X_test[-2])
    print("ultimo X: X_test[-3] ", X_test[-3])
    print("ultimo y: y_test[-1]", y_test[-1])
    print("ultimo y: y_test[-2]", y_test[-2])
    print("ultimo y: y_test[-3]", y_test[-3])
    
# X_train_tensors = torch.reshape(X_train_tensors,   
#                                       (X_train_tensors.shape[0], 100, 
#                                        X_train_tensors.shape[2]))
# X_test_tensors = torch.reshape(X_test_tensors,  
#                                      (X_test_tensors.shape[0], 100, 
#                                       X_test_tensors.shape[2])) 


class LSTM(nn.Module):
    
    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size # input size
        self.num_layers = num_layers # number of recurrent layers in the lstm
        self.hidden_size = hidden_size # neurons in each lstm layer
        self.output_size = output_size # output size

        # LSTM model
        self.fc_1 =  nn.Linear(input_size, hidden_size) # fully connected 
        self.relu = nn.ReLU()
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=config["model"]["dropout"]) # lstm
        self.dropout = nn.Dropout(config["model"]["dropout"])

        self.fc_2 = nn.Linear(num_layers * hidden_size, output_size) # fully connected last layer
     
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self,x):
        # prepara hidden state e cell state
        # h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True)
        # c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True)
        # propagate input through LSTM
        x = x.float()
        batchsize = x.shape[0]   
        # layer 1
        x = self.fc_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        h_n = h_n.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        
        # layer 2
        x = self.dropout(x)
        out = self.fc_2(x)       

        return out

def training_loop(n_epochs, lstm, optimiser, loss_fn, X_train, y_train,X_test, y_test):
    for epoch in range(n_epochs):
        lstm.train()

        X_train = X_train.float()
        y_train = y_train.float()
        X_test = X_test.float()
        y_test = y_test.float()

        outputs = lstm.forward(X_train) # forward pass
        optimiser.zero_grad() # calculate the gradient, manually setting to 0
        # obtain the loss function
        loss = loss_fn(outputs, y_train)
        loss.backward() # calculates the loss of the loss function
        optimiser.step() # improve from loss, i.e backprop
        # test loss
        lstm.eval()
        test_preds = lstm(X_test)
        test_loss = loss_fn(test_preds, y_test)
    
        # lr = scheduler.get_last_lr()[0]
        print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f}'
        .format(epoch + 1, n_epochs, loss.item(), test_loss.item()))
            
# ??? prob per i filtri in print
# import warnings
# warnings.filterwarnings('ignore')

lstm = LSTM(  config["model"]["output_size"],
              config["model"]["input_size"], 
              config["model"]["lstm_size"], 
              config["model"]["num_lstm_layers"])

loss_fn = torch.nn.MSELoss()    # mean-squared error for regression
optimiser = torch.optim.Adam(lstm.parameters(), lr=config["training"]["learning_rate"]) 
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

training_loop(n_epochs=config["training"]["num_epoch"],
              lstm=lstm,
              optimiser=optimiser,
              loss_fn=loss_fn,
              X_train=X_train_tensors,
              y_train=y_train_tensors,
              X_test=X_test_tensors,
              y_test=y_test_tensors)

# df_X_ss = scaler.fit_transform(df.drop(columns=['n_dcp'])) # old transformers
# df_X_ss = scaler.fit_transform(df) # old transformers

# df_y_mm = scaler.fit_transform(df.n_dcp.values).reshape(-1, 1) # old transformers

# split the sequence
df_X_ss, df_y_mm = split_sequences(df, df.dcp.values.reshape(-1, 1), 20, 1)

# converting to tensors
# df_X_ss = Variable(torch.Tensor(df_X_ss))
# df_y_mm = Variable(torch.Tensor(df_y_mm))
# uso Variable bench� deprecato perch� in questo caso l'altro metodo d� errore, da approfondire

df_X_ss = torch.tensor(df_X_ss, requires_grad=True)
df_y_mm = torch.tensor(df_y_mm,  requires_grad=True)

# reshaping the dataset
# df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 20, df_X_ss.shape[2]))
print("df_X_ss.shape: ", df_X_ss.shape)
train_predict = lstm(df_X_ss) # forward pass
data_predict = train_predict.data.numpy() # numpy conversion
dataY_plot = df_y_mm.data.numpy()

# reverse transformation
# data_predict = scaler.inverse_transform(data_predict) 
# dataY_plot = scaler.inverse_transform(dataY_plot)

true, preds = [], []
for i in range(len(dataY_plot)):
    true.append(dataY_plot[i][0])
for i in range(len(data_predict)):
    preds.append(data_predict[i][0])

if(disegna == 1):
    plt.figure(figsize=(10,6)) #plotting
    plt.axvline(x=train_test_cutoff, c='r', linestyle='--') # size of the training set
    plt.plot(true, label='Actual Data') # actual plot
    plt.plot(preds, label='Predicted Data') # predicted plot
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.savefig("whole_plot.png", dpi=300)
    plt.show() 
    
    # zoom
    plt.figure(figsize=(10,6)) #plotting
    plt.plot(true[train_test_cutoff:], label='Actual Data') # actual plot
    plt.plot(preds[train_test_cutoff:], label='Predicted Data') # predicted plot
    plt.title('Only test data')
    plt.legend()
    plt.savefig("test_plot.png", dpi=300)
    plt.show() 

if(salva_su_file == 1):
    np.savetxt("reali.csv", np.flip(np.round(true, 2), 0), delimiter='\n', header="Reali", fmt="%2f")
    np.savetxt("previsti.csv", np.flip(np.round(preds, 2), 0), delimiter='\n', header="Previsti", fmt="%2f")


# ultima previsione: prossima chiusura
test_predict = lstm(X_test_tensors[-1].unsqueeze(0)) # get the last sample
test_predict = test_predict.detach().numpy()
# test_predict = scaler.inverse_transform(test_predict)
test_predict = test_predict[0].tolist()

test_target = y_test_tensors[-1].detach().numpy() # last sample again
# test_target = scaler.inverse_transform(test_target.reshape(1, -1))
test_target = test_target[0].tolist()

print("test_predict", test_predict)
print("test_target", test_target)

print("FINE, CIAO")