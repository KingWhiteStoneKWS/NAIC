# import tensorflow as tf
# from tensorflow import keras
import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio

from modelDesign import *
from utils import *

## device
device = get_device()

## Parameters
SC_num = 128  ## subcarrier number
Tx_num = 32 ## Tx antenna number
Rx_num = 4  ## Rx antenna number
sigma2_UE =  0.1

#### Read Data
f = scio.loadmat('./train.mat')
data = f['train']
location_data = data[0,0]['loc']
channel_data = data[0,0]['CSI']
##neww
Codebook,Codeword = CodeBookCodeWord(Tx_num,channel_data)
print("Codebook:   Codeword:")
print(Codebook.shape)
print(Codeword.shape)
Codebook = torch.from_numpy(Codebook)
#

## basic train parameters
loss_fn = nn.MSELoss()
EPOCHS =100
LR = 1e-4
batch_size = 32


########################################################   Scheme 2   ########################################################
########################################################   Scheme 2   ########################################################
## ANN for codeword index
train_dataset_iter2, val_dataset_iter2 = get_dataset_iter(location_data, Codeword, device)

input_dim = 3
net_type2 = ANN_TypeII(input_dim).to(device)
opt_adam_type2 = torch.optim.Adam(net_type2.parameters(),lr=LR)

for i in range(EPOCHS):
    print(f"Epoch {i+1}\n-------------------------------")
    train(train_dataset_iter2, net_type2, loss_fn, opt_adam_type2)
    verify(val_dataset_iter2, net_type2, loss_fn)
print("Done!")

#### Prediction
test_tensor = get_tensor(location_data, device=device)
Codeword_est = inference(test_tensor, net_type2)
data_rate2 = DataRate(Codeword_est, sigma2_UE)

### Generate Radio Map (Input:location, Output:beamforming vector)
RadioMap_TypeII = RadioMap_Model_TypeII(Codebook=Codebook,input_dim=input_dim)
RadioMap_TypeII.to(device)
torch.save(RadioMap_TypeII.state_dict(), "./RadioMap_TypeII.pth")
print("Saved PyTorch Model State to RadioMap_TypeII.pth")

####
########################################################   Scheme 2   ########################################################
print('The score of RadioMapI is %.8f bps/Hz'  %data_rate2)




