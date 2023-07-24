# import tensorflow as tf
# from tensorflow import keras
import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio

from modelDesign import *
from utils import *
import sys
import os

#with open("output_file.txt", "w") as f:
#    sys.stdout = f
#    print("Hello, this is the output!")

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
Intermediary_data = np.transpose(channel_data,(0,3,2,1,4))
codeword_data = Intermediary_data[:,:,:,0,0]
print(codeword_data.shape)

## basic train parameters
loss_fn = nn.MSELoss()
EPOCHS =500
LR = 1e-3
batch_size=256

########################################################   Scheme 1   ########################################################
## ANN for CSI estimation
train_dataset_iter, val_dataset_iter = get_dataset_iter(location_data, channel_data, device)

input_dim = 3
net_type1 = ANN_TypeI(input_dim).to(device)
opt_adam_type1 = torch.optim.Adam(net_type1.parameters(),lr=LR)

for i in range(EPOCHS):
    print(f"Epoch {i+1}\n-------------------------------")
    train(train_dataset_iter, net_type1, loss_fn, opt_adam_type1)
    verify(val_dataset_iter, net_type1, loss_fn)
print("Done!")

#### Prediction
test_tensor = get_tensor(location_data, device=device)
channel_est = inference(test_tensor, net_type1)
PrecodingVector1 = DownPrecoding(channel_est)
SubCH_gain_codeword_1 = EqChannelGain(channel_data, PrecodingVector1)
data_rate1 = DataRate(SubCH_gain_codeword_1, sigma2_UE)

### Generate Radio Map (Input:location, Output:beamforming vector)
RadioMap_TypeI = RadioMap_Model_TypeI(input_dim=input_dim)
RadioMap_TypeI.to(device)
RadioMap_TypeI.ann = net_type1
torch.save(RadioMap_TypeI.state_dict(), "./RadioMap_TypeI.pth")
print("Saved PyTorch Model State to RadioMap_TypeI.pth")

####
########################################################   Scheme 1   ########################################################



print('The score of RadioMapI is %.8f bps/Hz'  %data_rate1)

#f.close()
#sys.stdout = temp
#print('RECOVER')

