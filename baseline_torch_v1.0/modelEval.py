import scipy.io as scio
import torch

from modelDesign import *
from utils import *

device = get_device()

## Parameters
sigma2_UE = 0.1
input_dim = 3

## Load data
f = scio.loadmat('./train.mat')
data = f['train'] #f['test']
location_data = data[0,0]['loc']
channel_data = data[0,0]['CSI']


########################################################   Scheme 1   ########################################################
# ### Load model
RadioMap = RadioMap_Model_TypeI(input_dim=input_dim).to(device)
RadioMap.load_state_dict(torch.load("./RadioMap_TypeI.pth"))


###  Prediction 
test_tensor = get_tensor(location_data, device=device)
PrecodingVector = inference(test_tensor, RadioMap)
PrecodingVector=np.reshape(PrecodingVector,(-1,SC_num,Tx_num,1))

### Calculate the score
SubCH_gain_codeword = EqChannelGain(channel_data, PrecodingVector)
data_rate = DataRate(SubCH_gain_codeword, sigma2_UE)
print('The score of RadioMap_TypeI is %f bps/Hz' %data_rate)




