# import tensorflow as tf
# from tensorflow.keras import layers
# from tensorflow.keras.layers import BatchNormalization
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

Rx_num = 4
Tx_num = 32
SC_num = 128

def DownPrecoding(channel_est):  ### Optional
    ### estimated channel
    HH_est = np.reshape(channel_est, (-1, Rx_num, Tx_num, SC_num, 2)) ## Rx, Tx, Subcarrier, RealImag
    HH_complex_est = HH_est[:,:,:,:,0] + 1j * HH_est[:,:,:,:,1]  ## Rx, Tx, Subcarrier
    HH_complex_est = np.transpose(HH_complex_est, [0,3,1,2])

    ### precoding based on the estimated channel
    MatRx, MatDiag, MatTx = np.linalg.svd(HH_complex_est, full_matrices=True) ## SVD
    PrecodingVector = np.conj(MatTx[:,:,0,:])  ## The best eigenvector (MRT transmission)
    PrecodingVector = np.reshape(PrecodingVector,(-1, SC_num, Tx_num, 1))
    return PrecodingVector

def EqChannelGain(channel, PrecodingVector):
    ### The authentic CSI
    HH = np.reshape(channel, (-1, Rx_num, Tx_num, SC_num, 2)) ## Rx, Tx, Subcarrier, RealImag
    HH_complex = HH[:,:,:,:,0] + 1j * HH[:,:,:,:,1]  ## Rx, Tx, Subcarrier
    HH_complex = np.transpose(HH_complex, [0,3,1,2])

    ### Power Normalization of the precoding vector
    Power = np.matmul(np.transpose(np.conj(PrecodingVector), (0, 1, 3, 2)), PrecodingVector)
    PrecodingVector = PrecodingVector/ np.sqrt(Power)

    ### Effective channel gain
    R = np.matmul(HH_complex, PrecodingVector)
    R_conj = np.transpose(np.conj(R), (0, 1, 3, 2))
    h_sub_gain =  np.matmul(R_conj, R)
    h_sub_gain = np.reshape(np.absolute(h_sub_gain), (-1, SC_num))  ### channel gain of SC_num subcarriers
    return h_sub_gain

def CodeBookCodeWord(Tx_num, channel_data):
    #### Define a beamforming codebook
    codebook_size = Tx_num
    DFTcodebook = np.zeros([codebook_size,codebook_size], dtype=complex)  ## In this example, we define a DFT codebook
    for isub in range(codebook_size):
        DFTcodebook[isub, :] = np.exp(isub * 1j * np.pi * 2 * np.arange(0, codebook_size, 1) / codebook_size)

    ### Codeword-performance table
    SubCH_gain = np.zeros([channel_data.shape[0], SC_num, codebook_size])
    for ixx in range(codebook_size):
        codeword_temp = DFTcodebook[ixx, :].reshape(-1, 1)
        codeword_temp = np.expand_dims(codeword_temp, axis=0)
        codeword_temp = np.repeat(codeword_temp, SC_num, axis=0)
        codeword_temp = np.expand_dims(codeword_temp, axis=0)
        codeword_temp = np.repeat(codeword_temp, channel_data.shape[0], axis=0)
        SubCH_gain_codeword = EqChannelGain(channel_data, codeword_temp)
        SubCH_gain[:, :, ixx] = SubCH_gain_codeword
    return DFTcodebook, SubCH_gain

def DataRate(h_sub_gain, sigma2_UE):  ### Score
    SNR = h_sub_gain / sigma2_UE
    Rate = np.log2(1 + SNR)  ## rate
    Rate_OFDM = np.mean(Rate, axis=-1)  ###  averaging over subcarriers
    Rate_OFDM_mean = np.mean(Rate_OFDM)  ### averaging over CSI samples
    return Rate_OFDM_mean


##################################### define neural network #################################
class ANN_TypeI(nn.Module): ## Neural Network (input:location, output: the estimated CSI)
    def __init__(self, in_features):
        super(ANN_TypeI, self).__init__()
        self.x_con1 = nn.Linear(in_features, out_features=50)
        self.x_con2 = nn.BatchNorm1d(num_features=50)
        self.x_con3 = nn.Linear(in_features=50, out_features=100)
        self.x_con4 = nn.BatchNorm1d(num_features=100)
        self.x_con5 = nn.Linear(in_features=100, out_features=Rx_num*Tx_num*SC_num*2)        

    def forward(self, p):
        x_con = self.x_con1(p)
        x_con = self.x_con2(x_con)
        x_con = self.x_con3(x_con)
        x_con = self.x_con4(x_con)
        x_con = self.x_con5(x_con)
        output = torch.reshape(x_con, (-1, Rx_num, Tx_num, SC_num, 2))
        return output

class RadioMap_Model_TypeI(nn.Module): ### Generate RadioMapI (Input:location, Output:beamforming vector)
    def __init__(self, input_dim=3):
        super(RadioMap_Model_TypeI, self).__init__()
        self.ann = ANN_TypeI(input_dim) ## Neural Network (input:location, output: the estimated CSI)

    def forward(self, p):
        CSI_est = self.ann(p)
        HH_est = torch.reshape(CSI_est, [-1, Rx_num, Tx_num, SC_num, 2]) ## the estimated CSI, shape: Rx, Tx, Subcarrier, RealImag
        HH_complex_est = torch.complex(HH_est[:,:,:,:,0], HH_est[:,:,:,:,1])
        HH_complex_est = HH_complex_est.permute(0,3,1,2)
        _, _, MatTx = torch.linalg.svd(HH_complex_est, full_matrices=True) ## Note that tf.svd is different from np.svd
        PrecodingVector = MatTx[:,:,:,0]  ## The best eigenvector (MRT transmission)
        PrecodingVector = torch.reshape(PrecodingVector,[-1, SC_num, Tx_num, 1])
        return PrecodingVector

class ANN_TypeII(nn.Module): ## Neural Network (input:location, output: codeword index)
    def __init__(self, in_features):
        super(ANN_TypeII, self).__init__()
        self.x_con1 = nn.Linear(in_features, out_features=50)
        self.x_con2 = nn.BatchNorm1d(num_features=50)
        self.x_con3 = nn.Linear(in_features=50, out_features=100)
        self.x_con4 = nn.BatchNorm1d(num_features=100)
        self.x_con5 = nn.Linear(in_features=100, out_features=SC_num*Tx_num)  
        self.x_con6 = nn.ReLU()      

    def forward(self, p):
        x_con = self.x_con1(p)
        x_con = self.x_con2(x_con)
        x_con = self.x_con3(x_con)
        x_con = self.x_con4(x_con)
        x_con = self.x_con5(x_con)
        x_con = self.x_con6(x_con)
        output = torch.reshape(x_con, (-1, SC_num, Tx_num))
        return output


class RadioMap_Model_TypeII(nn.Module): ### Generate RadioMapII (Input:location, Output:beamforming vector)
    def __init__(self, Codebook, input_dim=3):
        super(RadioMap_Model_TypeII, self).__init__()
        self.ann = ANN_TypeII(input_dim) ## Neural Network (input:location, output: codeword index)
        self.codebook = Codebook
        self.real = self.codebook.real.to(torch.float64)
        self.imag = self.codebook.imag.to(torch.float64)

    def forward(self, p):
        SubCH_gain_est = self.ann(p)
        Beam_index = torch.argmax(SubCH_gain_est, axis=-1)
        PrecodingVector_real = self.real[Beam_index, :] 
        PrecodingVector_real = torch.reshape(PrecodingVector_real, [-1, SC_num, Tx_num, 1])
        PrecodingVector_imag = self.imag[Beam_index, :] 
        PrecodingVector_imag = torch.reshape(PrecodingVector_imag, [-1, SC_num, Tx_num, 1])
        PrecodingVector = torch.complex(PrecodingVector_real, PrecodingVector_imag)

        return PrecodingVector
