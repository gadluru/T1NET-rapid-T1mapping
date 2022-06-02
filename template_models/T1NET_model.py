import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

class T1NET_model(torch.nn.Module):
    def __init__(self,input_size=2,hidden_size=64,num_layers=7,seq_len=3,bidirectional=True,batch_first=True,output_dim=3,device='cuda'):
        super(T1NET_model,self).__init__()

#---------------------------------------------------------------------------------------------------
#
#    model = T1NET_model(input_size=2,hidden_size=64,num_layers=7,seq_len=3,bidirectional=True,
#                        batch_first=Tru#e,output_dim=3):
#
#---------------------------------------------------------------------------------------------------
#
#   inputs (T1NET parameters)
#
#        -input_size (integer: 2): number of input channels [1: Inversion Times 2:T1 weighted images]
#        -hidden_size (integer): number of latent features in each lstm module [default: 64]
#        -num_layers (integer): number of lstm modules in the T1NET [default: 7]
#        -seq_len (integer: 1-8): number of T1 weighted images and inversion times used for training 
#                                 [default: 3]
#        -bidirectional (boolean): flag for using bidirectional or unidirectional lstm [default: True]
#        -batch_first (boolean): flag if batch dimension is first [default:True]
#        -output_dims (integer: 3): number of output channels (A, B, and T1 from T1 recovery model)
#
#
#---------------------------------------------------------------------------------------------------
#
#   outputs (T1NET parameters)
#        - T1NET model architecture
#
#
#---------------------------------------------------------------------------------------------------

        #output dimension corresponding to A, B, and T1 from T1 recovery model
        self.output_dim = output_dim

        # LSTM encoding layer
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,bidirectional=bidirectional,batch_first=batch_first)

        # fuly connected prediction layer
        if bidirectional:
            self.fc1 = nn.Linear(2*seq_len*hidden_size,2*seq_len*hidden_size)
            self.fc2 = nn.Linear(2*seq_len*hidden_size,self.output_dim)

            h0 = torch.zeros((2*num_layers,1,hidden_size)).to(device)
            c0 = torch.zeros((2*num_layers,1,hidden_size)).to(device)
        else:
            self.fc1 = nn.Linear(seq_len*hidden_size,seq_len*hidden_size)
            self.fc2 = nn.Linear(seq_len*hidden_size,self.output_dim)

            h0 = torch.zeros((num_layers,1,hidden_size)).to(device)
            c0 = torch.zeros((num_layers,1,hidden_size)).to(device)

        # learnable hidden states
        self.h0 = nn.Parameter(h0, requires_grad=True)
        self.c0 = nn.Parameter(c0, requires_grad=True)

    def return_weights(self):
        weight_dict = {key:val.clone().detach().cpu().numpy() for key,val in self.named_parameters() if 'weight' in key}

        return weight_dict

    def forward(self,x):

        nb,nt,ch = x.shape

        output,(hn,cn) = self.lstm(x,(self.h0.repeat((1,nb,1)),self.c0.repeat((1,nb,1))))

        nb,nt,ch = output.shape

        output = self.fc1(output.reshape((nb,nt*ch)))

        output = self.fc2(output)

        output = output.reshape((nb,self.output_dim,1))

        return output,(hn,cn)
