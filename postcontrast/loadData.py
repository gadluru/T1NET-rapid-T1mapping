import sys
sys.path.append('../utils')

import pickle
import os.path
import scipy.io
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from  supportingFunctions import *
from torch.utils.data import Dataset

class loadData(Dataset):
    def __init__(self,split='train', N=3, path=''):

# -------------------------------------------------------------------------------------------
#
#     inputs,T1,signal,TI,myoMask = loadData(split='train', N=3, path)
#                                      - loads and prepares T1 mapping datasets for training 
#                                        and testing the T1NET
#
# -------------------------------------------------------------------------------------------
#    
#     inputs (Preprocessed T1 mapping datasets)
#        -Preprocessed T1 mapping datasets have dimensions [b,ch,sx,sy,nt]
#            -b: batch dimension
#            -ch: channel dimension (1)
#            -sx: spatial dimension (128)
#            -sy: spatial dimension (128)
#            -nt: number of T1 weighted images and inversion times (8)
#
#        -N (integer: 1-8): number of training T1 weighted images and inverion times for each 
#                           batch [default: 3]
#        -split (string: 'train|val|test'): load selected datasets [default:'train']
#        -path (string): save location for training files
#
# -------------------------------------------------------------------------------------------
#     outputs
#        -inputs  [nb,nt,ch]: inputs to feed into the T1NET 
#        -T1      [nb,1,1]: MOLLI-5(3)3 references for training the T1NET
#        -signal  [nb,nt,ch]: raw T1-weighted signals for cyclic model-based loss
#        -TI      [nb,nt,ch]: raw inversion times for cyclic model-based loss
#        -myoMask [nb,1,1]: myocardial masks based on standard AHA 16-segment model 
#                          (validation and test only)
#              -nb: batch dimension
#              -nt: number of T1 weighted images and inversion times used for training chosen by N 
#              -ch: channel dimension where first channel are the inversion times and the second 
#                   channel is the T1 weighted images (2) 
#
# -------------------------------------------------------------------------------------------

        self.split = split

        # Modify: use if full dataset is available
        # datasets = scipy.io.loadmat('data/postcontrast_datasets.mat')

        # Modify: use to visualize example testsets
        datasets = scipy.io.loadmat('data/postcontrast_testsets.mat')

        if self.split == 'train':
            self.signal = np.array(datasets['train']['input'][0,0][...,:N],dtype=np.float32)
            self.TI = np.array(datasets['train']['TI'][0,0][...,:N],dtype=np.float32)
            self.T1 = np.array(datasets['train']['T1'][0,0],dtype=np.float32)

            nb,ch,sx,sy,nt = self.signal.shape
            
            self.TI = np.tile(self.TI,[1,1,sx,sy,1])
            self.TI = np.reshape(self.TI,[nb*ch*sx*sy,nt])
            
            self.T1 = np.reshape(self.T1,[nb*ch*sx*sy,1,1])
            
            self.signal = np.reshape(self.signal,[nb*ch*sx*sy,nt])

            self.signal_mean = np.mean(self.signal)
            self.signal_std = np.std(self.signal)

            with open(path + 'signal_scale_factors.pkl','wb') as f:
                pickle.dump([self.signal_mean,self.signal_std],f)

            self.TI_mean = np.mean(self.TI)
            self.TI_std = np.std(self.TI)

            with open(path + 'TI_scale_factors.pkl','wb') as f:
                pickle.dump([self.TI_mean,self.TI_std],f)

            self.input = np.stack(((self.TI - self.TI_mean)/self.TI_std,(self.signal - self.signal_mean)/self.signal_std),axis=-1)

            self.signal = self.signal[...,None]
            self.TI = self.TI[...,None]

        elif self.split == 'validation':
            self.myoMask = np.array(datasets['validation']['myoMask'][0,0],dtype=np.float32)
            self.signal = np.array(datasets['validation']['input'][0,0][...,:N],dtype=np.float32)
            self.TI = np.array(datasets['validation']['TI'][0,0][...,:N],dtype=np.float32)
            self.T1 = np.array(datasets['validation']['T1'][0,0],dtype=np.float32)

            nb,ch,sx,sy,nt = self.signal.shape
    
            self.TI = np.tile(self.TI,[1,1,sx,sy,1])
            self.TI = np.reshape(self.TI,[nb*ch*sx*sy,nt])
        
            self.myoMask = np.reshape(self.myoMask,[nb*ch*sx*sy,1,1])
            self.T1 = np.reshape(self.T1,[nb*ch*sx*sy,1,1])
        
            self.signal = np.reshape(self.signal,[nb*ch*sx*sy,nt])

            if os.path.isfile(path + 'signal_scale_factors.pkl'):
                with open(path + 'signal_scale_factors.pkl','rb') as f:
                    self.signal_mean,self.signal_std = pickle.load(f)
            else:
                print('file does not exist')
                sys.exit()

            if os.path.isfile(path + 'TI_scale_factors.pkl'):
                with open(path + 'TI_scale_factors.pkl','rb') as f:
                    self.TI_mean,self.TI_std = pickle.load(f)
            else:
                print('file does not exist')
                sys.exit()

            self.input = np.stack(((self.TI - self.TI_mean)/self.TI_std,(self.signal - self.signal_mean)/self.signal_std),axis=-1)

            self.signal = self.signal[...,None]
            self.TI = self.TI[...,None]

        elif self.split == 'test':
            self.myoMask = np.array(datasets['test']['myoMask'][0,0],dtype=np.float32)
            self.signal = np.array(datasets['test']['input'][0,0][...,:N],dtype=np.float32)
            self.TI = np.array(datasets['test']['TI'][0,0][...,:N],dtype=np.float32)
            self.T1 = np.array(datasets['test']['T1'][0,0],dtype=np.float32)

            nb,ch,sx,sy,nt = self.signal.shape
    
            self.TI = np.tile(self.TI,[1,1,sx,sy,1])
            self.TI = np.reshape(self.TI,[nb*ch*sx*sy,nt])
        
            self.myoMask = np.reshape(self.myoMask,[nb*ch*sx*sy,1,1])
            self.T1 = np.reshape(self.T1,[nb*ch*sx*sy,1,1])
        
            self.signal = np.reshape(self.signal,[nb*ch*sx*sy,nt])

            if os.path.isfile(path + 'signal_scale_factors.pkl'):
                with open(path + 'signal_scale_factors.pkl','rb') as f:
                    self.signal_mean,self.signal_std = pickle.load(f)
            else:
                print('file does not exist')
                sys.exit()

            if os.path.isfile(path + 'TI_scale_factors.pkl'):
                with open(path + 'TI_scale_factors.pkl','rb') as f:
                    self.TI_mean,self.TI_std = pickle.load(f)
            else:
                print('file does not exist')
                sys.exit()

            self.input = np.stack(((self.TI - self.TI_mean)/self.TI_std,(self.signal - self.signal_mean)/self.signal_std),axis=-1)

            self.signal = self.signal[...,None]
            self.TI = self.TI[...,None]

    def __getitem__(self,index):
        if self.split == 'train':
            return self.input[index],self.T1[index],self.signal[index],self.TI[index]
        if self.split == 'validation':
            return self.input[index],self.T1[index],self.signal[index],self.TI[index],self.myoMask[index]
        if self.split == 'test':
            return self.input[index],self.T1[index],self.signal[index],self.TI[index],self.myoMask[index]
        
    def __len__(self):
        return len(self.input)
