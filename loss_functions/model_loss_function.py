import torch
import torch.nn as nn
import numpy as np

class model_loss(nn.Module):
    def __init__(self,alpha,beta):
        super(model_loss,self).__init__()

#
#---------------------------------------------------------------------------
#
#       loss =  model_loss(alpha,beta)
#                  -alpha (float): weight for cyclic-model based loss [default: 0.5]
#                  -beta (float): weight for standard L1 loss [default: 1.0]
#
#---------------------------------------------------------------------------
#

        # weights for loss function
        self.alpha = alpha
        self.beta = beta

        # data trackers for visualization
        self.train_tracker = {'model':[],'l1':[],'all':[]}
        self.validation_tracker = {'model':[],'l1':[],'all':[]}
        self.test_tracker = {'model':[],'l1':[],'all':[]}

    def cyclic_model_loss(self,outputs,input_signal,TI,mode):

#
#------------------------------------------------------------------------------------------------
#
#       loss =  cyclic_model_loss(alpha,beta)
#                  -outputs [nb,ch,1]: T1NET predicted A, B, and T1 parameters 
#                  -input_signal [nb,nt,ch]: raw T1-weighted images
#                  -TI [nb,nt,ch]: raw inversion times
#                  -mode (string: 'train','validation','test'): tracker for 
#                                                               training,validation,and testing
#                        -nb: batch dimension
#                        -nt: number of T1 weighted images and inversion times used for training
#                        -ch: channel dimension where first channel are the inversion times and 
#                             the second channel is the T1 weighted images (2) 
#
#------------------------------------------------------------------------------------------------
#

        # T1NET learns to perform the MOLLI T1 correction factor 
        # including this equation in the cyclic model-based loss tended to destabilized training
        # and was not used
        if self.alpha:
            A = outputs[:,0,None,:]
            B = outputs[:,1,None,:]
            T1 = torch.abs(outputs[:,2,None,:])
            output_signal = torch.abs(A - B*torch.exp(-TI/T1))
            
            loss = torch.mean(torch.abs(output_signal - input_signal))
        else:
            loss = torch.tensor(0).cuda()

        # cyclic model-based loss data tracker for training, validation, and testing
        if mode == 'train':
            self.train_tracker['model'].append(loss.clone().detach().cpu().numpy())
        if mode == 'validation':
            self.validation_tracker['model'].append(loss.clone().detach().cpu().numpy())
        if mode == 'test':
            self.test_tracker['model'].append(loss.clone().detach().cpu().numpy())

        return loss

    def l1_loss(self,outputs,targets,mode):

#
#------------------------------------------------------------------------------------------------
#
#       loss =  l1_loss(alpha,beta)
#                  -outputs [nb,ch,1]: T1NET predicted A, B, and T1 parameters (only T1 used here)
#                  -targets [nb,nt,ch]: reference MOLLI-5(3)3 T1 maps
#                  -mode (string: 'train','validation','test'): tracker for 
#                                                               training,validation,and testing
#                        -nb: batch dimension
#                        -nt: number of T1 weighted images and inversion times used for training
#                        -ch: channel dimension where first channel are the inversion times and 
#                             the second channel is the T1 weighted images (2) 
#
#------------------------------------------------------------------------------------------------
#

        if self.beta:
            loss = torch.mean(torch.abs(targets - outputs[:,-1,None,:]))
        else:
            loss = torch.tensor(0).cuda()

        # l1 loss data tracker for training, validation, and testing        
        if mode == 'train':
            self.train_tracker['l1'].append(loss.clone().detach().cpu().numpy())
        if mode == 'validation':
            self.validation_tracker['l1'].append(loss.clone().detach().cpu().numpy())
        if mode == 'test':
            self.test_tracker['l1'].append(loss.clone().detach().cpu().numpy())

        return loss

    #returns loss function data metrics
    def return_metrics(self,mode):
        if mode == 'train':
            tracker = self.train_tracker.copy()
            return {key:np.mean(val) for key,val in tracker.items()}
        if mode == 'validation':
            tracker = self.validation_tracker.copy()
            return {key:np.mean(val) for key,val in tracker.items()}            
        if mode == 'test':
            tracker = self.test_tracker.copy()
            return {key:np.mean(val) for key,val in tracker.items()}            

    # clears data trackers every epoch
    def clear_metrics(self,mode):
        if mode == 'train':
            for key,val in self.train_tracker.items():
                self.train_tracker[key] = []
        if mode == 'validation':
            for key,val in self.validation_tracker.items():
                self.validation_tracker[key] = []
        if mode == 'test':
            for key,val in self.test_tracker.items():
                self.test_tracker[key] = []

    # full cyclic model-based loss function call
    def forward(self,outputs,targets,signal,TI,mode):

        loss = self.alpha*self.cyclic_model_loss(outputs,signal,TI,mode) + self.beta*self.l1_loss(outputs,targets,mode)

        # full cyclic model-based loss data tracker for training, validation, and testing        
        if mode == 'train':
            self.train_tracker['all'].append(loss.clone().detach().cpu().numpy())
        if mode == 'validation':
            self.validation_tracker['all'].append(loss.clone().detach().cpu().numpy())
        if mode == 'test':
            self.test_tracker['all'].append(loss.clone().detach().cpu().numpy())

        return loss
