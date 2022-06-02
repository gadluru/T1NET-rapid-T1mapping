import sys
sys.path.append('../utils')
sys.path.append('../template_models')

import torch
import pickle
import numpy as np
from loadData import *
from T1NET_model import *
import matplotlib.pyplot as plt
from supportingFunctions import *

# ------------------------------------------------------------------------------------------------------
#
# script for visualizing results of trained network
#
# ------------------------------------------------------------------------------------------------------


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Modify this line with the location of the trained network
directory = 'trainedNetwork/31May_0143pm/'

with open(directory + 'parameters.pkl','rb') as f:
    parameters = pickle.load(f)

state = torch.load(directory + parameters['save_directory'] + '.pth')

save_location = directory + 'results/'

if not os.path.exists(save_location):
    os.makedirs(save_location)

print('saving results to ' + save_location)

test_set = loadData(split='test',N=parameters['seq_len'],path=directory)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=parameters['batch_size'],shuffle=False)

model = T1NET_model(input_size=parameters['input_size'],hidden_size=parameters['hidden_size'],
                    num_layers=parameters['num_layers'],seq_len=parameters['seq_len'],
                    output_dim=parameters['output_dim'],bidirectional=parameters['bidirectional'],
                    batch_first=parameters['batch_first']).cuda()

model.load_state_dict(state['state_dict'])

networks,targets,myoMasks = test_network(model,test_loader)

#select slices to visualize
sets = [0,1,2]

T1plot(outputs=networks[:,-1,:],targets=targets,save_name=save_location + 'figure.png',
       clip=1500,sets=sets,cmap='viridis')
T1metrics(writer=None,step=None,T1=T1segment(networks[:,-1,:],targets,myoMasks,sets=sets),
          save_name=save_location + 'correlation_bland_altman.png')
