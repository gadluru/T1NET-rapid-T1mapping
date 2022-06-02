import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from T1NET import *
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#batch size
batch_size = 2**15
#input size
input_size = 2
#number of latent features
hidden_size = 64
#number of layers
num_layers = 7
#number of T1 weighted images and inversion times
seq_len = 3
#bidirection lstm flag
bidirectional = True
#batch first flag
batch_first = True
#output features
output_dim = 3
#model loss weight
alpha = 0.5
#mean absolute error weight
beta = 1
#learning rate
lr = 0.0003
#number of training epochs
nepoch = 100
#interval at which to save checkpoint networks
save_interval = 20
#save directory
directory = 'trainedNetwork/'+datetime.now().strftime("%d%b_%I%M%P") + '/'
#name of final network to save
save_directory = 'T1NET'

if not os.path.exists(directory):
    os.makedirs(directory)

parameters = {'batch_size':batch_size,'input_size':input_size,'hidden_size':hidden_size,'num_layers':num_layers,'seq_len':seq_len,'output_dim':output_dim,'bidirectional':bidirectional,'batch_first':batch_first,'alpha':alpha,'beta':beta,'lr':lr,'nepoch':nepoch,'save_interval':save_directory,'save_directory':save_directory}

with open(directory+'parameters.pkl','wb') as f:
    pickle.dump(parameters,f)

T1NET(batch_size,input_size,hidden_size,num_layers,seq_len,bidirectional,batch_first,output_dim,alpha,beta,lr,nepoch,save_interval,directory,save_directory,device)
