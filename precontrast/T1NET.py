import sys
sys.path.append('../loss_functions')
sys.path.append('../template_models')
sys.path.append('../utils')

import time
import torch
import numpy as np
from loadData import *
from T1NET_model import *
from supportingFunctions import *
from model_loss_function import *
from torch.utils.tensorboard import SummaryWriter

def T1NET(batch_size,input_size,hidden_size,num_layers,seq_len,bidirectional,batch_first,output_dim,alpha,beta,lr,nepoch,save_interval,directory,save_directory,device):


# -------------------------------------------------------------------------------------------
#
#     T1NET(batch_size,in_features,out_features,resolution,num_layers,dp,alpha,beta,lr,nepoch,
#           save_interval,directory,save_directory,device)
#
# -------------------------------------------------------------------------------------------
#    
#     inputs (MOLLI-5(3)3 T1 mapping datasets)
#
#        -batch_size (integer): mini-batch size for training [default: 2**15]
#        -input_size (integer: 2): number of input channels [1: Inversion Times 2:T1 weighted images]
#        -hidden_size (integer): number of latent features in each lstm module [default: 64]
#        -num_layers (integer): number of lstm modules in the T1NET [default: 7]
#        -seq_len (integer: 1-8): number of T1 weighted images and inversion times used for training [default: 3]
#        -bidirectional (boolean): flag for using bidirectional or unidirectional lstm [default: True]
#        -batch_first (boolean): flag if batch dimension is first [default:True]
#        -output_dims (integer: 3): number of output channels (A, B, and T1 from T1 recovery model)
#        -alpha (float): weight for cyclic-model based loss of the loss function [default: 0.5]
#        -beta (float): weight for the standard L1 component of the loss function [default: 1.0] 
#        -lr (float): learning rate for the adam optimizer [default: 0.0003]
#        -nepoch (integer): number of epochs to train the network [default: 100]
#        -save_interval (integer): epoch interval to save intermediate networks [default: 20]
#        -directory (string): directory to save the network and other parameters
#        -save_directory (string): name to save network
#        -device (cuda|cpu): determines whether to train network on GPU or CPU if not available
#
# -------------------------------------------------------------------------------------------


    # save location for training files
    path = os.path.dirname(os.path.realpath(__file__)) + '/' + directory
    
    # writer for TensorBoard visualiztion
    train_writer = SummaryWriter(path + 'runs/train')
    validation_writer = SummaryWriter(path + 'runs/validation')
    test_writer = SummaryWriter(path + 'runs/test')
    
    # load data for training, validation, and testing
    train_set = loadData(split='train',N=seq_len,path=path)
    val_set = loadData(split='validation',N=seq_len,path=path)
    test_set = loadData(split='test',N=seq_len,path=path)

    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set,batch_size=batch_size,shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)
    
    # T1NET model architecture
    model = T1NET_model(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,
                        seq_len=seq_len,output_dim=output_dim,
                        bidirectional=bidirectional,batch_first=batch_first,device=device).to(device)

    # weight initializtion
    for name,param in model.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param,0.0)
        elif 'weight' in name:
            nn.init.kaiming_normal_(param)

    # cyclic model-based loss function
    loss_function = model_loss(alpha=beta,beta=beta)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    # gradient tracker during training
    grad_tracker = grad_flow(model)
    
    print('beginning training...')
    print('network saved to ', directory,' as ', save_directory)

    step = 0 
    currentLoss = 0
    for epoch in range(nepoch):
        start = time.time()
        for inputs,targets,signal,TI in train_loader:
            model.train()
            
            optimizer.zero_grad()
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            signal = signal.to(device)
            TI = TI.to(device)

            outputs,(hn,cn) = model(inputs)
            
            loss = loss_function(outputs,targets,signal,TI,'train')
            
            loss.backward()

            grad_tracker.calculate_metrics(model)
            
            optimizer.step()
            
        end = time.time()

        # Tensorboard writer for training intermediate data
        plot_grad_flow(train_writer,step,grad_tracker.return_metrics())
        tb_add_histogram(train_writer,step,model.return_weights())
        tb_add_scalar(train_writer,step,loss_function.return_metrics('train'))

        outputs,targets,masks = test_network(model,val_loader,loss_function,'validation')

        # Tensorboard writer for validation intermediate data
        tb_add_scalar(validation_writer,step,loss_function.return_metrics('validation'))
        T1metrics(validation_writer,step,T1segment(outputs[:,-1,:],targets,masks))

        outputs,targets,masks = test_network(model,test_loader,loss_function,'test')

        # Tensorboard writer for testing intermediate data
        tb_add_scalar(test_writer,step,loss_function.return_metrics('test'))
        T1metrics(test_writer,step,T1segment(outputs[:,-1,:],targets,masks))

        print('Epoch: %.d' %epoch, end='    ')
        print('Time: %.4f' %(end-start))
        print('Train Loss: %.4f' %loss_function.return_metrics('train')['all'])
        print('Validation Loss: %.4f' %loss_function.return_metrics('validation')['all'])
        print('Test Loss: %.4f' %loss_function.return_metrics('test')['all'])
        print('')

        state = {'epoch': epoch,
                 'state_dict':model.state_dict(),
                 'optimizer':optimizer.state_dict()}        

        # saves best network according to minimal validation loss
        if currentLoss == 0 or loss_function.return_metrics('validation')['all'] < currentLoss:
            currentLoss = loss_function.return_metrics('validation')['all']
            
            try:
                os.system('rm '+currentSave)
            except:
                pass
                
            currentSave = (path + save_directory + '_epoch_%.d_val_%.4f.pth' %(epoch,currentLoss))
            torch.save(state,currentSave)
            
        # saves intermediate networks according to save interval
        if epoch % save_interval == 0:
                torch.save(state,path + 'checkpoint_model_epoch_%.d_val_%.4f.pth' %(epoch,loss_function.return_metrics('validation')['all']))

        grad_tracker.clear_metrics()
        loss_function.clear_metrics('train')
        loss_function.clear_metrics('validation')
        loss_function.clear_metrics('test')

        step = step + 1

        train_writer.flush()
        validation_writer.flush()
        test_writer.flush()
            
    torch.save(state,path + save_directory + '.pth')
