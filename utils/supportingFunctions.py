import torch
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.metrics import normalized_root_mse as nrmse

def test_network(model,data_loader,lossfunction=None,mode=None):
    model.eval()

    with torch.no_grad():
        outputs = []
        targets = []
        masks = [];
        for input,target,signal,TI,mask in data_loader:

            input = input.cuda()
            target = target.cuda()
            signal = signal.cuda()
            TI = TI.cuda()
            mask = mask.cuda()
                        
            output,(hn,cn) = model(input)

            if lossfunction is not None:
                loss = lossfunction(output,target,signal,TI,mode)

            outputs.append(output.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            masks.append(mask.detach().cpu().numpy())

        outputs = np.concatenate(outputs,0)
        targets = np.concatenate(targets,0)
        masks = np.concatenate(masks,0)
            
    return outputs,targets,masks

def check_orientation(x):
    x = np.transpose(x[2,:,:,:],(1,2,0))

    x = np.sum(x,axis=-1)
    
    o1 = x
    o2 = np.rot90(x,1)
    o3 = np.rot90(x,2)
    o4 = np.rot90(x,3)
    o5 = np.flipud(o1)
    o6 = np.flipud(o2)
    o7 = np.flipud(o3)
    o8 = np.flipud(o4)

    fig,((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8)) = plt.subplots(nrows=2,ncols=4)

    ax1.imshow(o1,cmap='gray')
    ax1.set_title(1)

    ax2.imshow(o2,cmap='gray')
    ax2.set_title(2)

    ax3.imshow(o3,cmap='gray')
    ax3.set_title(3)

    ax4.imshow(o4,cmap='gray')
    ax4.set_title(4)

    ax5.imshow(o5,cmap='gray')
    ax5.set_title(5)

    ax6.imshow(o6,cmap='gray')
    ax6.set_title(6)

    ax7.imshow(o7,cmap='gray')
    ax7.set_title(7)

    ax8.imshow(o8,cmap='gray')
    ax8.set_title(8)
    
    plt.draw()
    plt.pause(1)

    correct = input('select correct orientation ')
    
    plt.close(fig)

    return correct

def orientate(x,orientation):

    x = np.transpose(x,(2,3,0,1))

    if orientation == 2:
        x = np.rot90(x,1)
    if orientation == 3:
        x = np.rot90(x,2)
    if orientation == 4:
        x = np.rot90(x,3)
    if orientation == 5:
        x = np.flipud(x)
    if orientation == 6:
        x = np.flipud(np.rot90(x,1))
    if orientation == 7:
        x = np.flipud(np.rot90(x,2))
    if orientation == 8:
        x = np.flipud(np.rot90(x,3))

    x = np.transpose(x,(2,3,0,1))

    return x

def tb_add_scalar(writer,step,loss_dict):

    for key,val in loss_dict.items():
        writer.add_scalar(key,val,step)

def tb_add_histogram(writer,step,weight_dict):

    for key,val in weight_dict.items():
        writer.add_histogram(key,val,step)

def T1segment(outputs,targets,masks,sets=None):
    N = 128

    nb = len(targets)

    outputs = np.abs(np.reshape(outputs,[nb//(N**2),N,N]))
    targets = np.abs(np.reshape(targets,[nb//(N**2),N,N]))
    masks = np.reshape(masks,[nb//(N**2),N,N])

    if sets is not None:
        outputs = outputs[sets,...]
        targets = targets[sets,...]
        masks = masks[sets,...]

    nb,sx,sy = targets.shape

    count = 0
    T1 = np.zeros((2,int(np.sum(np.amax(masks,axis=(2,1))))))
    for i in range(nb):
        for j in range(int(np.max(masks[i,:,:]))):
            targetsT1 = (targets[i,:,:] * (masks[i,:,:] == j+1))
            targetsT1 = np.mean(targetsT1[targetsT1 != 0])
            T1[0,count] = targetsT1

            outputsT1 = (outputs[i,:,:] * (masks[i,:,:] == j+1))
            outputsT1 = np.mean(outputsT1[outputsT1 != 0])
            T1[1,count] = outputsT1

            count = count + 1

    return T1
            
def T1metrics(writer,step,T1,save_name='',dpi=100):
    
    m,b = np.polyfit(T1[0,:],T1[1,:],1)

    rcoef = np.corrcoef(T1[0,:],T1[1,:])
    rval = rcoef[0,1]**2

    dataDiff = T1[1,:] - T1[0,:]
    dataMean = np.mean(np.stack((T1[1,:],T1[0,:]),0),0)

    bias = np.mean(dataDiff)
    sd = np.std(dataDiff)
    lB = bias - 1.96*sd
    uB = bias + 1.96*sd

    nrmse_val = nrmse(T1[0,:],T1[1,:])

    vals = np.linspace(np.min(T1[0,:])-50,np.max(T1[0,:])+50,100)

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

    ax1.scatter(T1[0,:],T1[1,:])
    ax1.plot(vals,vals,'k--',alpha=0.5)
    ax1.plot(vals,m*vals+b,'k')
    ax1.set_xlabel('Reference (ms)')
    ax1.set_ylabel('T1NET (ms)')
    ax1.set_xlim([np.min(T1[0,:])-50,np.max(T1[0,:])+50])
    ax1.set_ylim([np.min(T1[1,:])-50,np.max(T1[1,:])+50])
    if b > 0:
        ax1.text(0.05,0.95,'y = %.4fx + %.4f' %(m,b),transform=ax1.transAxes)
    else:
        ax1.text(0.05,0.95,'y = %.4fx - %.4f' %(m,-b),transform=ax1.transAxes)
    ax1.text(0.05,0.90,'R^2 = %.4f' %rval,transform=ax1.transAxes)
    ax1.text(0.05,0.85,'NRMSE = %.4f' %nrmse_val,transform=ax1.transAxes)

    ax2.scatter(dataMean,dataDiff)
    ax2.plot(bias*np.ones(int(np.max(dataMean))),'k',label='bias')
    ax2.plot(lB*np.ones(int(np.max(dataMean))),'k--',label='95% confidence limit')
    ax2.plot(uB*np.ones(int(np.max(dataMean))),'k--')
    ax2.set_xlabel('Mean')
    ax2.set_ylabel('Difference')
    ax2.set_xlim([np.min(dataMean),np.max(dataMean)])
    plt.legend()

    if writer is not None:
        writer.add_figure('T1 estimation',fig,step)
        plt.close()
    else:
        plt.savefig(save_name,dpi=dpi)

def T1plot(outputs,targets,save_name,clip,sets=None,cmap='inferno',dpi=100):
    N = 128

    nb = len(targets)

    outputs = np.abs(np.reshape(outputs,[nb//(N**2),N,N])) + 1e-10
    targets = np.abs(np.reshape(targets,[nb//(N**2),N,N])) + 1e-10
        
    diff = np.abs(targets - outputs)

    outputs[outputs > clip] = clip
    targets[targets > clip] = clip

    if sets is not None:
        outputs = outputs[sets,...]
        targets = targets[sets,...]

    fig = plt.figure(figsize=(9,5))
    grid = ImageGrid(fig,111,nrows_ncols=(3,3),axes_pad=0,direction='row')
    
    count = 0
    labels = ['Apex','Mid','Base']

    for i in range(3):
        grid[count].imshow(outputs[i,:,:],cmap=cmap)
        grid[count].set_xticks([])
        grid[count].set_yticks([])
        grid[count].set_title(labels[i])
        if i == 0:
            grid[count].set_ylabel('T1NET')
        count = count + 1

    for i in range(3):
        cb = grid[count].imshow(targets[i,:,:],cmap=cmap)
        grid[count].set_xticks([])
        grid[count].set_yticks([])
        if i == 0:
            grid[count].set_ylabel('Reference')
        count = count + 1

    for i in range(3):
        grid[count].imshow(diff[i,:,:],cmap=cmap)
        grid[count].set_xticks([])
        grid[count].set_yticks([])
        if i == 0:
            grid[count].set_ylabel('Difference')
        count = count + 1

    cbaxes = fig.add_axes([0.75, 0.1, 0.03, 0.8])
    
    plt.colorbar(cb,cax=cbaxes)

    plt.savefig(save_name,bbox_inches='tight',dpi=dpi)

class grad_flow():
    def __init__(self,model):

        self.grad_tracker = {'layers':[],'mean':[],'max':[],'min':[]}

        for name,param in model.named_parameters():
            if param.requires_grad and 'bias' not in name:
                self.grad_tracker['layers'].append(name)

    def calculate_metrics(self,model):
        
        for name,param in model.named_parameters():
            if param.requires_grad and 'bias' not in name:
                grad = param.grad.detach().cpu().numpy()
                
                self.grad_tracker['mean'].append(np.mean(np.abs(grad)))
                self.grad_tracker['max'].append(np.max(grad))
                self.grad_tracker['min'].append(np.min(grad))

    def return_metrics(self):
        tracker = self.grad_tracker.copy()

        tracker = {key:np.transpose(np.reshape(val,[len(tracker[key])//len(tracker['layers']),len(tracker['layers'])]),[1,0]) for key,val in tracker.items()}

        return tracker

    def clear_metrics(self):
        for key,val in self.grad_tracker.items():
            if 'layers' not in key:
                self.grad_tracker[key] = []

def plot_grad_flow(writer,step,grad_tracker):

    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)

    ax1.plot(grad_tracker['max'], alpha=0.3, color="g")
    ax1.plot(grad_tracker['min'], alpha=0.3, color="b")
    ax1.hlines(0, 0, len(grad_tracker['max'])+1, linewidth=1, color="k" )
    ax1.set_xticks(range(0,len(grad_tracker['max']), 1))
    ax1.set_xticklabels(grad_tracker['layers'], rotation=90)
    ax1.set_xlim((0,len(grad_tracker['max'])))
    ax1.set_xlabel("Layers")
    ax1.set_ylabel("Gradients")
    ax1.set_title("Gradient Flow")
    ax1.grid(True)

    ax2.plot(grad_tracker['mean'], alpha=0.3, color="r")
    ax2.hlines(0, 0, len(grad_tracker['mean'])+1, linewidth=1, color="k" )
    ax2.set_xticks(range(0,len(grad_tracker['mean']), 1))
    ax2.set_xticklabels(grad_tracker['layers'], rotation=90)
    ax2.set_xlim((0,len(grad_tracker['mean'])))
    ax2.set_xlabel("Layers")
    ax2.set_ylabel("Gradients")
    ax2.set_title("Gradient Flow")
    ax2.grid(True)

    writer.add_figure('Gradient Flow',plt.gcf(),step)

    plt.close()

