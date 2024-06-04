import matplotlib.pyplot as plt
import numpy
import torch
plt.rc('font', size=14)

def plot_spectro(ax,x,idx,label,vmin=0,vmax=None,ymax=None):
    fs=16000
    vmin=vmin
    x = x.detach().cpu().numpy().squeeze()
    freqs = numpy.linspace(0,fs//2+1,x.shape[0])
    tt = numpy.linspace(0,x.shape[-1],x.shape[-1])
    #ax = fig.add_subplot(1,1,1)
    xn = (x-x.min())/(x.max()-x.min())
    vmax=xn.max() if vmax is None else vmax
    im=ax.pcolormesh(tt,freqs,xn,linewidth=0,rasterized=True,vmin=vmin,vmax=vmax,cmap="jet")
    plt.colorbar(im,ax=ax)
    ax.set_title(f'Explaination class {label} {idx:02d}')
    if ymax is not None:
        ax.set_ylim((0,ymax))


def plot_reconstruct(x,x_hat,idx,label):
    fig = plt.figure(figsize=(10,5))    
    fs=16000
    vmin=0
    x = x.detach().cpu().numpy().squeeze()
    x_hat = x_hat.detach().cpu().numpy().squeeze()
    vmax=numpy.array([x.max(),x_hat.max()]).max()
    freqs = numpy.linspace(0,fs//2+1,x.shape[0])
    tt = numpy.linspace(0,x.shape[-1],x.shape[-1])
    ax = fig.add_subplot(2,1,1)
    im=ax.pcolormesh(tt,freqs,x,linewidth=0,rasterized=True,vmin=vmin,vmax=vmax)
    plt.colorbar(im,ax=ax)
    ax.set_title(f'Target {label} {idx:02d}')
    ax = fig.add_subplot(2,1,2)
    tt = numpy.linspace(0,x_hat.shape[-1],x_hat.shape[-1])
    im=ax.pcolormesh(tt,freqs,x_hat,linewidth=0,rasterized=True,vmin=vmin,vmax=vmax)
    plt.colorbar(im,ax=ax)
    ax.set_title(f'Reconstruction {label} {idx:02d}')
    fig.tight_layout()
    
def plot_nmf(W,H,idx,label):
    fig = plt.figure(figsize=(10,3))
    fs=16000
    vmin=0
    H = H.detach().cpu().numpy().squeeze()
    W = W.detach().cpu().numpy().squeeze()
    act_idx = numpy.linspace(0,H.shape[0],H.shape[0])
    tt = numpy.linspace(0,H.shape[-1],H.shape[-1])
    ff = numpy.linspace(0,8000,W.shape[0])
    ax = fig.add_subplot(1,2,1)
    vmax=W.max()
    im=ax.pcolormesh(act_idx,ff,W,linewidth=0,rasterized=True,vmin=vmin,vmax=vmax)
    plt.colorbar(im,ax=ax)
    ax.set_title(f'Components $W$ {label} {idx:02d}')
    ax = fig.add_subplot(1,2,2)
    vmax=H.max()
    im=ax.pcolormesh(tt,act_idx,H,linewidth=0,rasterized=True,vmin=vmin,vmax=vmax)
    plt.colorbar(im,ax=ax)
    ax.set_title(f'Activations $H$ {label} {idx:02d}')
    fig.tight_layout()    
    
def plot_logits(logits,idx,label,thres=0.0,fig=None,ax=None,**kwargs):
    if fig is None:
        fig = plt.figure(figsize=(10,3))
    vmin=0
    if isinstance(logits,torch.Tensor): 
        logits = logits.detach().cpu().numpy().squeeze()
    vmax=logits.max()
    act_idx = numpy.linspace(0,logits.shape[0],logits.shape[0])
    classes = ["sp", "no", "mu", "ov"] #numpy.arange(0,4,1).astype(int)
    if ax is None:
        ax = fig.add_subplot(1,1,1)
    for c in range(len(classes)):
        im=ax.plot(act_idx,logits[:,c],label=classes[c],linewidth=2)
        #plt.colorbar(im,ax=ax)
    ax.legend()
    ax.grid()
    ax.set_xlim([0,act_idx[-1]])
    ax.set_title(kwargs.get("title",f'Logits {label} {idx:02d}'))