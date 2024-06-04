import torch
import torchaudio
import numpy
import matplotlib.pyplot as plt

def get_relevant_components(H,
                            classif_weights,
                            thres=0.0,
                            reduction=None,
                            class_index=0,
                            lims=(-1,1),
                            doplot=False):
    H = H.squeeze()
    K = H.shape[0]
    #r_kc = torch.zeros((K))
    if reduction is None:
        r_kc = torch.zeros_like(H)
        for k in range(K):
            #r_kc[k] = z[k] * classif_weights[class_index,k]
            r_kc[k,:] = H[k,:] * classif_weights[class_index,k]
        r_kc = r_kc/torch.max(r_kc)
        mask = r_kc > thres
        idx_relevant = (r_kc > thres).nonzero(as_tuple=True)
    else:
        if reduction == "sum":
            z = H.sum(dim=-1)
        elif reduction == "mean":
            z = H.mean(dim=-1)
        elif reduction == "max":
            z = H.max(dim=-1)
        r_kc = torch.zeros((K))
        for k in range(K):
            r_kc[k] = z[k] * classif_weights[class_index,k]
        r_kc/=r_kc.max()
        #r_kc = (r_kc - r_kc.mean())/r_kc.std()        
        #print(r_kc.shape)
        #r_kc = torch.nn.functional.sigmoid(r_kc)
        #print(f"Statistics of r_kc:\n\t Mean: {r_kc.mean()}\n\t Std: {r_kc.std()}\n\t Min: {r_kc.min()}\n\t Max: {r_kc.max()}")
        mask = r_kc > thres
        idx_relevant = (r_kc > thres).nonzero(as_tuple=False)

    
    return r_kc, mask, idx_relevant

#TODO: plot W by masking useless components
def get_masks_from_indexes(relevant_idx,W,H):
    masks = {}
    W = W.squeeze()
    H = H.squeeze()
    norm_ = W.matmul(H)
    n_idx = relevant_idx.shape[0]
    all_k = numpy.arange(W.shape[-1]).astype(int)
    mask_k = numpy.delete(all_k,relevant_idx.numpy())
    
    W_mask = (W-W.min())/(W.max()-W.min())
    W_mask[:,mask_k]=0
    
    for i in range(n_idx):
        k = relevant_idx[i].numpy()[0]
        key = f"comp_{k}"
        masks[key] = W[:,k].unsqueeze(1).matmul(H[k,:].unsqueeze(0)) #/norm_
        #print(f"Statistics of {key}:\n\t Mean: {masks[key].mean()}\n\t Std: {masks[key].std()}\n\t Min: {masks[key].min()}\n\t Max: {masks[key].max()}")
    return masks, W_mask

def get_audio(X_stft,X_mag):
    X_phi = X_stft.angle()
    x_audio = torch.istft(X_mag*torch.exp(1j*X_phi),win_length=1024,hop_length=320,n_fft=1024)
    return x_audio

def generate_exlpaination(X_stft,soft_mask):
    X_mag = X_stft.abs()
    X_recons = soft_mask * X_mag
    x_audio = get_audio(X_stft,X_recons)
    return X_recons, x_audio

from tqdm import tqdm
def compute_relevance_set(audio_set,class_layer,emb_transform=None,full_model=None,w_nmf=None,teacher=None, target_class=None, cmp_thres=0.5, emb_lower_bound=1e-8):
    audio_select_cmp=numpy.array([])
    embed_zero_rate=[]
    classes = ["sp","no","mu","ov"]
    reduction="mean"
    weights = class_layer.state_dict()['weight']
    r_kc_acc = numpy.zeros((weights.shape[-1]))
    for idx, s in tqdm(enumerate(audio_set), desc="Working..."):
        if full_model is None:
            H, X_stft=forward(x=s,emb_transform=emb_transform,teacher=teacher)
        else:
            H, X_stft=forward_3mas(x=s,model=full_model,teacher=None)
        # zero-values rate
        embed_zero_rate.append((H<emb_lower_bound).sum()/H.numel())

        r_kc,mask,idx_relevant = get_relevant_components(H,weights,thres=cmp_thres,class_index=classes.index(target_class),lims=(0,1),doplot=False,reduction=reduction)
        r_kc_acc+=r_kc.detach().cpu().numpy()
        masks, w_mask = get_masks_from_indexes(idx_relevant,w_nmf,H)

        audio_select_cmp = numpy.append(audio_select_cmp,idx_relevant.detach().cpu().numpy()) #speech_select_cmp.append(idx_relevant.detach().cpu().numpy())
    r_kc_acc/=len(audio_set)
    avg_zero_rate = numpy.array(embed_zero_rate).mean()
    
    return audio_select_cmp, r_kc_acc, avg_zero_rate

def forward(x,emb_transform,teacher=None,device="cpu"):
    X_stft = torch.stft(x,n_fft=1024, hop_length=320, win_length=1024, return_complex=True)
    X_stft = X_stft[:,:,:199]
    X_src = torch.log(1+X_stft.abs()**2)
    x=x.unsqueeze(0)
    if teacher is not None:
        teacher_out = teacher(x)
        feat = teacher_out["feat"]
    else:
        feat = X_src
    H = emb_transform(feat)
    H = torch.nn.functional.relu(H)
    
    return H, X_stft


def forward_3mas(x,model,teacher=None,device="cpu"):
    X_stft = torch.stft(x,n_fft=1024, hop_length=320, win_length=1024, return_complex=True)
    X_stft = X_stft[:,:,:199]
    X_src = torch.log(1+X_stft.abs()**2)
    x=x.unsqueeze(0)
    #if teacher is not None:
    #    teacher_out = teacher(x)
    #    feat = teacher_out["feat"]
    #else:
    #    feat = X_src
    feat = model.wavlm(x)
    H = model.emb_transform(feat)
    #H = torch.nn.functional.relu(H)
    
    return H, X_stft