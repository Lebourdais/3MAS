import torchaudio
import torch

class SpecFeat(torch.nn.Module):
    def __init__(self,**kwargs):
        super(SpecFeat,self).__init__()
        kwargs["center"]=False
        kwargs["pad"]=int(kwargs["win_length"]//2-kwargs["hop_length"])
        self.spec = torchaudio.transforms.Spectrogram(**kwargs)
        
    def forward(self,x):
        if x.ndim > 2:
            x = x.squeeze(1)
        X = self.spec(x)
        X = torch.log(1 + X)
        #X = (X - X.min())/(X.max()-X.min())
        
        return X

class MelSpecFeat(torch.nn.Module):

    def __init__(self,**kwargs):
        super(MelSpecFeat,self).__init__()
        kwargs["center"]=False
        kwargs["pad"]=int(kwargs["win_length"]//2-kwargs["hop_length"])
        self.mel_spec = torchaudio.transforms.MelSpectrogram(**kwargs)

    def forward(self,x):
        if x.ndim > 2:
            x = x.squeeze(1)
        self.mel_spec(x)
        X = torch.log(1 + X)
        X = (X - X.min())/(X.max()-X.min())
        return X

         
