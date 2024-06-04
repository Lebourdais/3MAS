from typing import Dict, Optional

import torch
from torch import nn, Tensor
from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from src.utils.feat import SpecFeat
from .wavlm import WavLM_Feats
import torchaudio

from .emb_transformations import get_emb_transformation


class ModelNMF(Model):

    def __init__(
        self,
        teacher: torch.nn.Module=None,
        w_nmf: Tensor=None,
        spec_kw: Dict={"win_length":400,"hop_length":320,"n_fft":512,"center":False},
        emb_transform: Dict={'ttype':'tcn'},
        input_spec : bool = False,
        sample_rate: int=16000,
        num_channels: int=1,
        n_classes: int=4,
        task: Optional[Task]=None,
        return_embed: bool=False,
        non_negative_output: str=None,
        train_nmf: bool=False,
    ):
        super().__init__(
            sample_rate=sample_rate, num_channels=num_channels, task=task
        )
        self.teacher = teacher
        #self.w_nmf = w_nmf
        self.emb_transform_hparam = emb_transform
        self.return_embed = return_embed
        self.spec_kw = spec_kw
        self.input_spec = input_spec
        self.n_classes = n_classes
        self.non_negative_act = non_negative_output
        self.train_nmf = train_nmf
        self.nmf_rank = w_nmf.shape[-1]
        
        if train_nmf:
            self.w_nmf = torch.nn.Linear(self.nmf_rank,spec_kw["n_fft"]//2+1,bias=False)
        else:
            self.register_buffer("w_nmf",w_nmf)
        print(f"Type of NMF matrix: {type(self.w_nmf)}")
        self.spec_feat = SpecFeat(**self.spec_kw)
        
    def build(self):
        # either the student takes intermediate representation of the teacher of a spectrogram as input
        #in_chan = self.spec_kw["n_fft"]//2+1 if self.input_spec else self.teacher.tcn.bn_chan
        in_chan = self.spec_kw["n_fft"]//2+1 if self.input_spec else self.teacher.tcn.in_chan
        self.emb_transform = get_emb_transformation(
            **self.emb_transform_hparam, 
            in_chan=in_chan, 
            out_chan=self.nmf_rank,
        )
        #out = self.teacher.tcn.out_chan if self.teacher is not None else self.n_classes
        out = self.n_classes
        self.class_layer = nn.Linear(
            self.nmf_rank,out 
        )

    def forward(self, x: Tensor) -> Tensor:

        # extract intermediate representation and predictions from teacher model
        if self.teacher is not None:
            with torch.no_grad():
                teacher_out = self.teacher(x)

        target_spec = self.spec_feat(x)
        if self.input_spec:
            embeds = self.emb_transform(target_spec)
            if self.teacher is not None:
                teacher_logits = teacher_out['out']
            else:
                teacher_logits = None
        else:
            # apply transformations to the intermediate representation to get H matrix (embeddings) and student logits
            teacher_logits = teacher_out['out']
            teacher_embeds = teacher_out["feat"]
            embeds = self.emb_transform(teacher_embeds)
    
        if self.non_negative_act == "relu": 
            embeds = torch.nn.functional.relu(embeds)
        elif self.non_negative_act == "gelu": 
            embeds = torch.nn.functional.gelu(embeds)
        elif self.non_negative_act == "softplus": 
            embeds = torch.nn.functional.softplus(embeds)
        
        logits = self.class_layer(embeds.permute(0,2,1))

        # reconstruct spectrogram with NMF for each batch
        if self.train_nmf:
            reconstruct_spec = self.w_nmf(embeds.permute(0,2,1))
            reconstruct_spec = reconstruct_spec.permute(0,2,1)
        else:
            reconstruct_spec = torch.bmm(self.w_nmf.repeat(embeds.shape[0],1,1),embeds) #self.w_nmf.matmul(embeds)

        if self.return_embed:
            return {
                'teacher_logits': teacher_logits,
                'student_embeds': embeds,
                'student_logits': logits,
                'target_spec': target_spec,
                'reconstruct_spec': reconstruct_spec
            }
        else:
            return logits

    def save_student(self,name="none"):
        torch.save(self.emb_transform,f"{name}_transform.pt")
        torch.save(self.class_layer,f"{name}_classif.pt")
        torch.save(self.w_nmf,f"{name}_nmf_dict.pt")

class MASNMF(Model):

    def __init__(
        self,
        wavlm_cfg: Dict=None,
        w_nmf: Tensor=None,
        spec_kw: Dict={"win_length":400,"hop_length":320,"n_fft":512,"center":False},
        emb_transform: Dict={'ttype':'tcn'},
        input_spec : bool = False,
        sample_rate: int=16000,
        num_channels: int=1,
        n_classes: int=4,
        task: Optional[Task]=None,
        return_embed: bool=False,
        non_negative_output: str=None,
        train_nmf: bool=False,
        teacher=None
    ):
        super().__init__(
            sample_rate=sample_rate, num_channels=num_channels, task=task
        )
        self.emb_transform_hparam = emb_transform
        self.return_embed = return_embed
        self.spec_kw = spec_kw
        self.input_spec = input_spec
        self.n_classes = n_classes
        self.non_negative_act = non_negative_output
        self.train_nmf = train_nmf
        self.nmf_rank = w_nmf.shape[-1]
        self.in_chan=1024
        
        if train_nmf:
            self.w_nmf = torch.nn.Linear(self.nmf_rank,spec_kw["n_fft"]//2+1,bias=False)
        else:
            self.register_buffer("w_nmf",w_nmf)
        print(f"Type of NMF matrix: {type(self.w_nmf)} {self.w_nmf.shape}")
        self.spec_feat = SpecFeat(**self.spec_kw)

        if not self.input_spec:
            self.wavlm=WavLM_Feats(**wavlm_cfg)
        
    def build(self):
        # either the student takes intermediate representation of the teacher of a spectrogram as input
        #in_chan = self.spec_kw["n_fft"]//2+1 if self.input_spec else self.teacher.tcn.bn_chan
        in_chan = self.spec_kw["n_fft"]//2+1 if self.input_spec else self.in_chan
        self.emb_transform = get_emb_transformation(
            **self.emb_transform_hparam, 
            in_chan=in_chan, 
            out_chan=self.nmf_rank,
        )
        #out = self.teacher.tcn.out_chan if self.teacher is not None else self.n_classes
        out = self.n_classes
        self.class_layer = nn.Linear(
            self.nmf_rank,out 
        )

    def forward(self, x: Tensor) -> Tensor:

        # extract intermediate representation and predictions from teacher model
        target_spec = self.spec_feat(x)
        if self.input_spec:
            embeds = self.emb_transform(target_spec)
        else:
            wavlm_feat = self.wavlm(x)
            embeds = self.emb_transform(wavlm_feat)
    
        if self.non_negative_act == "relu": 
            embeds = torch.nn.functional.relu(embeds)
        elif self.non_negative_act == "gelu": 
            embeds = torch.nn.functional.gelu(embeds)
        elif self.non_negative_act == "softplus": 
            embeds = torch.nn.functional.softplus(embeds)
        
        logits = self.class_layer(embeds.permute(0,2,1))

        # reconstruct spectrogram with NMF for each batch
        if self.train_nmf:
            reconstruct_spec = self.w_nmf(embeds.permute(0,2,1))
            reconstruct_spec = reconstruct_spec.permute(0,2,1)
        else:
            reconstruct_spec = torch.bmm(self.w_nmf.repeat(embeds.shape[0],1,1),embeds) #self.w_nmf.matmul(embeds)

        if self.return_embed:
            return {
                'teacher_logits': None,
                'student_embeds': embeds,
                'student_logits': logits,
                'target_spec': target_spec,
                'reconstruct_spec': reconstruct_spec
            }
        else:
            return logits

    def save_student(self,name="none"):
        torch.save(self.emb_transform,f"{name}_transform.pt")
        torch.save(self.class_layer,f"{name}_classif.pt")
        torch.save(self.w_nmf,f"{name}_nmf_dict.pt")


class ModelWavlmNMF(ModelNMF):

    def __init__(
        self,
        teacher: torch.nn.Module=None,
        w_nmf: Tensor=None,
        emb_transform: Dict={'ttype':'tcn'},
        sample_rate: int=16000,
        num_channels: int=1,
        n_classes: int=3,
        task: Optional[Task]=None,
        return_embed: bool=False,
    ):
        super().__init__(
            sample_rate=sample_rate, num_channels=num_channels, task=task
        )
        self.teacher = teacher
        self.emb_transform_hparam = emb_transform
        self.return_embed = return_embed
        self.input_spec = input_spec
        self.n_classes = n_classes

        self.register_buffer("w_nmf",w_nmf)
        self.spec_feat = SpecFeat(**self.spec_kw)
        
    def build(self):
        # either the student takes intermediate representation of the teacher of a spectrogram as input
        #TODO Check size for wavlm inputs
        in_chan = self.spec_kw["n_fft"]//2+1 if self.input_spec else self.teacher.tcn.in_chan
        self.emb_transform = get_emb_transformation(
            **self.emb_transform_hparam, 
            in_chan=in_chan, 
            out_chan=self.w_nmf.shape[-1],
        )
        #out = self.teacher.tcn.out_chan if self.teacher is not None else self.n_classes
        out = self.n_classes
        self.class_layer = nn.Linear(
            self.w_nmf.shape[-1],out 
        )


    def forward(self, x: Tensor) -> Tensor:

        # extract intermediate representation and predictions from teacher model
        if self.teacher is not None:
            with torch.no_grad():
                teacher_out = self.teacher(x)

        target_spec = self.spec_feat(x)
        if self.input_spec:
            embeds = self.emb_transform(target_spec)
            if self.teacher is not None:
                teacher_logits = teacher_out['out']
            else:
                teacher_logits = None
        else:
            # apply transformations to the intermediate representation to get H matrix (embeddings) and student logits
            teacher_logits = teacher_out['out']
            #TODO modify teacher model to output wavlm features
            teacher_embeds = teacher_out["feat"]
            embeds = self.emb_transform(teacher_embeds)
        logits = self.class_layer(embeds.permute(0,2,1))

        # reconstruct spectrogram with NMF for each batch
        reconstruct_spec = torch.bmm(self.w_nmf.repeat(embeds.shape[0],1,1),embeds) #self.w_nmf.matmul(embeds)

        if self.return_embed:
            return {
                'teacher_logits': teacher_logits,
                'student_embeds': embeds,
                'student_logits': logits,
                'target_spec': target_spec,
                'reconstruct_spec': reconstruct_spec
            }
        else:
            return logits

    def save_parts(self,path="./"):
        torch.save(self.emb_transform,f"{path}transform.pt")
        torch.save(self.class_layer,f"{path}classif.pt")
        torch.save(self.w_nmf,f"{path}nmf_dict.pt")
        

        
