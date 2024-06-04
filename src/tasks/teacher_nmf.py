from typing import Dict, List, Optional, Sequence, Text, Tuple, Union

import torch
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric
from pyannote.audio.core.model import Model
from pyannote.database import Protocol
from pyannote.audio.tasks.segmentation.multilabel import MultiLabelSegmentation

from .losses import *


class MultiLabelSegmentationTeacherNMF(MultiLabelSegmentation):
    """Multi-label segmentation for NMF and Knowledge Distillation

    Parameters
    ----------
    teacher_model : pyannote.audio.core.model.Model
        pyannote teacher model
    w_nmf : torch.Tensor
        Non-negative matrix to reconstruct new embeddings
    """

    def __init__(
        self,
        protocol: Protocol,
        kwargs_class: Dict={'loss_type':'kd'},
        kwargs_recons: Dict={'loss_type':'mse'},
        kwargs_sparse: Dict={'loss_type':'l1'},
        beta_class: float=1.,
        beta_recons: float=1.,
        beta_sparse: float=0.,
        classes: Optional[List[str]] = None,
        duration: float = 2.0,
        warm_up: Union[float, Tuple[float, float]] = 0.0,
        balance: Text = None,
        weight: Text = None,
        batch_size: int = 32,
        num_workers: int = None,
        pin_memory: bool = False,
        augmentation: BaseWaveformTransform = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
        lr: float = 1e-3,
        scheduler_kw: dict = None,
    ):

        super().__init__(
            protocol,
            classes=classes,
            duration=duration,
            warm_up=warm_up,
            balance=balance,
            weight=weight,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            augmentation=augmentation,
            metric=metric,
        )

        #self.feat_extract = SpecFeat(**kwargs_spec)
        
        self.class_loss = get_class_loss(**kwargs_class)
        #self.recons_loss = get_recons_loss(**kwargs_recons)
        self.sparse_loss = get_sparse_loss(**kwargs_sparse)
        self.recons_loss=torch.nn.MSELoss()

        self.beta_class = beta_class
        self.beta_recons = beta_recons
        self.beta_sparse = beta_sparse

    def training_step(self, batch, batch_idx: int):

        x = batch["X"]
        labels = batch["y"]

        #spec = self.feat_extract(x)
        mask: torch.Tensor = labels != -1
        
        output = self.model(x)
        #pseudo_spec = self.model.w_nmf @ output['student_embeds'] #TO-DO: Review multiple matrix product

        labels = labels[mask]
        teacher_logits = output['teacher_logits'] #[mask]
        if teacher_logits is not None:
            student_logits = output['student_logits'] #[mask]
        else:
            student_logits = output['student_logits'][mask]

        spec = output["target_spec"]
        pseudo_spec = output["reconstruct_spec"]
        if not "ov" in self.classes:
            teacher_logits=teacher_logits[:,:,:3]
            
        class_loss = self.class_loss(student_logits, teacher_logits, labels)
        recons_loss = self.recons_loss(spec, pseudo_spec)
        sparse_loss = self.sparse_loss(output['student_embeds'])
        total_loss = self.beta_class*class_loss + \
            self.beta_recons*recons_loss+ self.beta_sparse*sparse_loss

        # skip batch if something went wrong for some reason
        if torch.isnan(total_loss):
            return None

        self.model.log("loss/train/class_loss",class_loss,on_step=True,on_epoch=False,prog_bar=False,logger=True,)
        self.model.log("loss/train/recons_loss",recons_loss,on_step=True,on_epoch=False,prog_bar=False,logger=True,)
        self.model.log("loss/train/sparse_loss",sparse_loss,on_step=True,on_epoch=False,prog_bar=False,logger=True,)
        self.model.log("loss/train/total_loss",total_loss,on_step=True,on_epoch=False,prog_bar=False,logger=True,)
        return {"loss": total_loss}


    def validation_step(self, batch, batch_idx: int):
        x = batch["X"]
        labels = batch["y"]

        #spec = self.feat_extract(x)
        mask: torch.Tensor = labels != -1
        
        output = self.model(x)
        #pseudo_spec = self.model.w_nmf @ output['student_embeds'] #TO-DO: Review multiple matrix product

        labels = labels[mask]
        #student_logits = output['student_logits'] #[mask]
        teacher_logits = output['teacher_logits'] #[mask]
        if teacher_logits is not None:
            student_logits = output['student_logits'] #[mask]
        else:
            student_logits = output['student_logits'][mask]

        spec = output["target_spec"]
        pseudo_spec = output["reconstruct_spec"]
        if not "ov" in self.classes:
            teacher_logits=teacher_logits[:,:,:3]

        #spec = output["target_spec"]
        #pseudo_spec = output["reconstruct_spec"]
    
        class_loss = self.class_loss(student_logits, teacher_logits, labels)
        recons_loss = self.recons_loss(spec, pseudo_spec)
        sparse_loss = self.sparse_loss(output['student_embeds'])
        total_loss = self.beta_class*class_loss + \
            self.beta_recons*recons_loss+ self.beta_sparse*sparse_loss

        self.model.log("loss/val/class_loss",class_loss,on_step=False,on_epoch=True,prog_bar=False,logger=True,)
        self.model.log("loss/val/recons_loss",recons_loss,on_step=False,on_epoch=True,prog_bar=False,logger=True,)
        self.model.log("loss/val/sparse_loss",sparse_loss,on_step=False,on_epoch=True,prog_bar=False,logger=True,)
        self.model.log("loss/val/total_loss",total_loss,on_step=False,on_epoch=True,prog_bar=False,logger=True,)

        return {"loss": total_loss}

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        if self.scheduler_kw is not None:
            scheduler = troch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_kw)
        return [optimizer], [scheduler]
