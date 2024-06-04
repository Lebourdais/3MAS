from typing import Dict, List, Optional, Sequence, Text, Tuple, Union

import torch
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torchmetrics import Metric
from pyannote.audio.core.model import Model
from pyannote.database import Protocol
from pyannote.audio.tasks.segmentation.multilabel import MultiLabelSegmentation

from .losses import *
import torch.nn.functional as F


class MultiLabelSegmentationNMF(MultiLabelSegmentation):

    def __init__(
        self,
        protocol: Protocol,
        kwargs_class: Dict={'loss_type':'ce'},
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
        lr: float = 1e-3,
        scheduler_kw: dict = None,
        augmentation: BaseWaveformTransform = None,
        metric: Union[Metric, Sequence[Metric], Dict[str, Metric]] = None,
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

        #self.class_loss = get_class_loss(**kwargs_class)
        #self.recons_loss = get_recons_loss(**kwargs_recons)
        self.sparse_loss = get_sparse_loss(**kwargs_sparse)

        self.beta_class = beta_class
        self.beta_recons = beta_recons
        self.beta_sparse = beta_sparse
        self.lr = lr
        self.scheduler_kw = scheduler_kw
        
        self.recons_loss=torch.nn.MSELoss(reduction="mean")

    def training_step(self, batch, batch_idx: int):

        x = batch["X"]
        labels = batch["y"]

        mask: torch.Tensor = labels != -1
        
        output = self.model(x)

        labels = labels[mask]
        logits = output['logits'][mask]
        class_loss = F.binary_cross_entropy(logits, labels.type(torch.float))

        feat = output["target_feat"]
        pseudo_feat = output["reconstruct_feat"]
    
        recons_loss = self.recons_loss(feat, pseudo_feat)
        sparse_loss = self.sparse_loss(output['embeds'])
        total_loss = self.beta_class*class_loss + \
            self.beta_recons*recons_loss + self.beta_sparse*sparse_loss

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

        mask: torch.Tensor = labels != -1
        
        output = self.model(x)

        labels = labels[mask]
        logits = output['logits'][mask]
        class_loss = F.binary_cross_entropy(logits, labels.type(torch.float))

        feat = output["target_feat"]
        pseudo_feat = output["reconstruct_feat"]
    
        recons_loss = self.recons_loss(feat, pseudo_feat)
        sparse_loss = self.sparse_loss(output['embeds'])
        total_loss = self.beta_class*class_loss + self.beta_recons*recons_loss #+ self.beta_sparse*sparse_loss

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
