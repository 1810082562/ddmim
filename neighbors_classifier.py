import os
from typing import Any, Optional
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torch.optim import AdamW,Adam
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR,ExponentialLR
from torch.nn import Module
import torch.nn.functional as F
import pytorch_lightning as pl
from models.ddmim import DDMIM
from models.ddmimv2 import DDMIMV2
from models.ddmimv4 import DDMIMV4
from models.ddmimv6 import DDMIMV6
from models.vit import ViT
from models.selfvitV2 import SelfViT

from models.swinTransformerV2 import SwinTransformerV2
from models.swinTransformer import SwinTransformer
from sklearn.neighbors import KNeighborsClassifier




class NeighborsClassifier(pl.LightningModule):
    def __init__(self, model=DDMIMV6, resume=None) -> None:
        super().__init__()

        print(str(self.device)+" load from "+resume)
        self.model=model.load_from_checkpoint(resume,map_location=self.device)
        self.model.modelEMA=nn.Identity()
        self.model.projection_head=nn.Identity()
        self.model.decoder_head=nn.Identity()
        self.model.freeze()
        self.model.train(False)
        self.knClassfier=KNeighborsClassifier()
        self.criterion = nn.CrossEntropyLoss()
        pass
    def training_step(self, batch:torch.Tensor, batch_idx):
        x,y=batch
        y_hat=self.model(x)
        self.knClassfier.fit(y_hat.numpy(),y.numpy())
        
        return None
        
    def validation_step(self,batch, batch_idx):
        x,y=batch
        y_hat=self.model(x)
        y_pred=self.knClassfier.predict(y_hat.numpy())
        val_loss=self.criterion(y_pred,y)

        val_acc1,val_acc5=accuracy(y_pred,y,topk=[1,5])

        self.log_dict({"val_loss":val_loss,"val_acc1":val_acc1,"val_acc5":val_acc5})

        return val_loss
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), 
            lr=0,
            # weight_decay=self.hparams.weight_decay,
            
            )
        #lr_scheduler = StepLR(optimizer,step_size=self.hparams.step_size,gamma=self.hparams.gamma)
        #lr_scheduler = CosineAnnealingLR(optimizer,T_max=self.hparams.step_size)
        #lr_scheduler = ExponentialLR(optimizer,gamma=0.993)
        return optimizer
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
