import os
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
from models.vit import ViT
from models.selfvitV2 import SelfViT

from models.swinTransformerV2 import SwinTransformerV2
from models.swinTransformer import SwinTransformer
class Classifier(pl.LightningModule):
    def __init__(self,
                 model=DDMIMV4,
                 resume="DDMIM/log/seed1/version_221/checkpoints/last.ckpt",
                #  hparams_file="DDMIM/log/seed1/version_/hparams.yaml",
                 num_classes=1001,
                 lr=1e-3,
                 momentum=0.9,
                 weight_decay=1e-4,
                 step_size=33,
                 gamma=None,
                 ) :
        super().__init__()
        self.save_hyperparameters()
        # self.model=SwinTransformerV2()
        self.model=model.load_from_checkpoint(resume)
        self.model.modelEMA=nn.Identity()
        self.model.projection_head=nn.Identity()
        self.model.decoder_head=nn.Identity()
        # self.model=SelfViT(dim=96,
        #     num_classes=num_classes,
        #     depth=[2,2,6,2],
        #     dim_stem=64,
        #     dim_head=96
        #     )
        # self.model=SwinTransformer()

        #self.model.freeze()
        '''  
        self.vit=ViT(
            image_size=self.model.hparams.image_size,
            patch_size=self.model.hparams.patch_size,
            num_classes=num_classes,
            dim=self.model.hparams.dim,
            depth=1,
            heads=self.model.hparams.heads,
            mlp_dim=self.model.hparams.mlp_dim,
            dim_head=self.model.hparams.dim_head,
            dropout=0.01,
        )
        self.vit.to_patch_embedding=nn.Identity()
        '''
        # self.model.encoder.head = nn.Sequential(
        #     nn.BatchNorm1d(self.model.encoder.num_features),
        #     nn.Linear(self.model.encoder.num_features, num_classes)
        # )
        self.mlp_head = nn.Sequential(
            nn.BatchNorm1d(self.model.hparams.num_classes),
            nn.Linear(self.model.hparams.num_classes, num_classes),
            nn.GELU(),
            nn.Linear(num_classes, num_classes),
        )
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(self.model.hparams.tsfm_dim),
        #     nn.Linear(self.model.hparams.tsfm_dim, num_classes)
        # )
        #self.mlp_head=nn.Identity()
        #self.relu=nn.ReLU(inplace=True)
              
        self.criterion = nn.CrossEntropyLoss()
        
        #self.lr=lr
        #self.momentum=momentum
    
    def forward(self, x):
        #x=self.model(x)
        y_hat=self.model(x)
        y_hat=self.mlp_head(y_hat)
        #x=x[:,-1]
        #y_hat=self.vit(x)
        #x=x.mean(dim=1)
        #logit=self.mlp_head(x)
        #y_hat=self.relu(logit)
        #y_hat=self.mlp_head(x)
        return  y_hat
        #return self.model(x)
    
    
    def training_step(self, batch, batch_idx): #-> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        #return super().training_step(*args, **kwargs)
        
        
        x,y=batch
        
        y_hat=self(x)
        '''
        x=self.model(x)
        #x=x.mean(dim=1)
        x=x[:,-1]
        #logit=self.mlp_head(x)
        y_hat=self.mlp_head(x)
        #y_hat=self.relu(logit)
        '''
        
        loss=self.criterion(y_hat,y)
        trn_acc1,trn_acc5=accuracy(y_hat,y,topk=[1,5])
        self.log_dict({"trn_loss":loss,"trn_acc1":trn_acc1,"trn_acc5":trn_acc5})
        
        return loss
    
    def validation_step(self,batch, batch_idx):
        #return super().validation_step(*args, **kwargs)
        x,y=batch
        y_hat=self(x)
        '''
        x=self.model(x)
        x=x[:,-1]
        #logit=self.mlp_head(x)
        #y_hat=self.relu(logit)
        y_hat=self.mlp_head(x)
        '''
        val_loss=self.criterion(y_hat,y)
        #val_loss=self.training_step(batch,batch_idx)
        #self.log("val_loss", val_loss)
        val_acc1,val_acc5=accuracy(y_hat,y,topk=[1,5])
        #self.log("acc1",acc1)
        #self.log("acc5",acc5)
        self.log_dict({"val_loss":val_loss,"val_acc1":val_acc1,"val_acc5":val_acc5})
        #loss=F.cross_entropy(y_hat,y)
        #acc=accuracy(y_hat,y)
        return val_loss
    
    def test_step(self, batch, batch_idx):
                #return super().validation_step(*args, **kwargs)
        x,y=batch
        y_hat=self(x)

        test_loss=self.criterion(y_hat,y)

        test_acc1,test_acc5=accuracy(y_hat,y,topk=[1,5])

        self.log_dict({"test_loss":test_loss,"test_acc1":test_acc1,"test_acc5":test_acc5})
  
        # loss=self.validation_step(batch, batch_idx)
        return test_loss#self.cover_step(batch,batch_idx)#
    
    def inference(self, x):
        '''
        x=self.model.inference(x)
        x=x.mean(dim=1)
        logit=self.mlp_head(x)
        y_hat=self.relu(logit)
        '''
        
        return  self(x)
    
    def cover_step(self, batch, batch_idx):
        pass
        #x,y=batch
        #self.model.cover(batch)
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            # weight_decay=self.hparams.weight_decay,
            
            )
        #lr_scheduler = StepLR(optimizer,step_size=self.hparams.step_size,gamma=self.hparams.gamma)
        #lr_scheduler = CosineAnnealingLR(optimizer,T_max=self.hparams.step_size)
        lr_scheduler = ExponentialLR(optimizer,gamma=0.993)
        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
    
 
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

