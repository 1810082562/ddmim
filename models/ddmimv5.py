import torch
from torch import nn,Tensor 
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel
from torch.nn import Module
import pytorch_lightning 
from pytorch_lightning import LightningModule
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR,CosineAnnealingWarmRestarts

import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.vit import Transformer
import matplotlib.pyplot as plt
from models.swinTransformerV2 import SwinTransformerV2
#from simmim import SwinTransformerV2ForSimMIM
from models.maxvit import MaxViT

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)



'''
class SwinTEncoder(SwinTransformerV2):
    def __init__(self,mask_ratio=0.3,**kwargs):
        super().__init__(**kwargs)
        self.mask_ratio=mask_ratio
    def forward(self, x:Tensor):
        x = self.patch_embed(x)

        # assert mask is not None
        B, L, _ = x.shape
        total_mask=torch.ones(x.shape)
        # mask_tokens = self.mask_token.expand(B, L, -1)
        # w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        # x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_masked =x.detach()
        for i_layer,layer in enumerate(self.layers):
 
            
            mask=torch.ones(x.shape)
            mask=F.dropout2d(mask,self.mask_ratio)*self.mask_ratio
            
            #update the total mask
            expend_mask=repeat(mask[...,...,:self.encoder.patch_embed.in_chans],"b l d -> b (l r) d",r=2 ** i_layer)
            total_mask=total_mask*expend_mask
            
            #masking
            x_masked=x_masked*mask
            
            x = layer(x)
            x_masked=layer(x_masked)
            
            # x_cover=x[mask<0.5]
            # x_masked_cover=x_masked[mask<0.5]
            # perceptual_loss=F.smooth_l1_loss(x_masked_cover, x_cover)            
            perceptual_loss=F.smooth_l1_loss(x,x_masked)
            
            
        
        
        
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)
        return x
        
''' 




class DDMIMV5(LightningModule):
    def __init__(
        self,
        *,
        image_size=224,
        patch_size=16,
        num_classes=2048,
        # tsfm_dim,
        depths=[2, 2, 6, 2],
        # heads=8,
        num_heads=[3, 6, 12, 24],
        dim_head=96,
        window_size=7, 
        mlp_ratio=4.,
        # mlp_dim=2048,
        channels=3,
        encoder_stride=32,
        masking_ratio=[0.3,0.3,0.5,0.3],
        stagefactor=[0.1,0.1,0.1,0.1],
        lr=1e-3,
        momentum=0.9,
        weight_decay=1e-4,
        tau=0.05,
        alpha=1,
        epsilon=1e-6,
        #args,
        # **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder=SwinTransformerV2( 
            img_size=image_size, 
            patch_size=4, 
            in_chans=channels, 
            num_classes=num_classes,
            embed_dim=dim_head, 
            depths=depths, 
            num_heads=num_heads,
            window_size=window_size, 
            mlp_ratio=mlp_ratio,  
            drop_path_rate=0., 
            )
        # self.encoderEMA=torch.utils
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(encoder_stride),
        ) 
        # self.projection_head=nn.Identity()       
        self.projection_head=nn.Sequential(
            nn.Linear(num_classes,int(num_classes/2)),
            nn.GELU(),
            nn.Linear(int(num_classes/2),int(num_classes/4)),
        )
        
        # assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        if isinstance(masking_ratio ,float ):
            masking_ratio = [masking_ratio]*4    
        
        self.masking_ratio = masking_ratio
        self.channels=channels
        self.stagefactor = torch.tensor(stagefactor)
        self.tau=tau
        self.lr=lr
        self.weight_decay=weight_decay
        self.alpha = alpha
        # self.bn=nn.BatchNorm1d(num_classes)

        
    def forward(self,x:torch.Tensor):
        x = self.encoder.patch_embed(x)

        # assert mask is not None
        B, L, _ = x.shape

        if self.encoder.ape:
            x = x + self.absolute_pos_embed
        x = self.encoder.pos_drop(x)
        for layer in self.encoder.layers:
            x = layer(x)
        
        x = self.encoder.norm(x)  # B L C
        x = self.encoder.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x=self.encoder.head(x)
        # x=self.bn(x)
        # x = self.encoder.norm(x)    
        # x = x.transpose(1, 2)
        # B, C, L = x.shape
        # H = W = int(L ** 0.5)
        # x = x.reshape(B, C, H, W)
        # x=self.decoder(x)
        return x
        

    def inferences(self,x):
         with torch.no_grad():
            x = self.encoder.patch_embed(x)

            # assert mask is not None
            B, L, _ = x.shape

            if self.encoder.ape:
                x = x + self.absolute_pos_embed
            x = self.encoder.pos_drop(x)

            for i_layer,layer in enumerate(self.encoder.layers):
                x=layer(x)
                
            x = self.encoder.norm(x)    
            x = x.transpose(1, 2)
            B, C, L = x.shape
            H = W = int(L ** 0.5)
            x = x.reshape(B, C, H, W)
            x=self.decoder(x)
            return x
        
        
    def cover(self,x):
        # get patches
        with torch.no_grad():
            origin=x.clone()
            total_mask=torch.ones(x.shape)
            x = self.encoder.patch_embed(x)

            # assert mask is not None
            B, L, _ = x.shape

            
            if self.encoder.ape:
                x = x + self.absolute_pos_embed
            x = self.encoder.pos_drop(x)
            x_masked =x
            for i_layer,layer in enumerate(self.encoder.layers):

                mask=torch.ones(x_masked.shape)
                mask=F.dropout2d(mask,self.masking_ratio[i_layer])*(1-self.masking_ratio[i_layer])
                
                #update the total mask
                B, L, _ = x_masked.shape
                H = W = int(L ** 0.5)
                expend_mask = rearrange(mask,"b (h w) d -> b d h w",h=H,w=W)
                expend_mask=repeat(expend_mask[:,:self.channels,:,:],"b d h w-> b d (h r1) (w r2)",h=H,w=W,r1=2 ** (i_layer+2),r2=2 ** (i_layer+2))
                total_mask=total_mask*expend_mask
                
                #masking
                x_masked=x_masked*mask
                
                # x = layer(x)
                x_masked=layer(x_masked)
                
                
            x_masked = self.encoder.norm(x_masked)    
            x_masked = x_masked.transpose(1, 2)
            B, C, L = x_masked.shape
            H = W = int(L ** 0.5)
            x_masked = x_masked.reshape(B, C, H, W)
            x_masked=self.decoder(x_masked)
            return x_masked
                
    def to_image(self,imgs):
        with torch.no_grad():
            mean=torch.tensor([0.485, 0.456, 0.406])
            std=torch.tensor([0.229, 0.224, 0.225])
            to_pil=transforms.ToPILImage()
            inverse_norm=transforms.Normalize((-mean/std).tolist(),(1.0/std).tolist())
            # pred_pixel_values = self.to_pixels(tokens)
            #pred_pixel_values=torch.clamp(pred_pixel_values,0,255)
            # imgs=self.pixels_to_img(pred_pixel_values)
            imgs=imgs.squeeze()
            
            imgs=inverse_norm(imgs)
            imgs=torch.clamp(imgs,0,255)
            recover=to_pil(imgs)
            #recover=torch.clamp(recover,0,255)
            return recover
    
    
    def testsavepicture(self,masked_bool_mask,masked_tokens,n,patches):
        #masked_pic = torch.where(masked_bool_mask[..., None], masked_tokens,patches)
        
        #recover to image data
        mean=torch.tensor([0.485, 0.456, 0.406])
        std=torch.tensor([0.229, 0.224, 0.225])
        topil=transforms.ToPILImage()
        revernorm=transforms.Normalize((-mean/std).tolist(),(1.0/std).tolist())
        pred_pixel_values = self.to_pixels(masked_tokens)
        masked_pic = torch.where(masked_bool_mask[..., None], pred_pixel_values,patches)
        imgs=self.pixels_to_img(masked_pic)
       
        #recover to image format
        imgs=imgs.squeeze()
        imgs=revernorm(imgs)
        imgs=torch.clamp(imgs,0,255)
        cover=topil(imgs)
        
        #save image
        plt.axis('off')
        plt.imshow(cover)
        plt.savefig(f'stage{n}.jpg',bbox_inches='tight',pad_inches=0)
        
    
    def training_step(self, batch, batch_idx): #-> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        #return super().training_step(*args, **kwargs)
        x,y=batch
        origin=x.clone()
        #loss=self(x)
        stage_loss=torch.zeros(len(self.hparams.depths))
        
        total_mask=torch.ones(x.shape).to(device=x.device)
        x = self.encoder.patch_embed(x)

        # assert mask is not None
        B, L, _ = x.shape


        if self.encoder.ape:
            x = x + self.absolute_pos_embed
        x = self.encoder.pos_drop(x)
        x_masked =x.clone()
        for i_layer,layer in enumerate(self.encoder.layers):

            
            mask=torch.ones(x.shape).to(device=x.device)
            mask=F.dropout2d(mask,self.masking_ratio[i_layer])*(1-self.masking_ratio[i_layer])
            
            #update the total mask
            B, L, _ = x.shape
            H = W = int(L ** 0.5)
            expend_mask = rearrange(mask,"b (h w) d -> b d h w",h=H,w=W)
            expend_mask=repeat(expend_mask[:,:self.hparams.channels,:,:],"b d h w-> b d (h r1) (w r2)",h=H,w=W,r1=2 ** (i_layer+2),r2=2 ** (i_layer+2))
            total_mask=total_mask*expend_mask
            
            #masking
            x_masked=x_masked*mask
            
            with torch.no_grad():
                x = layer(x)
            
            
            x_masked=layer(x_masked)
            
            # x_cover=x[mask<0.5]
            # x_masked_cover=x_masked[mask<0.5]
            # perceptual_loss=F.smooth_l1_loss(x_masked_cover, x_cover)            
            perceptual_loss=F.smooth_l1_loss(x,x_masked)
            stage_loss[i_layer]=perceptual_loss
        
        with torch.no_grad():
            x_cls = self.encoder.norm(x)  # B L C
            x_cls = self.encoder.avgpool(x_cls.transpose(1, 2))  # B C 1
            x_cls = torch.flatten(x_cls, 1)
            x_cls=self.encoder.head(x_cls)
            x_cls=self.projection_head(x_cls)
            # x_cls=self.bn(x_cls)
           
        x_masked_cls = self.encoder.norm(x_masked)  # B L C
        x_masked_cls = self.encoder.avgpool(x_masked_cls.transpose(1, 2))  # B C 1
        x_masked_cls = torch.flatten(x_masked_cls, 1)
        x_masked_cls=self.encoder.head(x_masked_cls)
        x_masked_cls=self.projection_head(x_masked_cls)
        # x_masked_cls=self.bn(x_masked_cls)
        
        # positive logits: Nx1
        # l_pos=torch.einsum("nc,nc->n",[x_masked_cls,x_cls]).unsqueeze(-1)#+self.hparams.epsilon
        l_pos=torch.cosine_similarity(x_masked_cls,x_cls.detach()).unsqueeze(-1)
        
        
        
        
        
        # negative logits: NxN
        # l_neg=torch.einsum("nc,mc->nm",[x_masked_cls,x_cls])
        l_neg=torch.cosine_similarity(x_masked_cls.unsqueeze(1),x_cls.unsqueeze(0),dim=2)
        
        
        
        # logits: Nx(1+N)
        logits=torch.cat([l_pos,l_neg],dim=1)/(self.tau)
        # logits=F.layer_norm(logits,[logits.shape[-1]])
        labels=torch.zeros(B,dtype=torch.long).to(device=x.device)
        contractive_loss=F.cross_entropy(logits,labels)
        #assert not contractive_loss == np.NaN ,"too many contract"
            
                
        x_masked = self.encoder.norm(x_masked)    
        x_masked = x_masked.transpose(1, 2)
        B, C, L = x_masked.shape
        H = W = int(L ** 0.5)
        x_masked = x_masked.reshape(B, C, H, W)
        x_masked=self.decoder(x_masked)
        
        token_masked=x_masked[total_mask<0.5]
        token_origin=origin[total_mask<0.5]
        
        # calculate reconstruction loss
        recon_loss = F.smooth_l1_loss(token_masked, token_origin)
        #recon_loss=F.mse_loss(pred_pixel_values,masked_patches)

        total_loss=torch.matmul(stage_loss,self.stagefactor)+recon_loss+self.alpha*contractive_loss
        # self.log_dict({"recon_loss":recon_loss,"total_loss":total_loss,})
        self.log_dict({"recon_loss":recon_loss,"total_loss":total_loss,"contractive_loss":contractive_loss})
        #assert torch.isnan(total_loss), "totalloss is nan"
        return total_loss
    
    def validation_step(self,batch, batch_idx):
        #return super().validation_step(*args, **kwargs)
        #x,y=batch
        val_loss=self.training_step(batch,batch_idx)
        self.log("val_loss", val_loss)
        #loss=F.cross_entropy(y_hat,y)
        #acc=accuracy(y_hat,y)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        #return super().test_step(*args, **kwargs)
        loss=self.validation_step(batch, batch_idx)
        return loss#self.cover_step(batch,batch_idx)#
    
    def cover_step(self, batch, batch_idx):
        
        return self.cover(batch)
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay,
            
            )
        lr_scheduler = StepLR(optimizer,step_size=10,gamma=0.618)
        # lr_scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=8,eta_min=1e-4)
        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
    
    