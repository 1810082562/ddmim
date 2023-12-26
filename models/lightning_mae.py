from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn,Tensor 
import torch.nn.functional as F
from torch.optim import AdamW,SGD
from torch.optim.swa_utils import AveragedModel
from torch.nn import Module
import pytorch_lightning 
from pytorch_lightning import LightningModule
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR,CosineAnnealingWarmRestarts,ExponentialLR
import warnings

import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.vit import Transformer
import matplotlib.pyplot as plt
from models.swinTransformerV2 import SwinTransformerV2,PatchEmbed,PatchMerging,BasicLayer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#from simmim import SwinTransformerV2ForSimMIM
from models.maxvit import MaxViT
from models.models_mae import mae_vit_base_patch16,mae_vit_small_patch16
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)



# class SwinTransformerV2ForDDMIM(SwinTransformerV2):
#     r""" Swin Transformer
#         A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
#           https://arxiv.org/pdf/2103.14030
#     Args:
#         img_size (int | tuple(int)): Input image size. Default 224
#         patch_size (int | tuple(int)): Patch size. Default: 4
#         in_chans (int): Number of input image channels. Default: 3
#         num_classes (int): Number of classes for classification head. Default: 1000
#         embed_dim (int): Patch embedding dimension. Default: 96
#         depths (tuple(int)): Depth of each Swin Transformer layer.
#         num_heads (tuple(int)): Number of attention heads in different layers.
#         window_size (int): Window size. Default: 7
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
#         qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
#         drop_rate (float): Dropout rate. Default: 0
#         attn_drop_rate (float): Attention dropout rate. Default: 0
#         drop_path_rate (float): Stochastic depth rate. Default: 0.1
#         norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
#         ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
#         patch_norm (bool): If True, add normalization after patch embedding. Default: True
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
#         pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
#     """

#     def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
#                  embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
#                  window_size=7, mlp_ratio=4., qkv_bias=True,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
#                  use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
#                  masking_ratio=[0.3,0.3,0.5,0.3],
#                  **kwargs):
#         super().__init__()

#         self.num_classes = num_classes
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.ape = ape
#         self.patch_norm = patch_norm
#         self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
#         self.mlp_ratio = mlp_ratio

#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#         num_patches = self.patch_embed.num_patches
#         patches_resolution = self.patch_embed.patches_resolution
#         self.patches_resolution = patches_resolution

#         # absolute position embedding
#         if self.ape:
#             self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
#             trunc_normal_(self.absolute_pos_embed, std=.02)

#         self.pos_drop = nn.Dropout(p=drop_rate)

#         # stochastic depth
#         # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
#         dpr=[]    
#         for i in range(len(depths)):
#             for j in range(depths[i]):
#                 if j==0:
#                     dpr.append(masking_ratio[i])
#                 else:
#                     dpr.append(0.)
    
#         # build layers
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
#                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
#                                                  patches_resolution[1] // (2 ** i_layer)),
#                                depth=depths[i_layer],
#                                num_heads=num_heads[i_layer],
#                                window_size=window_size,
#                                mlp_ratio=self.mlp_ratio,
#                                qkv_bias=qkv_bias,
#                                drop=drop_rate, attn_drop=attn_drop_rate,
#                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                                norm_layer=norm_layer,
#                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
#                                use_checkpoint=use_checkpoint,
#                                pretrained_window_size=pretrained_window_sizes[i_layer])
#             self.layers.append(layer)

#         self.norm = norm_layer(self.num_features)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

#         self.apply(self._init_weights)
#         for bly in self.layers:
#             bly._init_respostnorm()

#     # def _init_weights(self, m):
#     #     if isinstance(m, nn.Linear):
#     #         trunc_normal_(m.weight, std=.02)
#     #         if isinstance(m, nn.Linear) and m.bias is not None:
#     #             nn.init.constant_(m.bias, 0)
#     #     elif isinstance(m, nn.LayerNorm):
#     #         nn.init.constant_(m.bias, 0)
#     #         nn.init.constant_(m.weight, 1.0)

#     # def forward_features(self, x):
#     #     x = self.patch_embed(x)
#     #     if self.ape:
#     #         x = x + self.absolute_pos_embed
#     #     x = self.pos_drop(x)

#     #     for layer in self.layers:
#     #         x = layer(x)

#     #     x = self.norm(x)  # B L C
#     #     x = self.avgpool(x.transpose(1, 2))  # B C 1
#     #     x = torch.flatten(x, 1)
#     #     return x

#     # def forward(self, x):
#     #     x = self.forward_features(x)
#     #     x = self.head(x)
#     #     return x


class ReparameterizationTrick(nn.Module):
    def forward(self,x:torch.Tensor):
        B,D=x.shape
        mid=int(D/2)
        if D % 2 == 0:
            epsilon=torch.normal(0,1,[B,mid]).to(device=x.device)
            out=x[:,:mid]+torch.exp(x[:,mid:])*epsilon
        else: 
            out=x[:,:mid]
            warnings.warn("dim is not odd")
        return out

class ReparameterizationLoss(nn.Module):
    def forward(self,x:torch.Tensor):
        B,D=x.shape
        mid=int(D/2)
        
        if D % 2 == 0:
            loss=torch.exp(x[:,mid:])-x[:,mid:]-1+x[:,:mid]**2
        else:
            loss=x[:,:mid]**2
        return loss.sum()

class Lightning_mae(LightningModule):
    def __init__(self, 
                 mask_ratio=0.75,
                 lr=7.5e-4,
                 lr_decay_epochs=30,
                 weight_decay=0.05,
                 
                #  *args: Any, **kwargs: Any
                 
                 ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model=mae_vit_small_patch16()
        
    def forward(self, x, *args, **kwargs) -> Any:
        return self.model.forward_vit(x)
    
    def training_step(self, batch, batch_idx,*args, **kwargs) -> STEP_OUTPUT:
        # return super().training_step(*args, **kwargs) 
        x,y=batch
        recon_loss,pred,mask=self.model.forward(x,self.hparams.mask_ratio)
        self.log_dict({"recon_loss":recon_loss})
        total_loss=recon_loss
        return total_loss
    
    def validation_step(self,batch, batch_idx):
        #return super().validation_step(*args, **kwargs)
        #x,y=batch
        val_loss=self.training_step(batch,batch_idx)
        self.log("val_loss", val_loss)
        #loss=F.cross_entropy(y_hat,y)
        #acc=accuracy(y_hat,y)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9,0.95),
            )
        # optimizer=SGD(
        #     self.model.parameters(),
        #     lr=self.hparams.lr,
        #     # momentum=0.9,
        #     weight_decay=self.hparams.weight_decay,
        # )
        # lr_scheduler = StepLR(optimizer,step_size=self.hparams.lr_decay_epochs,gamma=0.1)
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=self.hparams.lr_decay_epochs,T_mult=2,eta_min=1e-4)
        # lr_scheduler = ExponentialLR(optimizer,gamma=0.93)
        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}

class DDMIMV8(LightningModule):
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
        stagefactor=[1.0,1.0,1.0,1.0],
        lr=1e-3,
        momentum=0.9,
        weight_decay=1e-4,
        tau=0.05,
        alpha=0.5,
        epsilon=1e-6,
        ema_decay=0.999,
        ema_steps=32,
        #args,
        # **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model=mae_vit_base_patch16
        # self.encoder=SwinTransformerV2( 
        #     img_size=image_size, 
        #     patch_size=4, 
        #     in_chans=channels, 
        #     num_classes=num_classes,
        #     embed_dim=dim_head, 
        #     depths=depths, 
        #     num_heads=num_heads,
        #     window_size=window_size, 
        #     mlp_ratio=mlp_ratio,  
        #     # drop_path_rate=0., 
        #     # masking_ratio=masking_ratio
        #     )
        # self.modelEMA=AveragedModel(
        #     model=self.encoder,
        #     avg_fn=lambda avg_param,param,num_avg:ema_decay*avg_param+(1-ema_decay)*param,
        # )
        # self.modelEMA.requires_grad_(False)
        # self.modelEMA.train(False)
        
        # self.decoder = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=self.encoder.num_features,
        #         out_channels=encoder_stride ** 2 * 3, kernel_size=1),
        #     nn.PixelShuffle(encoder_stride),
        # ) 
        # self.reparamtrick=ReparameterizationTrick()
        # self.reparamloss=ReparameterizationLoss()
        # # self.projection_head=nn.Identity()       
        # self.projection_head=nn.Sequential(
        #     #self.reparamtrick,
        #     # nn.BatchNorm1d(num_classes),
        #     nn.Linear(num_classes,int(num_classes/2)),
        #     # nn.Linear(num_classes,512),
        #     nn.GELU(),
        #     nn.Linear(int(num_classes/2),512),
        #     # nn.Linear(512,512),
        # )
        
        # # assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        # image_height, image_width = pair(image_size)
        # patch_height, patch_width = pair(patch_size)
        # assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # if isinstance(masking_ratio ,float ):
        #     masking_ratio = [masking_ratio]*4    
        
        self.masking_ratio = masking_ratio
        self.channels=channels
        self.stagefactor = torch.tensor(stagefactor)
        self.tau=tau
        self.lr=lr
        self.weight_decay=weight_decay
        self.alpha = alpha
        self.ema_steps=ema_steps
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
        
    
    def repersentation(self,x):
         with torch.no_grad():
            x = self.encoder.patch_embed(x)

            # assert mask is not None
            B, L, _ = x.shape

            if self.encoder.ape:
                x = x + self.absolute_pos_embed
            x = self.encoder.pos_drop(x)

            for i_layer,layer in enumerate(self.encoder.layers):
                x=layer(x)
            
            x_cls = self.encoder.norm(x)  # B L C
            x_cls = self.encoder.avgpool(x_cls.transpose(1, 2))  # B C 1
            x_cls = torch.flatten(x_cls, 1)
            x_cls=self.encoder.head(x_cls)
            x_cls=self.projection_head(x_cls)
            
                
            # x = self.encoder.norm(x)    
            # x = x.transpose(1, 2)
            # B, C, L = x.shape
            # H = W = int(L ** 0.5)
            # x = x.reshape(B, C, H, W)
            # x=self.decoder(x)
            return x_cls
    
        
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

            B, L, _ = x.shape
            H = W = int(L ** 0.5)
            
            mask=torch.ones(x.shape).to(device=x.device)
            mask=F.dropout2d(mask,self.masking_ratio[i_layer])*(1-self.masking_ratio[i_layer])
            
            #update the total mask

            expend_mask = rearrange(mask,"b (h w) d -> b d h w",h=H,w=W)
            expend_mask=repeat(expend_mask[:,:self.hparams.channels,:,:],"b d h w-> b d (h r1) (w r2)",h=H,w=W,r1=2 ** (i_layer+2),r2=2 ** (i_layer+2))
            total_mask=total_mask*expend_mask
            
            # masking
            x_masked=x_masked*mask
            
            with torch.no_grad():
                x = self.modelEMA.module.layers[i_layer](x)
            
            
            x_masked=layer(x_masked)
            
            # x_cover=x[mask<0.5]
            # x_masked_cover=x_masked[mask<0.5]
            # perceptual_loss=F.smooth_l1_loss(x_masked_cover, x_cover)    
            target=torch.ones(B).to(device=x.device)
            perceptual_loss=F.smooth_l1_loss(x,x_masked)
            # perceptual_loss=F.cosine_embedding_loss(x.view(B,-1),x_masked.view(B,-1),target=target)
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
        reparam_loss=self.reparamloss(x_masked_cls)
        x_masked_cls=self.projection_head(x_masked_cls)
        # x_masked_cls=self.bn(x_masked_cls)
        
        # positive logits: Nx1
        # l_pos=torch.einsum("nc,nc->n",[x_masked_cls,x_cls]).unsqueeze(-1)#+self.hparams.epsilon
        l_pos=torch.cosine_similarity(x_masked_cls,x_cls.detach()).unsqueeze(-1)
    
        
        # negative logits: NxN
        # l_neg=torch.einsum("nc,mc->nm",[x_masked_cls,x_cls])
        l_neg=torch.cosine_similarity(x_masked_cls.unsqueeze(1),x_cls.unsqueeze(0),dim=2)
        l_neg=l_neg-torch.diag_embed(torch.diag(l_neg))
        
        
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
        
        #token_masked=x_masked[total_mask<0.5]
        token_masked=x_masked
        #token_origin=origin[total_mask<0.5]
        token_origin=origin
        # calculate reconstruction loss
        recon_loss = F.smooth_l1_loss(token_masked, token_origin)
        #recon_loss=F.mse_loss(pred_pixel_values,masked_patches)

        total_loss=torch.matmul(stage_loss,self.stagefactor)+recon_loss+self.alpha*contractive_loss#+0.01*reparam_loss
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
    
    def on_train_batch_end(self, outputs, batch, batch_idx,unused=0) -> None:
        if batch_idx % self.ema_steps == 0:
            self.modelEMA.update_parameters(self.encoder)
        
        return super().on_train_batch_end(outputs, batch, batch_idx, unused)
    
    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.weight_decay,
            
            )
        # lr_scheduler = StepLR(optimizer,step_size=10,gamma=0.618)
        # lr_scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=8,eta_min=1e-4)
        # lr_scheduler = ExponentialLR(optimizer,gamma=0.93)
        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
    
    