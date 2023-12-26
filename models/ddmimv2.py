import torch
from torch import nn,Tensor 
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import Module
import pytorch_lightning 
from pytorch_lightning import LightningModule
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.vit import Transformer
import matplotlib.pyplot as plt

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)



class DDMIMV2(LightningModule):
    def __init__(
        self,
        *,
        image_size=224,
        patch_size=16,
        #num_classes,
        tsfm_dim,
        stages_depth=[2,2,6,2],
        heads=8,
        mlp_dim=2048,
        channels=3,
        dim_head=64,
        masking_ratio=0.3,
        stagefactor=[0.01,0.01,0.01,0.01],
        lr=1e-3,
        momentum=0.9,
        weight_decay=1e-4,
        tau=0.7,
        alpha=1,
        epsilon=1e-6
        #args,
    ):
        super(DDMIMV2, self).__init__()
        self.save_hyperparameters()
        
        
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        self.masking_ratio = masking_ratio
        self.stagefactor = torch.tensor(stagefactor)

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, tsfm_dim),
        )        
        self.to_patch,self.patch_to_emb=self.to_patch_embedding[:2]
        self.cls_token = nn.Parameter(torch.randn(1, 1, tsfm_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, tsfm_dim))
        
        #pixel_values_per_patch = self.patch_to_emb.weight.shape[-1]
        
        self.token_to_mask = nn.Parameter(torch.randn(tsfm_dim))
        
        self.to_pixels = nn.Linear(tsfm_dim, patch_dim)
        
        self.pixels_to_img=Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=image_height // patch_height,p1=patch_height,p2=patch_width)
        
        self.stages=nn.ModuleList([])
        
        self.CELoss = nn.CrossEntropyLoss()
        
        self.layer_norm=nn.LayerNorm(tsfm_dim)

        for depth in stages_depth:
            self.stages.append(Transformer(
                dim=tsfm_dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim
                )
            )
            
        #self.maskmaps=[]
        
    def forward(self,imgs:torch.Tensor):
        patches=self.to_patch(imgs)
        tokens =self.patch_to_emb(patches)
        cls_token=repeat(self.cls_token, '1 1 d -> b 1 d', b = imgs.shape[0])
        tokens=torch.cat((tokens,cls_token),dim=1)
        #tokens=patches
        tokens =tokens + self.pos_embedding 
        for i in range(len(self.stages)):
            tokens=self.stages[i](tokens)
        return tokens
    
    def cover(self,img):
        # get patches
        with torch.no_grad():
            patches=self.to_patch(img)
            batch,num_patches,_=patches.shape
            batch_range = torch.arange(batch)[:, None]
            
            # patch to encoder tokens and add positions
            tokens =self.patch_to_emb(patches)
            cls_token=repeat(self.cls_token, '1 1 d -> b 1 d', b = img.shape[0])
            tokens=torch.cat((tokens,cls_token),dim=1)
            #tokens=patches
            tokens =tokens + self.pos_embedding
            
            # prepare tokens for masking

            mask_tokens = repeat(self.token_to_mask, 'd -> b n d', b = batch, n = num_patches+1)
            mask_tokens = mask_tokens + self.pos_embedding
                
            # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
            num_masked = int(self.masking_ratio * num_patches)
            
            masked_indices = torch.rand(batch, num_patches).topk(k = num_masked, dim = -1).indices
            masked_bool_mask = torch.zeros((batch, num_patches+1)).scatter_(-1, masked_indices, 1).bool().to(tokens.device)

            # mask tokens
            masked_tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)


            for i in range(len(self.stages)):
                masked_tokens = self.stages[i](masked_tokens)
                self.testsavepicture(masked_bool_mask[:,:-1],masked_tokens[:,:-1],i,patches)

            return masked_tokens
            '''
            # small linear projection for predicted pixel values
            pred_pixel_values = self.to_pixels(masked_tokens[:,:-1])
            imgs=self.pixels_to_img(pred_pixel_values)
            return imgs
            '''
    '''  
    def inference(self,imgs):
        
        
        # get patches
        patches=self.to_patch(imgs)
        batch,num_patches,_=patches.shape
        batch_range = torch.arange(batch)[:, None]
        
        # patch to encoder tokens and add positions
        tokens =self.patch_to_emb(patches)
        #tokens=patches
        tokens =tokens + self.pos_embedding 
        for i in range(len(self.stages)):
            tokens=self.stages[i](tokens)
        return tokens
    '''
                
    def to_image(self,tokens):
        with torch.no_grad():
            mean=torch.tensor([0.485, 0.456, 0.406])
            std=torch.tensor([0.229, 0.224, 0.225])
            to_pil=transforms.ToPILImage()
            inverse_norm=transforms.Normalize((-mean/std).tolist(),(1.0/std).tolist())
            pred_pixel_values = self.to_pixels(tokens)
            #pred_pixel_values=torch.clamp(pred_pixel_values,0,255)
            imgs=self.pixels_to_img(pred_pixel_values)
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
        #loss=self(x)
        stage_loss=torch.zeros(len(self.stages))

        # get patches
        patches=self.to_patch(x)
        num_batches,num_patches,_=patches.shape
        batch_range = torch.arange(num_batches)[:, None]
        
        # patch to encoder tokens and add positions
        tokens =self.patch_to_emb(patches)
        cls_token=repeat(self.cls_token, '1 1 d -> b 1 d', b = num_batches)
        tokens=torch.cat((tokens,cls_token),dim=1)
        #tokens=patches
        tokens =tokens + self.pos_embedding
        
        # prepare tokens for masking
        mask_tokens = repeat(self.token_to_mask, 'd -> b n d', b = num_batches, n = num_patches+1)
        #mask_cls_token=repeat(self.cls_token, '1 1 d -> b 1 d', b = num_batches)
        #mask_tokens=torch.cat((mask_tokens,mask_cls_token),dim=1)
        mask_tokens = mask_tokens + self.pos_embedding

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_patches)
        masked_indices = torch.rand(num_batches, num_patches).topk(k = num_masked, dim = -1).indices
        masked_bool_mask = torch.zeros((num_batches, num_patches+1)).scatter_(-1, masked_indices, 1).bool().to(tokens.device)

        # mask tokens
        masked_tokens = torch.where(masked_bool_mask[..., None], mask_tokens, tokens)
    
        for i in range(len(self.stages)):
            masked_tokens = self.stages[i](masked_tokens)
            encoded_masked_tokens=masked_tokens[batch_range, masked_indices]
            #if self.training==True:
            
            tokens=self.stages[i](tokens)
            encoded_tokens=tokens[batch_range, masked_indices]
            perceptual_loss=F.smooth_l1_loss(encoded_masked_tokens, encoded_tokens)
            # perceptual_loss=F.mse_loss(encoded_masked_tokens, encoded_tokens)
            stage_loss[i]=perceptual_loss
        
        
        #_____InfoNCE_______
        # keys: NxC
        normal_cls=self.layer_norm(tokens[:,num_patches])
        
        # queries: NxC
        masked_cls=self.layer_norm(masked_tokens[:,num_patches])
        
        
        # positive logits: Nx1
        l_pos=torch.einsum("nc,nc->n",[masked_cls,normal_cls]).unsqueeze(-1)+self.hparams.epsilon
        
        # negative logits: NxN
        l_neg=torch.einsum("nc,mc->nm",[masked_cls,normal_cls])
        
        # logits: Nx(1+N)
        logits=torch.cat([l_pos,l_neg],dim=1)/self.hparams.tau
        
        labels=torch.zeros(num_batches,dtype=torch.long).to(device=tokens.device)
        contractive_loss=self.CELoss(logits,labels)
        #assert not contractive_loss == np.NaN ,"too many contract"
        
        # small linear projection for predicted pixel values
        pred_pixel_values = self.to_pixels(encoded_masked_tokens)
        
        # get the masked patches for the final reconstruction loss
        masked_patches = patches[batch_range, masked_indices]
        
        
        
        # calculate reconstruction loss
        recon_loss = F.smooth_l1_loss(pred_pixel_values, masked_patches)
        #recon_loss=F.mse_loss(pred_pixel_values,masked_patches)

        total_loss=torch.matmul(stage_loss,self.stagefactor)+recon_loss+self.hparams.alpha*contractive_loss
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
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            
            )
        lr_scheduler = StepLR(optimizer,step_size=10)
        return {"optimizer":optimizer,"lr_scheduler":lr_scheduler}
    
    