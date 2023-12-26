
# -*- coding: utf-8 -*-
# url:https://github.com/lucidrains/vit-pytorch
"""@misc{dosovitskiy2020image,
    title   = {An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author  = {Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
    year    = {2020},
    eprint  = {2010.11929},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}"""


import warnings
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange,Reduce

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def nan_to_num(val:torch.Tensor,massage=""):
    if val.isnan().sum()>0:
        val[val.isnan()]=0.0
        warnings.warn(f"output NaN with {massage}")
    if val.dtype==torch.float16 and val.isinf().sum()>0:
        val[val.isposinf()]=2**16*1.0-2**9
        val[val.isneginf()]=-2**16*1.0+2**9
        warnings.warn(f"output inf with {massage}")
    return val

class PrintLayer(nn.Module):
    def __init__(self,message=""):
        super(PrintLayer, self).__init__()
        self.message =message
    def forward(self, x):
        # Do your print / debug stuff here
        print(self.message,end=" ")
        print(x.shape)      #print(x.shape)
        return x


# classes

# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__() 
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn
#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x)) + x
    
# class PostNormResidual(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn

#     def forward(self, x):
#         return self.norm(self.fn(x)) + x

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout = 0.):
        super().__init__()
        hidden_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.,window_size =-1):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        self.window_size = window_size
        if self.window_size>0:
            # relative positional bias
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

            pos = torch.arange(window_size)
            #grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
            grid = torch.stack(torch.meshgrid(pos, pos))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
            rel_pos += window_size - 1
            rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

            self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)
        

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        #________________________ change this _______________________________________
        dots = torch.matmul(q , k.transpose(-1, -2))* self.scale

        # add positional bias
        if self.window_size>0:
            bias = self.rel_pos_bias(self.rel_pos_indices)
            dots=dots +  rearrange(bias, 'i j h -> h i j')* self.scale


        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        #assert not torch.isnan(out).sum()>0, "nan"

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



# MBConv
'''
class SqueezeExcitation(nn.Module):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Module):
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        #PrintLayer("In MBBlock "),
        nn.Conv2d(dim_in, hidden_dim, 1),

        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net
'''


class AttentionBlock(nn.Module):
    def __init__(self,
        *,
        image_size=224,
        dim, 
        heads = 8, 
        dim_head = 64, 
        dropout = 0.,
        window_size = 7,
        window_wise = False,
        gridlike_wise = False,
        channel_wise= False,
    ):
        super().__init__()
        window_rows=image_size//window_size
        
        self.rearrange=nn.Identity()
        self.expose=nn.Identity()
        self.recover=nn.Identity()
        
        if window_wise:
            if not channel_wise:
                #self.printlayer = PrintLayer("In WindowAttention")
                self.rearrange= Rearrange('b (x w1 y w2) d-> (b x y) (w1 w2) d', w1 = window_size, w2 = window_size,x=window_rows,y=window_rows)
                self.expose=Rearrange('(b x y) (w1 w2) d -> b (x w1 y w2) d',w1 = window_size, w2 = window_size,x=window_rows,y=window_rows)
                self.recover=nn.Identity()
                #self.recover=Rearrange('b (x w1 y w2) d -> b d (x w1) (y w2)',w1 = window_size, w2 = window_size,x=window_rows,y=window_rows)
            
            else:    #elif channel_wise:
                #self.printlayer = PrintLayer("In ChannelAttention")
                self.rearrange= Rearrange('b (x w1 y w2) d-> (b x y) d (w1 w2)', w1 = window_size, w2 = window_size,x=window_rows,y=window_rows)
                self.expose=Rearrange('(b x y) d (w1 w2) -> b d (x w1 y w2)',w1 = window_size, w2 = window_size,x=window_rows,y=window_rows)
                self.recover= Rearrange('b d (x w1 y w2) -> b (x w1 y w2) d',w1 = window_size, w2 = window_size,x=window_rows,y=window_rows)    
        
        elif gridlike_wise:
            if not channel_wise:
                #self.printlayer = PrintLayer("In GridLikeAttention")
                self.rearrange=Rearrange('b (w1 x w2 y) d-> (b x y) (w1 w2) d', w1 = window_size, w2 = window_size,x=window_rows,y=window_rows)
                self.expose=Rearrange('(b x y) (w1 w2) d -> b (w1 x w2 y) d',w1 = window_size, w2 = window_size,x=window_rows,y=window_rows)
                self.recover=nn.Identity()
                # self.recover=Rearrange('b (w1 x w2 y) d -> b d (w1 x) (w2 y)',w1 = window_size, w2 = window_size,x=window_rows,y=window_rows)
            else: #elif channel_wise:
                #self.printlayer = PrintLayer("In ChannelAttention")
                self.rearrange= Rearrange('b (w1 x w2 y) d -> (b x y) d (w1 w2) ', w1 = window_size, w2 = window_size,x=window_rows,y=window_rows)
                self.expose=Rearrange('(b x y) d (w1 w2) -> b d (w1 x w2 y)',w1 = window_size, w2 = window_size,x=window_rows,y=window_rows)
                self.recover= Rearrange('b d (w1 x w2 y) -> b (w1 x w2 y) d',w1 = window_size, w2 = window_size,x=window_rows,y=window_rows)
        
        input_dim = dim if not channel_wise else window_size*window_size
        self.attntion= PreNormResidual(
                input_dim,
                Attention(
                    dim = input_dim, 
                    heads = heads, 
                    dim_head = dim_head, 
                    dropout = dropout,
                    window_size=window_size if not channel_wise else -1
                )
            )
        ffdim=dim if not channel_wise else input_dim*window_rows*window_rows
        self.mlp=PreNormResidual(ffdim, FeedForward(dim=ffdim,dropout=dropout)) 

        
    def forward(self,x):
        #self.printlayer(x)
        x=self.rearrange(x)
        x=self.attntion(x)
        x=self.expose(x)
        x=self.mlp(x)
        x=self.recover(x)
        # x=self.recover(
        #     self.feedforward(
        #         self.expose(
        #             self.attntion(
        #                 self.rearrange(x)
        #                 )
        #             )
        #         )
        #     )
        return x
         

           
class SelfViT(nn.Module):
    def __init__(
        self,
        *,
        image_size=224,
        num_classes=1000,
        channels=3,
        dim=64, 
        dim_stem = 64,
        heads=[2,4,8,16],
        dim_head = 64,
        depth = [2,2,6,2],
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
    ):
        super().__init__()
        
        dim_stem = default(dim_stem, dim)

        self.stem = nn.Sequential(
            Rearrange("b d (h p1) (w p2) -> b (p1 p2 d) h w ",p1=2,p2=2),
            #nn.Linear(in_features=channels*4,out_features=channels*16),
            
            nn.Conv2d(channels*4,dim_stem,3,padding=1)
            # nn.Conv2d(channels, dim_stem, 3, stride = 2, padding = 1),
            # nn.Conv2d(dim_stem, dim_stem, 3, padding = 1)
        )
        
        num_stages = len(depth)
        #heads=[4,6,8,12]
        #heads= [2**i*2 for i in range(num_stages)]
        #dims=[96,192,384,768]
        dims = [2**i*dim for i in range(num_stages)]
        size = [image_size//(2**(i+2)) for i in range(num_stages)]
        
        self.layers = nn.ModuleList([])
        
        for i in range(num_stages):

            is_first= i==0 
            
            layer=None
            layer=nn.Sequential(
                # #PrintLayer(f"In layer i:{i}")
                
                # nn.BatchNorm2d(
                #     dim_conv_stem if is_first else dims[i-1],
                #     #size[i]*2 if is_first else size[i-1],
                #     #size[i]*2 if is_first else size[i-1]
                #     ),
                # nn.GELU(),
                # nn.Conv2d(
                # in_channels= dim_conv_stem if is_first else dims[i-1],
                # out_channels=dims[i],
                # kernel_size=3,
                # stride=2,
                # groups=1 if is_first else dims[i-1],
                # padding=1,        
                # ) 
            )
            if is_first:
                layer.add_module("Patch Mergeing",
                                 Rearrange("b d (h p1) (w p2) -> b (h w) (p1 p2 d)",p1=2,p2=2,h=size[0],w=size[0]))
                layer.add_module("downchannel",nn.Linear(dim_stem*4,dims[i]))
            else:
                layer.add_module("Patch Mergeing",
                                 Rearrange("b (h p1 w p2) d -> b (h w) (p1 p2 d)",p1=2,p2=2,h=size[i],w=size[i]))
                layer.add_module("downchannel",nn.Linear(dims[i-1]*4,dims[i]))
            
            # layer.add_module("downsample{i}",nn.Conv2d(
            #     in_channels= dim_conv_stem if is_first else dims[i-1],
            #     out_channels=dims[i],
            #     kernel_size=3,
            #     stride=2,
            #     padding=1,        
            # ))
            # layer.add_module("MBConv",MBConv(
            #         dim_in=dim_conv_stem if is_first else dims[i-1],
            #         dim_out=dims[i],
            #         downsample=True,
            #         expansion_rate = mbconv_expansion_rate,
            #         shrinkage_rate = mbconv_shrinkage_rate,
            #     ))
            for j in range(depth[i]):
                # block=AttentionBlock(
                #     image_size=size[i],
                #     dim=dims[i],
                #     heads=heads,
                #     dim_head=dim_head,
                #     window_size=window_size,
                #     dropout=dropout,
                # )
                if j%4==0:
                    layer.add_module(f"WindowAttention{j}",AttentionBlock(
                        image_size=size[i],
                        dim=dims[i],
                        heads=heads[i],
                        dim_head=dim_head,
                        window_size=window_size,
                        dropout=dropout,
                        window_wise=True,
                    ))
                if j%4==1:
                    layer.add_module(f"GridLikeAttention{j}",AttentionBlock(
                        image_size=size[i],
                        dim=dims[i],
                        heads=heads[i],
                        dim_head=dim_head,
                        window_size=window_size,
                        dropout=dropout,
                        gridlike_wise=True,
                    ))
                if j%4==2:
                    layer.add_module(f"WChannelAttention{j}",
                    AttentionBlock(
                        image_size=size[i],
                        dim=dims[i],
                        heads=heads[i],
                        dim_head=dim_head,
                        window_size=window_size,
                        dropout=dropout,
                        window_wise=True,
                        channel_wise=True,
                    ))
                if j%4==3:
                    layer.add_module(f"GChannelAttention{j}",AttentionBlock(
                        image_size=size[i],
                        dim=dims[i],
                        heads=heads[i],
                        dim_head=dim_head,
                        window_size=window_size,
                        dropout=dropout,
                        gridlike_wise=True,
                        channel_wise=True,
                    ))
                

            self.layers.append(layer)
            
        self.gate=nn.Sequential(
            #nn.BatchNorm2d(dims[-1]),
            nn.LayerNorm(dims[-1]),
            Rearrange("b (h w) d -> b d h w",h=window_size,w=window_size),
            #nn.A(window_size)
            nn.Conv2d(dims[-1],dims[-1]*4,window_size,groups=dims[-1]),
            # Reduce('b d h w -> b d', 'mean'),
            Rearrange('b d h w -> b (d h w)')
            #nn.BatchNorm1d(dims[-1]*4),
            # nn.Linear(dims[-1]*2, num_classes),
            
            )
        
        self.head =  nn.Linear(dims[-1]*4, num_classes)
        # nn.Sequential(
            #Reduce('b d h w -> b d', 'mean'),

            # nn.LayerNorm(dims[-1]*2,eps=0.001),
            # nn.Linear(dims[-1]*2, num_classes)
        # )        
        
    
    def forward(self,x):
        # with autocast():
        x=self.stem(x)

        for i in range(len(self.layers)):
            x=self.layers[i](x)
            #x=nan_to_num(x,f"stage{i}")
        
        # with autocast():
        x_gate=self.gate(x)
        #x_ln=F.layer_norm(x_gate,[x_gate.shape[-1]])
        x_out=self.head(x_gate)    
            # if torch.isnan(x_out).sum()>0:
            #     warnings.warn("output NaN")
            #x_out=nan_to_num(x_out,"x_out")
        return x_out
    
        
    
# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
#             ]))
#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x
    
    

# class ViT(nn.Module):
#     def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
#         super().__init__()
#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(patch_size)

#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

#         num_patches = (image_height // patch_height) * (image_width // patch_width)
#         patch_dim = channels * patch_height * patch_width
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

#         self.to_patch_embedding = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
#             nn.Linear(patch_dim, dim),
#         )

#         self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

#         self.pool = pool
#         self.to_latent = nn.Identity()

#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )

#     def forward(self, img):
#         x = self.to_patch_embedding(img)
#         b, n, _ = x.shape

#         cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x += self.pos_embedding[:, :(n + 1)]
#         x = self.dropout(x)

#         x = self.transformer(x)

#         x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

#         x = self.to_latent(x)
#         return self.mlp_head(x)