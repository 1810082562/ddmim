import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
from timm.models.vision_transformer import DropPath, Mlp, trunc_normal_
from timm.models.layers import to_2tuple

class Attention(nn.Module):
    def __init__(self, input_size, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpe=True):
        super().__init__()
        self.input_size = input_size
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * input_size - 1) * (2 * input_size - 1), num_heads)
        ) if rpe else None

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpe_index=None, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if rpe_index is not None:
            S = int(math.sqrt(rpe_index.size(-1)))
            relative_position_bias = self.relative_position_bias_table[rpe_index].view(-1, S, S, self.num_heads)
            relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()
            attn = attn + relative_position_bias
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.float().clamp(min=torch.finfo(torch.float32).min, max=torch.finfo(torch.float32).max)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, inner_patches=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.inner_patches = inner_patches
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        conv_size = [size // inner_patches for size in patch_size]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=conv_size, stride=conv_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        patches_resolution = (H // self.patch_size[0], W // self.patch_size[1])
        num_patches = patches_resolution[0] * patches_resolution[1]
        x = self.proj(x).view(
            B, -1, 
            patches_resolution[0], self.inner_patches, 
            patches_resolution[1], self.inner_patches, 
        ).permute(0, 2, 4, 3, 5, 1).reshape(B, num_patches, self.inner_patches, self.inner_patches, -1)
        if self.norm is not None:
            x = self.norm(x)
        return x
    
    
class PatchMerge(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim * 4)
        self.reduction = nn.Linear(dim * 4, dim * 2, bias=False)
    
    def forward(self, x):
        x0 = x[..., 0::2, 0::2, :] 
        x1 = x[..., 1::2, 0::2, :] 
        x2 = x[..., 0::2, 1::2, :] 
        x3 = x[..., 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x
    
class BlockWithRPE(nn.Module):
    def __init__(self, input_size, dim, num_heads=0., mlp_ratio=4., qkv_bias=True, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., rpe=True, 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # 是否启用自注意力机制，即是否有头注意力机制
        with_attn = num_heads > 0.
        # 第一层归一化操作，用于输入数据的标准化处理
        self.norm1 = norm_layer(dim) if with_attn else None
        # 如果启用了自注意力机制，则创建Attention模块
        self.attn = Attention(
            input_size, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop, rpe=rpe,
        ) if with_attn else None
        # DropPath操作，可以在前向传播中进行随机元素丢弃
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 第二层归一化操作
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, rpe_index=None, mask=None):
        if self.attn is not None:
            x = x + self.drop_path(self.attn(self.norm1(x), rpe_index, mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    
    
class HiViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=512, depths=[2, 2, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, ape=True, rpe=True, patch_norm=True, use_checkpoint=False, 
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.ape = ape
        self.rpe = rpe
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.num_main_blocks = depths[-1]

        embed_dim = embed_dim // 2 ** (self.num_layers - 1)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        Hp, Wp = self.patch_embed.patches_resolution
        assert Hp == Wp

        # absolute position embedding
        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.num_features)
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)
        if rpe:
            coords_h = torch.arange(Hp)
            coords_w = torch.arange(Wp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w])) 
            coords_flatten = torch.flatten(coords, 1) 
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] 
            relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
            relative_coords[:, :, 0] += Hp - 1 
            relative_coords[:, :, 1] += Wp - 1
            relative_coords[:, :, 0] *= 2 * Wp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = iter(x.item() for x in torch.linspace(0, drop_path_rate, sum(depths) + sum(depths[:-1])))

        # build blocks
        self.blocks = nn.ModuleList()
        for stage_depth in depths:
            is_main_stage = (embed_dim == self.num_features)
            nhead = num_heads if is_main_stage else 0
            ratio = mlp_ratio if is_main_stage else stem_mlp_ratio
            # every block not in main stage include two mlp blocks 主要阶段以外的每个块包括两个MLP块
            stage_depth = stage_depth if is_main_stage else stage_depth * 2
            for _ in range(stage_depth):
                self.blocks.append(
                    BlockWithRPE(
                        Hp, embed_dim, nhead, ratio, qkv_bias, qk_scale, 
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=next(dpr), 
                        rpe=rpe, norm_layer=norm_layer,
                    )
                )
            if not is_main_stage:
                self.blocks.append(
                    PatchMerge(embed_dim, norm_layer)
                )
                embed_dim *= 2

        self.fc_norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, ids_keep=None, mask=None, return_hifeat=False):
        B = x.shape[0]
        x = self.patch_embed(x) # 将输入数据进行patch嵌入（转换为patches）
        if ids_keep is not None: # 如果指定了ids_keep，则从输出中选择特定的patches
            x = torch.gather(
                x, dim=1, index=ids_keep[:, :, None, None, None].expand(-1, -1, *x.shape[2:])
            )
        hifeat = [] # 用于存储PatchMerge块输出的特征向量
        for blk in self.blocks[:-self.num_main_blocks]: # 对每个块进行处理
            if isinstance(blk, PatchMerge):
                hifeat.append(x) # 将当前输出添加到hifeat列表中
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)  # 使用checkpointing技术在前向传播中使用内存时保存内存

        x = x[..., 0, 0, :]
        if self.ape:
            pos_embed = self.absolute_pos_embed
            if ids_keep is not None:
                pos_embed = torch.gather(
                    pos_embed.expand(B, -1, -1), 
                    dim=1, 
                    index=ids_keep[:, :, None].expand(-1, -1, pos_embed.shape[2]),
                )
            x += pos_embed
        x = self.pos_drop(x)

        rpe_index = None
        if self.rpe:
            if ids_keep is not None:
                B, L = ids_keep.shape
                rpe_index = self.relative_position_index
                rpe_index = torch.gather(
                    rpe_index[ids_keep, :], dim=-1, index=ids_keep[:, None, :].expand(-1, L, -1)
                ).reshape(B, -1)
            else:
                rpe_index = self.relative_position_index.view(-1)
        
        for blk in self.blocks[-self.num_main_blocks:]:
            x = checkpoint.checkpoint(blk, x, rpe_index, mask) if self.use_checkpoint else blk(x, rpe_index, mask)
        
        return x if not return_hifeat else hifeat + [x]

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(dim=1)
        x = self.fc_norm(x)
        x = self.head(x)
        return x