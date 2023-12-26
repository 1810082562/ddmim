# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import torch
import torch.nn as nn


class MaskedAutoencoder(nn.Module):
    def __init__(self):
        """
        构造函数,初始化MaskedAutoencoder类的实例。
        """
        nn.Module.__init__(self)
        self.norm_pix_loss = True
    
    def patchify(self, imgs):
        """
        将图像划分为补丁(patch)，并将其重新排列成特定的形状。
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.decoder_patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        将经过划分和排列的补丁(x)恢复为原始图像。
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.decoder_patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def masking_id(self, batch_size, mask_ratio):
        """
        生成用于掩码的标识符(ids_keep)、恢复标识符(ids_restore)和二进制掩码(mask)。掩码比例由mask_ratio确定。

        Args:
            batch_size (_type_): _description_
            mask_ratio (_type_): _description_

        Returns:
            _type_: _description_
        """
        N, L = batch_size, self.patch_embed.num_patches
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=self.pos_embed.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=self.pos_embed.device)
        mask[:, :ids_keep.size(1)] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, ids_restore, mask

    def random_masking(self, x, ids_keep):
        """根据给定的ids_keep，对输入数据x进行随机掩码操作。

        Args:
            x (_type_): _description_
            ids_keep (_type_): _description_

        Returns:
            _type_: _description_
        """
        N, L, D = x.shape
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked

    def forward_encoder(self, x, mask_ratio):
        """编码器的前向传播函数，根据输入数据x生成潜变量(latent)、掩码信息(mask)和恢复标识符(ids_restore)。

        Args:
            x (_type_): _description_
            mask_ratio (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def forward_decoder(self, x, ids_restore):
        """ 解码器的前向传播函数，根据输入的潜变量x和恢复标识符(ids_restore)生成分类预测(cls_pred)和解码重构结果(pred)。

        Args:
            x (_type_): _description_
            ids_restore (_type_): _description_

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    def forward_loss(self, imgs, cls_pred, pred, mask):
        """
        计算前向传播中的损失函数。根据原始图像(imgs)、分类预测(cls_pred)、解码重构结果(pred)和掩码(mask)计算损失值。
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        num_preds = mask.sum()
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / num_preds
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        """模型的完整前向传播过程。根据输入图像(imgs)和掩码比例(mask_ratio)，生成损失值(loss)、解码重构结果(pred)和掩码(mask)。

        Args:
            imgs (_type_): _description_
            mask_ratio (float, optional): _description_. Defaults to 0.75.

        Returns:
            _type_: _description_
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        cls_pred, pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, cls_pred, pred, mask)
        return loss, pred, mask
    
    
    
    
    
