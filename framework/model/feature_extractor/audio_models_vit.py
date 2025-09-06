# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import numpy as np
import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed, Block
from .util.patch_embed import PatchEmbed_new, PatchEmbed3D_new
from typing import Tuple
import torch.nn.functional as F
import math


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, mask_2d=True, use_custom_patch=False, n_progr = 3, n_seq = 256, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
        #del self.norm  # remove the original norm
        self.mask_2d = mask_2d
        self.use_custom_patch = use_custom_patch
        num_heads=12
        depth=12
        mlp_ratio=4
        

        self.n_seq = n_seq
        self.n_progr = n_progr

        self.latent_dim = 128
        self.latent_heads = 1

        self.learnable_prompts_init = nn.Parameter(torch.randn(n_progr*(len(self.blocks)//6),768) * 768**-0.5)
        self.learnable_prompts_progr = nn.ParameterList(nn.Parameter(torch.randn(n_progr,768) * 768**-0.5) for i in range(len(self.blocks)//6))
        
        self.temporal_att_post = nn.ModuleList([nn.Sequential(nn.Linear(self.latent_dim, 768), nn.GELU()) for i in range(len(self.blocks))])
        self.all_gate = nn.ParameterList([nn.Parameter(torch.zeros(1)) for i in range(len(self.blocks))])
       
    def forward_block_pre(self, ii, x):
        B = x.shape[0]
        if ii == 0:
            x = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x, self.learnable_prompts_init.expand(B, -1, -1)), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)        
        x = self.blocks[ii](x)
        return x

    def forward_block_post(self, ii, x, x_t):

        x_t = self.temporal_att_post[ii](x_t)
        x = x + nn.functional.tanh(self.all_gate[ii])* x_t

        if ii % 6 == 0:
            x[:, self.n_seq+1+ii//6*self.n_progr:self.n_seq+1+(ii//6+1)*self.n_progr,:] = x[:,self.n_seq+1+ii//6*self.n_progr:self.n_seq+1+(ii//6+1)*self.n_progr,:] + self.learnable_prompts_progr[ii//6]
        if ii == (len(self.blocks)-1):
            x = self.norm(x)
            return x[:,0]       
        return x

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        #x = x + self.pos_embed[:, 1:, :]
        #cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x, self.learnable_prompts_init.expand(B, -1, -1)), dim=1)
        x = x + self.pos_embed[:, 2:, :]
        x = self.pos_drop(x)        
        features = []
        for ii, blk in enumerate(self.blocks):
            x = blk(x)
            features.append(x) #print('audio shape: ', x.shape)
            if ii != 11:
                x[:,197+ii*3:197+(ii+1)*3,:] = x[:,197+ii*3:197+(ii+1)*3,:] + self.learnable_prompts_progr[ii]
        #x = self.norm(x)
        return x[:,0], features


    def random_masking(self, x, mask_ratio):
        print('IN RANDOM MASKING')
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        print('IN RANDOM MASKING 2d') 
        N, L, D = x.shape  # batch, length, dim
        if self.use_custom_patch:
            # # for AS
            T=101 #64,101
            F=12 #8,12
            # # for ESC
            # T=50
            # F=12 
            # for SPC
            # T=12
            # F=12
        else:
            # ## for AS 
            T=64
            F=8
            # ## for ESC
            #T=32
            #F=8            
            ## for SPC
            # T=8
            # F=8
        
        # mask T
        x = x.reshape(N, T, F, D)
        len_keep_T = int(T * (1 - mask_t_prob))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, D)
        #x_masked = torch.gather(x, dim=1, index=index)
        #x_masked = x_masked.reshape(N,len_keep_T*F,D)
        x = torch.gather(x, dim=1, index=index) # N, len_keep_T(T'), F, D

        # mask F
        #x = x.reshape(N, T, F, D)
        x = x.permute(0,2,1,3) # N T' F D => N F T' D
        len_keep_F = int(F * (1 - mask_f_prob))
        noise = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_F]
        #index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, D)
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_T, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0,2,1,3) # N F' T' D => N T' F' D 
        #x_masked = x_masked.reshape(N,len_keep*T,D)
        x_masked = x_masked.reshape(N,len_keep_F*len_keep_T,D)
            
        return x_masked, None, None


    def forward_features_mask(self, x, mask_t_prob, mask_f_prob):
        B = x.shape[0] #4,1,1024,128
        x = self.patch_embed(x) # 4, 512, 768

        x = x + self.pos_embed[:, 1:, :]
        if self.random_masking_2d:
            x, mask, ids_restore = self.random_masking_2d(x, mask_t_prob, mask_f_prob)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_t_prob)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)        
        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            #print('here', x.shape)
            #outcome = x[:, 0]

        return x #outcome



    # overwrite original timm
    def forward(self, x, v=None, mask_t_prob=0.0, mask_f_prob=0.0):
        if mask_t_prob > 0.0 or mask_f_prob > 0.0:
            x = self.forward_features_mask(x, mask_t_prob=mask_t_prob, mask_f_prob=mask_f_prob)
        else:
            #print('forwarding just features')
            x = self.forward_features(x)
        #x = self.head(x)
        return x



def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model








# generate Audio_MAE
def resize_pos_embed(
        posemb: torch.Tensor,
        posemb_new: torch.Tensor,
        num_prefix_tokens: int = 1,
        gs_new: Tuple[int, int] = (),
        interpolation: str = 'bicubic',
        antialias: bool = False,
        gs_old = None,
) -> torch.Tensor:
    # function from timm
    """ Rescale the grid of position embeddings when loading from state_dict.

    *DEPRECATED* This function is being deprecated in favour of resample_abs_pos_embed

    Adapted from:
        https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    if gs_old is None:
        gs_old = (int(math.sqrt(len(posemb_grid))), int(math.sqrt(len(posemb_grid))))

    if gs_new is None or not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    posemb_grid = posemb_grid.reshape(1, gs_old[0], gs_old[1], -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=interpolation, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb

