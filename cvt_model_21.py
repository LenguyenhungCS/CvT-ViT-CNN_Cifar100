from functools import partial
from itertools import repeat
from collections.abc import Iterable

import logging
import os
from collections import OrderedDict

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import DropPath, trunc_normal_

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 **kwargs
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q,
            stride_q, 'linear' if method == 'avg' else method
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv,
            stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self,
                          dim_in,
                          dim_out,
                          kernel_size,
                          padding,
                          stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=False,
                    groups=dim_in
                )),
                ('bn', nn.BatchNorm2d(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'avg':
            proj = nn.Sequential(OrderedDict([
                ('avg', nn.AvgPool2d(
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    ceil_mode=True
                )),
                ('rearrage', Rearrange('b c h w -> b (h w) c')),
            ]))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token and x.shape[1] > h*w:
            cls_token, x = torch.split(x, [1, h*w], 1)
        else:
            cls_token = None

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        if self.with_cls_token and cls_token is not None:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v

    def forward(self, x, h, w):
        if (
            self.conv_proj_q is not None
            or self.conv_proj_k is not None
            or self.conv_proj_v is not None
        ):
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 with_cls_token=True,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 **kwargs):
        super().__init__()
        self.with_cls_token = with_cls_token

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, 
            with_cls_token=self.with_cls_token,
            method=method,
            kernel_size=kernel_size,
            stride_kv=stride_kv,
            stride_q=stride_q,
            padding_kv=padding_kv,
            padding_q=padding_q
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        mlp_hidden_dim = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, h, w):
        res = x
        x = self.norm1(x)
        attn = self.attn(x, h, w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ConvEmbed(nn.Module):
    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        return x, H, W

class VisionTransformer(nn.Module):
    def __init__(self,
                 patch_size=4,
                 patch_stride=4,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=192,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 with_cls_token=True,
                 input_size=32,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 **kwargs):
        super().__init__()
        self.with_cls_token = with_cls_token
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.patch_embed = ConvEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            stride=patch_stride,
            padding=patch_padding,
            norm_layer=norm_layer
        )

        if self.with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=.02)
        else:
            self.cls_token = None

        # Calculate num_patches based on input_size and patch_stride
        if isinstance(input_size, int):
            input_size = to_2tuple(input_size)
        patch_output_size = (
            input_size[0] // patch_stride,
            input_size[1] // patch_stride
        )
        self.num_patches = patch_output_size[0] * patch_output_size[1]

        if self.with_cls_token:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim_in=embed_dim, 
                dim_out=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[i], 
                norm_layer=norm_layer, 
                act_layer=act_layer,
                with_cls_token=self.with_cls_token, 
                method=method,
                kernel_size=kernel_size,
                stride_kv=stride_kv,
                stride_q=stride_q,
                padding_kv=padding_kv,
                padding_q=padding_q
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init = init
        self._init_weights(self)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # Patch Embedding
        x, H, W = self.patch_embed(x) # x is (B, N, C)
        B, N, C = x.shape

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
            
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, H, W) # Pass H, W for attention calculation

        x = self.norm(x)
        # Return all tokens (including CLS if present) and H, W of patch tokens
        return x, H, W 

    def forward(self, x):
        # Simply return all processed tokens and H, W of patch tokens
        # The main ConvolutionalVisionTransformer will decide how to use these.
        return self.forward_features(x) 

class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=100,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        self.num_classes = num_classes
        self.spec = spec

        self.stages = nn.ModuleList()
        self.channel_matchers = nn.ModuleList() # To match dimensions if they change

        current_in_chans = in_chans
        current_input_size = spec.get('INITIAL_INPUT_SIZE', 32) # e.g. 32 for CIFAR

        for i in range(spec['NUM_STAGES']):
            stage_spec = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'in_chans': current_in_chans,
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'], # General QKV bias
                'drop_rate': spec['DROP_RATE'],
                'attn_drop_rate': spec['ATTN_DROP_RATE'],
                'drop_path_rate': spec['DROP_PATH_RATE'], 
                'act_layer': act_layer,
                'norm_layer': norm_layer,
                'init': init,
                'with_cls_token': spec['WITH_CLS_TOKEN'][i],
                'input_size': current_input_size, # H or W of the input feature map to this stage's ConvEmbed
                **spec['ATTN_PARAMS'][i] # Unpack attention specific params
            }
            self.stages.append(VisionTransformer(**stage_spec))

            # Update for the next stage
            # The input channels for the next stage's ConvEmbed is the embed_dim of the current stage's tokens
            current_in_chans = spec['DIM_EMBED'][i]
            # The input H/W for the next stage's ConvEmbed is the output H/W of current stage's ConvEmbed
            current_input_size = current_input_size // spec['PATCH_STRIDE'][i]

            # Add channel matcher if embed_dim changes for the *next* stage's input
            if i < spec['NUM_STAGES'] - 1:
                if spec['DIM_EMBED'][i] != spec['DIM_EMBED'][i+1]:
                    # This matcher will operate on the token sequence from the current stage
                    # before it's reshaped and fed to the next stage's ConvEmbed.
                    # However, ConvEmbed takes C_in from the image/feature map. So, matcher is tricky here.
                    # For CvT, ConvEmbed itself handles projection to a new dim.
                    # The 'in_chans' for the next ConvEmbed is set to current_in_chans (which is current embed_dim).
                    # So, direct channel matcher between VisionTransformer token outputs is not standard CvT way.
                    # CvT's ConvEmbed projects from C_prev_embed to C_curr_embed.
                    # Let's assume ConvEmbed handles the dimension change via its in_chans and embed_dim.
                    # No explicit channel_matcher module list needed here if ConvEmbed handles projection.
                    pass # No explicit channel matcher here for now, ConvEmbed handles this.

        # Final normalization layer (applied to tokens from the last stage)
        self.norm = norm_layer(spec['DIM_EMBED'][-1])
        self.head = nn.Linear(spec['DIM_EMBED'][-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        
        # Iterate through stages dynamically
        for i in range(self.spec['NUM_STAGES']):
            stage = self.stages[i]
            # Input x is either the initial image (for stage 0) or the feature map from the previous stage
            x_tokens, H, W = stage(x) # Each stage is a VisionTransformer, outputs all tokens + H, W of patch tokens

            # If it's not the last stage, we need to prepare x for the next stage's ConvEmbed
            if i < self.spec['NUM_STAGES'] - 1:
                if stage.with_cls_token:
                    # Remove CLS token before reshaping patch tokens into a feature map
                    # The next stage will add its own CLS token if configured
                    patch_tokens = x_tokens[:, 1:]
                else:
                    patch_tokens = x_tokens
                
                # Reshape patch_tokens to be a 4D feature map for the next stage
                # The channel dimension for this map is the embed_dim of the current stage
                current_embed_dim = self.spec['DIM_EMBED'][i]
                x = rearrange(patch_tokens, 'b (h w) c -> b c h w', h=H, w=W, c=current_embed_dim)
            else:
                # For the last stage, x_tokens are the final output tokens
                x = x_tokens 

        # Apply final norm (if it wasn't part of the last stage, though typically it is)
        # self.norm here is the final norm for the whole CVT model, applied to the tokens from the last stage.
        x = self.norm(x)
        return x # Return all tokens from the last stage

    def forward(self, x):
        x = self.forward_features(x)
        # The head expects the CLS token, which should be x[:, 0] if the last stage used with_cls_token=True
        return self.head(x[:, 0])

CVT_SPEC = {
    'cvt-cifar-custom': { # Reverting to CvT-21 like scale, adapted for CIFAR-100
        'INIT': 'trunc_norm',
        'NUM_STAGES': 3,
        'PATCH_SIZE': [3, 2, 2],
        'PATCH_STRIDE': [2, 2, 2],
        'PATCH_PADDING': [1, 0, 0],
        'DIM_EMBED': [64, 192, 384],  # CvT-21 like embed dimensions
        'NUM_HEADS': [1, 3, 6],     # CvT-21 like number of heads
        'DEPTH': [1, 4, 16],         # CvT-21 like depth (1+4+16 = 21 layers)
        'MLP_RATIO': [4.0, 4.0, 4.0],
        'ATTN_PARAMS': [
            {'method': 'dw_bn', 'kernel_size': 3, 'padding_kv': 1, 'padding_q': 1, 'stride_kv': 1, 'stride_q': 1, 'qkv_bias': True, 'attn_drop': 0.1, 'proj_drop': 0.1},
            {'method': 'dw_bn', 'kernel_size': 3, 'padding_kv': 1, 'padding_q': 1, 'stride_kv': 1, 'stride_q': 1, 'qkv_bias': True, 'attn_drop': 0.1, 'proj_drop': 0.1},
            {'method': 'dw_bn', 'kernel_size': 3, 'padding_kv': 1, 'padding_q': 1, 'stride_kv': 1, 'stride_q': 1, 'qkv_bias': True, 'attn_drop': 0.1, 'proj_drop': 0.1},
        ],
        'WITH_CLS_TOKEN': [False, False, True],
        'QKV_BIAS': True,
        'DROP_RATE': 0.1,
        'ATTN_DROP_RATE': 0.1,
        'DROP_PATH_RATE': 0.1,
        'INPUT_SIZE_STAGE': [32, 16, 8]
    }
}

def get_cvt_model(num_classes=100, model_name='cvt-cifar-custom'):
    spec = CVT_SPEC[model_name]
    model = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=num_classes,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init=spec['INIT'],
        spec=spec
    )
    return model 