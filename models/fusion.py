"""
Implementation of different fusion modules.
Here are the name mappings (class name in the code -> fuser name in the paper (https://arxiv.org/abs/2210.12649)):
    MATT: MATT fusion module from RULSTM (https://arxiv.org/abs/1905.09035)
    CMFuser: SA-Fuser without modality token
    TemporalCMFuser: T-SA-Fuser
    TemporalCrossAttentFuser: CA-Fuser
    ModalTokenCMFuser: SA-Fuser with modality token
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from typing import Dict, Callable, Tuple
from functools import partial
from torch import Tensor

from models.transformerblock import Block, DecoderBlock


def _init_weights(m):
    # Copied from Timm VisionTransformer,
    # removing init for layernorm, since this init is already the default for pytorch
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)


def generate_square_subsequent_mask(sz: int):
    """from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html"""
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


class MATT(nn.Module):
    """modality attention module from RULSTM, an MLP with 3 layers"""
    def __init__(self, modal_dims, dim=None, drop_rate=0.8):
        super().__init__()
        num_modality = len(modal_dims)
        in_size = dim * num_modality if dim else sum(modal_dims.values())
        self.matt = nn.Sequential(nn.Linear(in_size, int(in_size / 4)),
                                  nn.ReLU(),
                                  nn.Dropout(drop_rate),
                                  nn.Linear(int(in_size / 4), int(in_size / 8)),
                                  nn.ReLU(),
                                  nn.Dropout(drop_rate),
                                  nn.Linear(int(in_size / 8), num_modality))

    def forward(self, modal_feats: Dict[str, Tensor], ordered_feature_list: Callable) -> Tensor:
        """
        :param modal_feats: {'modality': feature vector}
        :return: modality weights, used to weight the classification scores
        """
        # list of modality feats in given order
        for_fusion_feats = ordered_feature_list(modal_feats)
        for_fusion_feats = torch.cat(for_fusion_feats, dim=2)
        modality_attns = self.matt(for_fusion_feats).softmax(dim=-1)
        return modality_attns


class CMFuser(nn.Module):
    """Corresponds to SA-Fuser without modality token in the paper."""
    def __init__(self, dim, depth=1, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, embd_drop_rate=0.,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), cross_attn=False):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.norm = norm_layer(dim)
        self.embd_drop = nn.Dropout(embd_drop_rate)
        self.cross_attn = cross_attn
        self.apply(_init_weights)

    @staticmethod
    def generate_cross_attention_mask(sz):
        mask = torch.eye(sz)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, modal_feats: Dict[str, Tensor], ordered_feature_list: Callable) -> Tuple[Tensor, Tensor]:
        """
        :param modal_feats: {'modality': feature vector}
        :return: fused feature
        """
        shape = next(iter(modal_feats.values())).shape  # get the shape of the first modality
        assert all([v.shape == shape for v in modal_feats.values()]), \
            'The shape of all inputs of the fusion module should be the same!'

        B, T, C = shape
        attn_mask = None
        num_modality = len(modal_feats.keys())
        if self.cross_attn:
            attn_mask = self.generate_cross_attention_mask(num_modality).to('cuda')

        # list of modality feats in given order
        for_fusion_feats = ordered_feature_list(modal_feats)

        # n * (B, T, C) -> (B*T, n, C)
        for_fusion_feats = torch.cat([fe.reshape(B * T, 1, -1) for fe in for_fusion_feats], dim=1)

        # fusion part
        x = self.embd_drop(for_fusion_feats)
        attn_weights = []
        for blk in self.blocks:
            x, attn_weight = blk(x, attn_mask)
            attn_weights.append(attn_weight.view(B, T, *attn_weight.shape[1:]))

        x = self.norm(x)
        feats_fusion = torch.mean(x, dim=1)
        feats_fusion = feats_fusion.view(-1, T, C)

        return feats_fusion, torch.stack(attn_weights).transpose(0, 1)


class TemporalCMFuser(nn.Module):
    """Corresponds to T-SA-Fuser in the paper.
    This module performs temporal (causal) and multi-modal attention at the same time"""
    def __init__(self, dim, depth=1, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, embd_drop_rate=0.,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), modalities=None, modal_encoding=True,
                 frame_level_token=False, temporal_sequence_length=None, max_position_embeddings=64):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.norm = norm_layer(dim)

        # frame position embedding and modality embedding
        self.num_mods = len(modalities) + 1 if frame_level_token else len(modalities)
        self.modality_embedding = nn.Parameter(torch.zeros(self.num_mods, dim)) if modal_encoding else None
        self.position_embeddings = nn.Embedding(max_position_embeddings, dim)

        self.embd_drop = nn.Dropout(embd_drop_rate)
        self.frame_level_token = frame_level_token
        self.temporal_sequence_length = temporal_sequence_length

        # modality agnostic token
        self.modal_token = None
        if frame_level_token:
            assert temporal_sequence_length is not None, "Temporal sequence length must be provided!"
            self.modal_token = nn.Parameter(torch.zeros(1, temporal_sequence_length, dim))

        if self.modal_token is not None:
            trunc_normal_(self.modal_token, std=.02)
        if self.modality_embedding is not None:
            trunc_normal_(self.modality_embedding, std=.02)
        self.apply(_init_weights)

    def forward(self, modal_feats: Dict[str, Tensor], ordered_feature_list: Callable) -> Tuple[Tensor, Tensor]:
        """
        :param modal_feats: {'modality': feature vector}
        :return: fused feature
        """
        shape = next(iter(modal_feats.values())).shape  # get the shape of the first modality
        assert all([v.shape == shape for v in modal_feats.values()]), \
            'The shape of all inputs of the fusion module should be the same!'

        # compute causal mask
        B, T, C = shape
        causal_mask = generate_square_subsequent_mask(T).to('cuda')
        causal_modality_mask = causal_mask.repeat(self.num_mods, self.num_mods)

        # list of modality feats in given order
        for_fusion_feats = ordered_feature_list(modal_feats)

        # n * (B, T, C) -> (B, n*T, C)
        for_fusion_feats = torch.cat([fe for fe in for_fusion_feats], dim=1)

        # prepare and prepend modal token if required
        if self.frame_level_token:
            assert self.temporal_sequence_length == T, \
                f"Temporal sequence length not valid {self.temporal_sequence_length} vs {T}"
            modal_tokens = self.modal_token.expand(B, -1, -1)
            for_fusion_feats = torch.cat((modal_tokens, for_fusion_feats), dim=1)

        # add position embedding
        position_ids = torch.arange(T, dtype=torch.long, device='cuda')
        position_embeds = self.position_embeddings(position_ids)
        position_embeds_final = position_embeds.repeat(self.num_mods, 1).expand(B, -1, -1)
        for_fusion_feats += position_embeds_final

        # add modality embedding if required
        if self.modality_embedding is not None:
            modality_embeds = [embeds.repeat(T, 1) for embeds in self.modality_embedding]
            modality_embeds_final = torch.cat(modality_embeds, dim=0).expand(B, -1, -1)
            for_fusion_feats += modality_embeds_final

        # fusion part
        x = self.embd_drop(for_fusion_feats)
        attn_weights = []
        for blk in self.blocks:
            x, attn_weight = blk(x, causal_modality_mask)
            attn_weights.append(attn_weight)

        x = self.norm(x)

        if self.frame_level_token:
            # We select the outputs of frame leval modal tokens as the fused outputs
            feats_fusion = x[:, :T, :]
        else:
            # We average the outputs of frame level multi-modal tokens as the fused outputs
            ranges = [range(i, x.size(1), T) for i in range(T)]
            feats_fusion = [torch.mean(x[:, r, :], dim=1, keepdim=True) for r in ranges]
            feats_fusion = torch.cat(feats_fusion, dim=1)
        return feats_fusion, torch.stack(attn_weights).transpose(0, 1)


class TemporalCrossAttentFuser(nn.Module):
    """Corresponds to CA-Fuser in the paper.
    This module uses rgb as the main modality and extracts information from other modalities.
    The depth of this module corresponds to the number of modalities minus 1."""
    def __init__(self, dim, modalities=None,
                 num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, embd_drop_rate=0.,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 max_position_embeddings=128):
        super().__init__()

        depth = len(modalities) - 1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            DecoderBlock(
                dim=dim, mem_dim=None, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.norm = norm_layer(dim)
        self.embd_drop = nn.Dropout(embd_drop_rate)
        self.position_embeddings = nn.Embedding(max_position_embeddings, dim)
        self.apply(_init_weights)

    def forward(self, modal_feats: Dict[str, Tensor], ordered_feature_list: Callable) -> Tuple[Tensor, Tensor]:
        """
        :param modal_feats: {'modality': feature vector}
        :return: fused feature
        """
        shape = next(iter(modal_feats.values())).shape  # get the shape of the first modality
        assert all([v.shape == shape for v in modal_feats.values()]), \
            'The shape of all inputs of the fusion module should be the same!'

        # compute causal mask and positional embeddings
        B, T, C = shape
        causal_mask = generate_square_subsequent_mask(T).to('cuda')
        position_ids = torch.arange(T, dtype=torch.long, device='cuda')
        position_embeds = self.position_embeddings(position_ids)

        # list of modality feats in given order
        for_fusion_feats = ordered_feature_list(modal_feats)

        # fusion part
        for_fusion_feats = [self.embd_drop(feat + position_embeds) for feat in for_fusion_feats]
        # we take the first modality (usually rgb) as main modality and the rest as mems
        x, mems = for_fusion_feats[0], for_fusion_feats[1:]

        for i, blk in enumerate(self.blocks):
            x = blk(x, mems[i], causal_mask)
        x = self.norm(x)
        dummy_attention = torch.zeros(B, requires_grad=False)  # to satisfy the framework
        return x, dummy_attention


class ModalTokenCMFuser(nn.Module):
    """Corresponds to SA-Fuser with modality token in the paper"""
    def __init__(self, dim, depth=1, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, embd_drop_rate=0.,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU,
                 norm_elementwise=True, cross_attn=False, modalities=None, modal_encoding=False,
                 frame_level_token=False, temporal_sequence_length=None):
        super().__init__()
        # move norm_layer here, so that elementwise affine can be controlled by hydra
        norm_layer = partial(nn.LayerNorm, eps=1e-6, elementwise_affine=norm_elementwise)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.norm = norm_layer(dim)

        self.num_mods = len(modalities) + 1  # add 1 since we add a modal token
        self.modality_embedding = nn.Parameter(torch.zeros(1, self.num_mods, dim)) if modal_encoding else None
        self.embd_drop = nn.Dropout(embd_drop_rate)
        self.cross_attn = cross_attn
        self.frame_level_token = frame_level_token
        self.temporal_sequence_length = temporal_sequence_length

        # modality agnostic token
        if not frame_level_token:
            # We use a universal token for all temporal sequences
            self.modal_token = nn.Parameter(torch.zeros(1, 1, dim))
        else:
            # We use an individual token for different sequences
            assert temporal_sequence_length is not None, "Temporal sequence length must be provided!"
            self.modal_token = nn.Parameter(torch.zeros(1, temporal_sequence_length, dim))

        trunc_normal_(self.modal_token, std=.02)
        if self.modality_embedding is not None:
            trunc_normal_(self.modality_embedding, std=.02)
        self.apply(_init_weights)

    @staticmethod
    def generate_cross_attention_mask(sz):
        mask = torch.eye(sz)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, modal_feats: Dict[str, Tensor], ordered_feature_list: Callable) -> Tuple[Tensor, Tensor]:
        """
        :param modal_feats: {'modality': feature vector}
        :param ordered_feature_list: a function to convert a dict to a list with a specific order
        :return: fused feature
        """
        shape = next(iter(modal_feats.values())).shape  # get the shape of the first modality
        assert all([v.shape == shape for v in modal_feats.values()]), \
            'The shape of all inputs of the fusion module should be the same!'

        B, T, C = shape
        attn_mask = None
        if self.cross_attn:
            attn_mask = self.generate_cross_attention_mask(self.num_mods).to('cuda')

        # list of modality feats in given order
        for_fusion_feats = ordered_feature_list(modal_feats)

        # n * (B, T, C) -> (B*T, n, C)
        for_fusion_feats = torch.cat([fe.reshape(B * T, 1, -1) for fe in for_fusion_feats], dim=1)

        # prepare modal token
        if not self.frame_level_token:
            modal_tokens = self.modal_token.expand(B * T, -1, -1)
        else:
            assert self.temporal_sequence_length == T, \
                f"Temporal sequence length not valid {self.temporal_sequence_length} vs {T}"
            modal_tokens = self.modal_token.expand(B, -1, -1).reshape(B * T, 1, -1)

        # prepend the modality agnostic token
        for_fusion_feats = torch.cat((modal_tokens, for_fusion_feats), dim=1)

        # add modality embedding if required
        if self.modality_embedding is not None:
            for_fusion_feats += self.modality_embedding

        # fusion part
        x = self.embd_drop(for_fusion_feats)
        attn_weights = []
        for blk in self.blocks:
            x, attn_weight = blk(x, attn_mask)
            attn_weights.append(attn_weight.view(B, T, *attn_weight.shape[1:]))

        x = self.norm(x)
        feats_fusion = x[:, 0, :]  # we only use the output of the modality agnostic token
        feats_fusion = feats_fusion.view(-1, T, C)
        return feats_fusion, torch.stack(attn_weights).transpose(0, 1)
