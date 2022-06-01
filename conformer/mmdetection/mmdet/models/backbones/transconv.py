from mimetypes import init
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class AdditiveAttention(nn.Module):
    ''' AttentionPooling used to weighted aggregate news vectors
    Arg: 
        d_h: the last dimension of input
    '''
    def __init__(self, d_h, hidden_size=5):
        super(AdditiveAttention, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, candidate_size, candidate_vector_dim
            attn_mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  # (bz, 400)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None, additive_fusion=False, up_shape=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.additive_fusion = additive_fusion
        if self.additive_fusion:
            d_h = up_shape[1]*up_shape[2]*up_shape[3]
            d_e = 10
            self.additive_attention = AdditiveAttention(d_h, d_e)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def additive(self, x, x_t):
        if self.additive_fusion:
            B, C, H, W = x.shape
            x_stacked = torch.stack([x, x_t], dim=1).flatten(2)
            x_fused = self.additive_attention(x_stacked)
            x_fused = x_fused.reshape((B, C, H, W))
        else:
            x_fused = x + x_t
        return x_fused

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(self.additive(x, x_t))
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        # [N, C, H, W], [N, 64, 56, 56] -> [N, 384, 56, 56]
        x = self.conv_project(x)
        # [16, 384, 56, 56] -> [N, 384, 14, 14] -> [N, 384, 196] -> [N, 196, 384]
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)       
        x = self.act(self.ln(x))
        # class_token of x_t [N, 384] -> [N, 1, 384] -> + [N, 196, 384] -> [N, 197, 384]
        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class FCUUp(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        H_x_r = W_x_r = int(math.sqrt(x.shape[1]-1))
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H_x_r, W_x_r)
        # [N, 384, 14, 14] ->  [N, 64, 14, 14]
        x_r = self.act(self.bn(self.conv_project(x_r)))
        # [N, 64, 14, 14] -> [N, 64, 56, 56]
        return F.interpolate(x_r, size=(H, W))


class Med_ConvBlock(nn.Module):
    """ special case for Convblock with down sampling,
    """
    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x


class ConvTransBlock(nn.Module):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1,
                 pre_trans_block=None, pre_convtrans_block=None, finetune_vit=False, finetune_conv=True,
                 additive_fusion_down=False, down_shape=None, additive_fusion_up=False, up_shape=None):

        super(ConvTransBlock, self).__init__()
        self.has_pre_trans_block = True if pre_trans_block is not None else False
        self.has_pre_conv_block = True if pre_convtrans_block is not None else False
        self.finetune_vit = finetune_vit
        self.finetune_conv = finetune_conv
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups) \
            if pre_convtrans_block is None else pre_convtrans_block.cnn_block

        # note that the fusion block is never freezed, however, weight can be initialized
        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups, \
                additive_fusion=additive_fusion_up, up_shape=up_shape)
            if pre_convtrans_block is not None:
                self.fusion_block.load_state_dict(pre_convtrans_block.fusion_block.state_dict(), strict=False)
                self.fusion_block.apply(flag_pretrain)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups, \
                additive_fusion=additive_fusion_up, up_shape=up_shape)
            if pre_convtrans_block is not None:
                    self.fusion_block.load_state_dict(pre_convtrans_block.fusion_block.state_dict(), strict=False)
                    self.fusion_block.apply(flag_pretrain)

        if num_med_block > 0:
            if pre_convtrans_block is None:
                self.med_block = []
                for i in range(num_med_block):
                    self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
                self.med_block = nn.ModuleList(self.med_block)
            else:
                self.med_block = pre_convtrans_block.med_block

        # only when transformer tower is not initialized from ViT and conv tower is from conformer, go with conformer
        if pre_convtrans_block is not None and pre_trans_block is None:
            self.squeeze_block = pre_convtrans_block.squeeze_block
        else:
            self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        # only when transformer tower is not initialized from ViT and conv tower is from conformer, go with conformer
        if pre_convtrans_block is not None and pre_trans_block is None:
            self.expand_block = pre_convtrans_block.expand_block
        else:
            self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        if self.has_pre_trans_block:
            self.trans_block = pre_trans_block
        else:
            self.trans_block = TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate) \
                    if pre_convtrans_block is None else pre_convtrans_block.trans_block

        self.additive_fusion = additive_fusion_down
        if self.additive_fusion:
            d_h = down_shape[1] * down_shape[2]
            d_e = 5
            self.additive_attention = AdditiveAttention(d_h, d_e)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def additive(self, x_st, x_t):
        if self.additive_fusion:
            B, P, E = x_t.shape
            x_t_stacked = torch.stack([x_st, x_t], dim=1).flatten(2)
            x_t_fused = self.additive_attention(x_t_stacked)
            x_t_fused = x_t_fused.reshape((B, P, E))
        else:
            x_t_fused = x_st + x_t
        return x_t_fused

    def forward(self, x, x_t):

        if self.has_pre_trans_block and not self.finetune_vit:
            x = self.cnn_block(x, return_x_2=False)           
            x_t = self.trans_block(x_t)
        else:
            x, x2 = self.cnn_block(x)           
            x_st = self.squeeze_block(x2, x_t)
            x_t = self.trans_block(self.additive(x_st, x_t))
        _, _, H, W = x.shape  

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        if self.has_pre_conv_block and not self.finetune_conv:
            #TODO design choice, fuse or not if conv tower does not allow finetune, Yes for now
            x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
            x = self.fusion_block(x, x_t_r, return_x_2=False)
            # don't fuse with transformer block            
            # x = self.fusion_block(x, None, return_x_2=False)
        else:
            # fuse with transformer block
            # [N, 197, 384] -> [N, 64, 56, 56]
            x_t_r = self.expand_block(x_t, H, W)
            # [N, 256, 56, 56] + [N, 64, 56, 56] -> [N, 256, 56, 56]
            x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t


class TransConv(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False,
                 pre_trained_vit=None, finetune_vit=False, vit_depth=12, pre_trained_conformer=None, finetune_conv=True,
                 additive_fusion_down=False, additive_fusion_up=False, up_ftr_map_size=None, down_ftr_map_size=None,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):

        # Transformer
        super().__init__()

        self.has_pre_trained_vit = True if pre_trained_vit is not None else False
        self.has_pre_trained_conformer = True if pre_trained_conformer is not None else False
        if pre_trained_vit is not None:
            pre_trained_vit.apply(flag_pretrain)
            if finetune_vit is False:
                freeze(pre_trained_vit)
            self.pre_trained_vit = pre_trained_vit
            self.vit_final_stage = vit_depth + 1
        if pre_trained_conformer is not None:
            pre_trained_conformer.apply(flag_pretrain)
            if finetune_conv is False:
                freeze(pre_trained_conformer)

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        assert depth % 3 == 0

        if pre_trained_vit is None:
            if pre_trained_conformer is None:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
                trunc_normal_(self.cls_token, std=.02)
            else:
                self.cls_token = flag_pretrain(pre_trained_conformer.cls_token)
        else:
            self.cls_token = pre_trained_vit.cls_token
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # Classifier head for transformer tower
        if pre_trained_vit is None:
            self.trans_norm = nn.LayerNorm(embed_dim) if pre_trained_conformer is None else pre_trained_conformer.trans_norm
        else:
            self.global_pool = None if pre_trained_vit is None else pre_trained_vit.global_pool
            if self.global_pool:
                self.trans_norm = pre_trained_vit.fc_norm
            else:
                self.trans_norm = pre_trained_vit.norm
        if pre_trained_vit is None:
            if pre_trained_conformer is None:
                self.trans_cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            else:
                self.trans_cls_head = pre_trained_conformer.trans_cls_head
        else:
            self.trans_cls_head = pre_trained_vit.head
        # Classifier head for conv tower
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes) if pre_trained_conformer is None else pre_trained_conformer.conv_cls_head

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False) if pre_trained_conformer is None else pre_trained_conformer.conv1  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64) if pre_trained_conformer is None else pre_trained_conformer.bn1
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 1 / 4 [56, 56]

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1) if pre_trained_conformer is None else pre_trained_conformer.conv_1
        if pre_trained_vit is not None:
            self.pos_drop = pre_trained_vit.pos_drop
            self.pos_embed = pre_trained_vit.pos_embed
            self.trans_patch_conv = pre_trained_vit.patch_embed
            self.trans_1 = pre_trained_vit.blocks[0] if pre_trained_vit is not None else None
        else:
            self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0) if pre_trained_conformer is None else pre_trained_conformer.trans_patch_conv
            self.trans_1 = TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0],
                                ) if pre_trained_conformer is None else pre_trained_conformer.trans_1

        expansion = 4
        # 2~4 stage
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block,
                        pre_trans_block=pre_trained_vit.blocks[i-1] if pre_trained_vit is not None else None,
                        pre_convtrans_block=eval('pre_trained_conformer.conv_trans_' + str(i)) if pre_trained_conformer is not None else None,
                        finetune_vit=finetune_vit, finetune_conv=finetune_conv,
                        additive_fusion_down=additive_fusion_down, down_shape=(None, down_ftr_map_size[i-1], embed_dim), additive_fusion_up=additive_fusion_up,
                        up_shape=(None, stage_1_channel // expansion, up_ftr_map_size[i-1], up_ftr_map_size[i-1])
                    )
            )

        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage
        init_stage = fin_stage # 5
        fin_stage = fin_stage + depth // 3 # 9     
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block,
                        pre_trans_block=pre_trained_vit.blocks[i-1] if pre_trained_vit is not None else None,
                        pre_convtrans_block=eval('pre_trained_conformer.conv_trans_' + str(i)) if pre_trained_conformer is not None else None,
                        finetune_vit=finetune_vit, finetune_conv=finetune_conv,
                        additive_fusion_down=additive_fusion_down, down_shape=(None, down_ftr_map_size[i-1], embed_dim), additive_fusion_up=additive_fusion_up,
                        up_shape=(None, stage_2_channel // expansion, up_ftr_map_size[i-1], up_ftr_map_size[i-1])
                    )
            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13      
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                        num_med_block=num_med_block, last_fusion=last_fusion,
                        pre_trans_block=pre_trained_vit.blocks[i-1] if pre_trained_vit is not None else None,
                        pre_convtrans_block=eval('pre_trained_conformer.conv_trans_' + str(i)) if pre_trained_conformer is not None else None,
                        finetune_vit=finetune_vit, finetune_conv=finetune_conv,
                        additive_fusion_down=additive_fusion_down, down_shape=(None, down_ftr_map_size[i-1], embed_dim), additive_fusion_up=additive_fusion_up,
                        up_shape=(None, stage_3_channel // expansion, up_ftr_map_size[i-1], up_ftr_map_size[i-1])
                    )
            )
        self.fin_stage = fin_stage

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # don't initialize weight if it is initialized by pretrained model!
        if is_pretrain(m):
            return
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}


    def forward(self, x, monitor=False, writer=None, global_step=None):
        B = x.shape[0]

        # pdb.set_trace()
        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        # 1 stage
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.has_pre_trained_vit:
            x_t = self.trans_patch_conv(x)
            x_t = torch.cat((cls_tokens, x_t), dim=1)
            x_t = x_t + self.pos_embed
            x_t = self.pos_drop(x_t)
        else:
            x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
            x_t = torch.cat([cls_tokens, x_t], dim=1)
        x = self.conv_1(x_base, return_x_2=False)
        x_t = self.trans_1(x_t)

        # 2 ~ final 
        for i in range(2, self.fin_stage):         
            x, x_t = eval('self.conv_trans_' + str(i))(x, x_t)
            if monitor:
                log_ftr_map_histograms(writer, x, x_t, i, global_step)            

        # final ~ vit_final
        for i in range(self.fin_stage, self.vit_final_stage):         
            x_t = self.pre_trained_vit.blocks[i-1](x_t)

        # conv classification
        x_p = self.pooling(x).flatten(1)
        conv_cls = self.conv_cls_head(x_p)

        # trans classification
        if self.has_pre_trained_vit:        
            if self.global_pool:
                x_t = x_t[:, 1:, :].mean(dim=1)  # global pool without cls token
                x_t = self.trans_norm(x_t)
            else:
                x_t = self.trans_norm(x_t)
                x_t = x_t[:, 0]
        else:
            x_t = self.trans_norm(x_t)
            x_t = x_t[:, 0]
        tran_cls = self.trans_cls_head(x_t)

        return [conv_cls, tran_cls]


def log_ftr_map_histograms(writer, x, x_t, i, global_step):
    flattened_x = x.flatten()
    flattened_x_t = x_t.flatten()
    writer.add_histogram(f"conv_trans_{i}.feature_map.conv_tower", flattened_x, global_step=global_step, bins='tensorflow')
    writer.add_histogram(f"conv_trans_{i}.feature_map.trans_tower", flattened_x_t, global_step=global_step, bins='tensorflow')


def freeze(module):
    for p in module.parameters():
        p.requires_grad = False


def flag_pretrain(module, pretrained=True):
    setattr(module, "pretrain", pretrained)
    return module


def is_pretrain(module):
    is_pretrain = hasattr(module, 'pretrain') and module.pretrain
    return is_pretrain
