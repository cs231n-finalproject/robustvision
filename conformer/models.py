import torch
import torch.nn as nn
from functools import partial

# from timm.models.vision_transformer import VisionTransformer, _cfg

from vision_transformer import VisionTransformer, _cfg
from conformer import Conformer
from transconv import TransConv
from timm.models.registry import register_model


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_med_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=576, depth=12, num_heads=9, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def mae_vit_base_patch16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)        
    return model

@register_model
def mae_vit_large_patch16(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)          
    return model

@register_model
def mae_vit_huge_patch14(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_huge.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model

@register_model
def Conformer_tiny_patch16(pretrained=False, **kwargs):
    model = Conformer(patch_size=16, channel_ratio=1, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def Conformer_small_patch16(pretrained=False, **kwargs):
    model = Conformer(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def Conformer_small_patch32(pretrained=False, **kwargs):
    model = Conformer(patch_size=32, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def Conformer_base_patch16(pretrained=False, **kwargs):
    model = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                      num_heads=9, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model

@register_model
def Transconv_small_patch16(pretrained=False, **kwargs):
    model = TransConv(patch_size=16, channel_ratio=4, embed_dim=768, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        pre_trained_vit = mae_vit_base_patch16(pretrained=True)
        model = TransConv(patch_size=16, channel_ratio=4, embed_dim=768, depth=12,
                        num_heads=6, mlp_ratio=4, qkv_bias=True,
                        pre_trained_vit=pre_trained_vit, finetune=False,
                        **kwargs)
    return model