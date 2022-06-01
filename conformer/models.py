import os
import torch
import torch.nn as nn
from functools import partial

# from timm.models.vision_transformer import VisionTransformer, _cfg

from vision_transformer import VisionTransformer, _cfg
from mae.models_vit import VisionTransformer as MAE_VisionTransformer
from conformer import Conformer
from transconv import TransConv
from utils import load_pretrain_model
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
    model = MAE_VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=1000, drop_path_rate=0.1,
        global_pool=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth",
        #     map_location="cpu", check_hash=True
        # )
        # model.load_state_dict(checkpoint["model"], strict=False)
        model = load_pretrain_model(model, model_path=os.path.expanduser('~/robustvision/conformer/mae/checkpoints/mae_finetuned_vit_base.pth'))
    return model

@register_model
def mae_vit_large_patch16(pretrained=False, **kwargs):
    model = MAE_VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  num_classes=1000, drop_path_rate=0.1,
        global_pool=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth",
        #     map_location="cpu", check_hash=True
        # )
        # model.load_state_dict(checkpoint["model"], strict=False)
        model = load_pretrain_model(model, model_path=os.path.expanduser('~/robustvision/conformer/mae/checkpoints/mae_finetuned_vit_large.pth'))
    return model

@register_model
def mae_vit_huge_patch14(pretrained=False, **kwargs):
    model = MAE_VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  num_classes=1000, drop_path_rate=0.1,
        global_pool=True, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_huge.pth",
        #     map_location="cpu", check_hash=True
        # )
        # model.load_state_dict(checkpoint["model"], strict=False)
        model = load_pretrain_model(model, model_path=os.path.expanduser('~/robustvision/conformer/mae/checkpoints/mae_finetuned_vit_huge.pth'))     
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
        model = load_pretrain_model(model, model_path=os.path.expanduser('~/robustvision/conformer/mmdetection/pretrain_models/Conformer_small_patch16.pth'))
    return model

@register_model
def Conformer_small_patch32(pretrained=False, **kwargs):
    model = Conformer(patch_size=32, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        model = load_pretrain_model(model, model_path=os.path.expanduser('~/robustvision/conformer/mmdetection/pretrain_models/Conformer_small_patch32.pth'))
    return model

@register_model
def Conformer_base_patch16(pretrained=False, **kwargs):
    model = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                      num_heads=9, mlp_ratio=4, qkv_bias=True, **kwargs)
    if pretrained:
        model = load_pretrain_model(model, model_path=os.path.expanduser('~/robustvision/conformer/mmdetection/pretrain_models/Conformer_base_patch16.pth'))
    return model

@register_model
def Transconv_small_patch16(pretrained=False, **kwargs):
    if pretrained:
        pre_trained_vit = mae_vit_base_patch16(pretrained=True)
        # pre_trained_vit = None  # uncomment to initialize only conv tower
        pre_trained_conformer = Conformer_small_patch16(pretrained=True)        
        model = TransConv(patch_size=16, channel_ratio=4, embed_dim=768 if pre_trained_vit is not None else 384, depth=12,
                        num_heads=6, mlp_ratio=4, qkv_bias=True,
                        pre_trained_vit=pre_trained_vit, vit_depth=12, pre_trained_conformer=pre_trained_conformer,
                        additive_fusion_down=False, additive_fusion_up=False, up_ftr_map_size=[56]+[56]*3+[28]*4+[14]*4, down_ftr_map_size=[197]*12,
                        **kwargs)
    else:
        model = TransConv(patch_size=16, channel_ratio=4, embed_dim=768, depth=12,
                        num_heads=6, mlp_ratio=4, qkv_bias=True,
                        additive_fusion_down=False, additive_fusion_up=False, up_ftr_map_size=[56]+[56]*3+[28]*4+[14]*4, down_ftr_map_size=[197]*12,               
                        **kwargs)
    return model

@register_model
def Transconv_large_patch16(pretrained=False, **kwargs):
    if pretrained:
        pre_trained_vit = mae_vit_large_patch16(pretrained=True)
        # pre_trained_vit = None  # uncomment to initialize only conv tower
        pre_trained_conformer = Conformer_base_patch16(pretrained=True)    
        model = TransConv(patch_size=16, channel_ratio=6, embed_dim=1024 if pre_trained_vit is not None else 576, depth=12,
                        num_heads=9, mlp_ratio=4, qkv_bias=True,
                        pre_trained_vit=pre_trained_vit, vit_depth=24, pre_trained_conformer=pre_trained_conformer,
                        additive_fusion_down=False, additive_fusion_up=False, up_ftr_map_size=[56]+[56]*3+[28]*4+[14]*4, down_ftr_map_size=[197]*12,
                        **kwargs)
    else:          
        model = TransConv(patch_size=16, channel_ratio=4, embed_dim=1024, depth=12,
                        num_heads=9, mlp_ratio=4, qkv_bias=True, vit_depth=24,
                        additive_fusion_down=False, additive_fusion_up=False, up_ftr_map_size=[56]+[56]*3+[28]*4+[14]*4, down_ftr_map_size=[197]*12,
                        **kwargs)
    return model

@register_model
def Transconv_base_patch14(pretrained=False, **kwargs):
    if pretrained:
        pre_trained_vit = mae_vit_huge_patch14(pretrained=True)
        pre_trained_conformer = Conformer_base_patch16(pretrained=True)
        model = TransConv(patch_size=16, channel_ratio=6, embed_dim=1280 if pre_trained_vit is not None else 576, depth=12,
                        num_heads=9, mlp_ratio=4, qkv_bias=True,
                        pre_trained_vit=pre_trained_vit, vit_depth=32, pre_trained_conformer=pre_trained_conformer,
                        additive_fusion_down=False, additive_fusion_up=False, up_ftr_map_size=[56]+[56]*3+[28]*4+[14]*4, down_ftr_map_size=[197]*12,
                        **kwargs)
    else:
        model = TransConv(patch_size=16, channel_ratio=6, embed_dim=1280, depth=12,
                        num_heads=9, mlp_ratio=4, qkv_bias=True, vit_depth=32,
                        additive_fusion_down=False, additive_fusion_up=False, up_ftr_map_size=[56]+[56]*3+[28]*4+[14]*4, down_ftr_map_size=[197]*12,
                        **kwargs)
    return model