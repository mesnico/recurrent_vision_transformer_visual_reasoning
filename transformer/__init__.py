from .sd_transformer import ConvViUT
from .vit import ViTModel, ConvViTModel

def transformer_convviut_act(pretrained=False, map_location=None, depth=6):
    return ConvViUT(act=True, vit_depth=depth)

def transformer_convviut(pretrained=False, map_location=None, depth=6):
    return ConvViUT(vit_depth=depth)

def transformer_econvviut(pretrained=False, map_location=None, depth=6, u_depth=6):
    return ConvViUT(vit_depth=depth, vit_conv_strides=[2, 2, 2, 2], patch_size=8, equivariant=True,
                    vit_conv_channels=[16, 32, 64, 128], mlp_dim=1024, vit_heads=4, u_heads=4, vit_dim=1024, max_hops=u_depth)

def transformer_convviut_hires(pretrained=False, map_location=None, depth=6):
    return ConvViUT(vit_depth=depth, vit_conv_strides=[1, 2, 2, 2], patch_size=8)

def transformer_convviut_hires_small(pretrained=False, map_location=None, depth=6, u_depth=6):
    return ConvViUT(vit_depth=depth, vit_conv_strides=[1, 2, 2, 2], patch_size=8,
                    mlp_dim=256, vit_heads=4, u_heads=4, vit_dim=256, max_hops=u_depth)

def transformer_convviut_hires_multiloss_medium(pretrained=False, map_location=None, depth=6, u_depth=6):
    return ConvViUT(vit_depth=depth, vit_conv_strides=[1, 2, 2, 2], patch_size=8, equivariant=False,
                    vit_conv_channels=[128, 192, 256, 384], mlp_dim=512, vit_heads=4, u_heads=4, vit_dim=512, max_hops=u_depth, multiloss=True, pretrained=pretrained)

def transformer_econvviut_hires(pretrained=False, map_location=None, depth=6, u_depth=6):
    return ConvViUT(vit_depth=depth, vit_conv_strides=[1, 2, 2, 2], patch_size=8, equivariant=True, vit_conv_channels=[32, 48, 64, 128], max_hops=u_depth)

def transformer_econvviut_hires_small(pretrained=False, map_location=None, depth=6, u_depth=6):
    return ConvViUT(vit_depth=depth, vit_conv_strides=[1, 2, 2, 2], patch_size=8, equivariant=True,
                    vit_conv_channels=[16, 24, 32, 48], mlp_dim=256, vit_heads=4, u_heads=4, vit_dim=256, max_hops=u_depth)

def transformer_econvviut_hires_small_act(pretrained=False, map_location=None, depth=6, u_depth=6):
    return ConvViUT(vit_depth=depth, vit_conv_strides=[1, 2, 2, 2], patch_size=8, equivariant=True,
                    vit_conv_channels=[16, 24, 32, 48], mlp_dim=256, vit_heads=4, u_heads=4, vit_dim=256, max_hops=u_depth, act=True)

def transformer_econvviut_hires_multiloss_medium(pretrained=False, map_location=None, depth=6, u_depth=6):
    return ConvViUT(vit_depth=depth, vit_conv_strides=[1, 2, 2, 2], patch_size=8, equivariant=True,
                    vit_conv_channels=[16, 24, 32, 48], mlp_dim=512, vit_heads=4, u_heads=4, vit_dim=512, max_hops=u_depth, multiloss=True, pretrained=pretrained)

def transformer_econvvit_hires_medium(pretrained=False, map_location=None, depth=6, u_depth=6):
    return ConvViUT(vit_depth=depth, vit_conv_strides=[1, 2, 2, 2], patch_size=8, equivariant=True,
                    vit_conv_channels=[16, 24, 32, 48], mlp_dim=512, vit_heads=4, u_heads=4, vit_dim=512, max_hops=u_depth, multiloss=False, pretrained=pretrained)

def transformer_econvviut_hires_multiloss_small(pretrained=False, map_location=None, depth=6, u_depth=6):
    return ConvViUT(vit_depth=depth, vit_conv_strides=[1, 2, 2, 2], patch_size=8, equivariant=True,
                    vit_conv_channels=[16, 24, 32, 48], mlp_dim=256, vit_heads=4, u_heads=4, vit_dim=256, max_hops=u_depth, multiloss=True, pretrained=pretrained)

def transformer_econvviut_hires_multiloss_smaller(pretrained=False, map_location=None, depth=6, u_depth=6):
    return ConvViUT(vit_depth=depth, vit_conv_strides=[1, 2, 2, 2], patch_size=8, equivariant=True,
                    vit_conv_channels=[16, 24, 32, 32], mlp_dim=128, vit_heads=2, u_heads=2, vit_dim=128, max_hops=u_depth, multiloss=True, pretrained=pretrained, dropout=0.2)

def transformer_econvviut_hires_smaller(pretrained=False, map_location=None, depth=6, u_depth=6):
    return ConvViUT(vit_depth=depth, vit_conv_strides=[1, 2, 2, 2], patch_size=8, equivariant=True,
                    vit_conv_channels=[16, 24, 32, 32], mlp_dim=128, vit_heads=4, u_heads=4, vit_dim=128, max_hops=u_depth)


def transformer_econvviut_hires_deep_small(pretrained=False, map_location=None, depth=6, u_depth=6):
    return ConvViUT(vit_depth=depth, vit_conv_strides=[1, 2, 2, 2], patch_size=8, equivariant=True,
                    vit_conv_channels=[16, 24, 32, 48], mlp_dim=128, vit_heads=4, u_heads=4, vit_dim=128, max_hops=u_depth, internal_enc_layers=3)


# def transformer_vit(pretrained=False, map_location=None, depth=6):
#     return ViTModel(depth=depth)
#
# def transformer_convvit(pretrained=False, map_location=None, depth=6):
#     return ConvViTModel(depth=depth)
#
# def transformer_econvvit(pretrained=False, map_location=None, depth=6):
#     return ConvViTModel(depth=depth, equivariant=True)
#
# def transformer_convvit_hires(pretrained=False, map_location=None, depth=6):
#     return ConvViTModel(depth=depth, strides=[1, 2, 2, 2])