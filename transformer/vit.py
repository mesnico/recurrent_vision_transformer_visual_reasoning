import torch
import torchvision
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from e2cnn import gspaces
from e2cnn import nn as enn

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, return_attention=False):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        # interesting_head = (attn[0, :, 0, 0].view(-1, 1, 1) != attn[0, :, :, :]).any(1).any(1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        if return_attention:
            return self.to_out(out), attn
        else:
            return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

    def get_last_selfattention(self, x):
        for it, (attn, ff) in enumerate(self.layers):
            if it < len(self.layers) - 1:
                x = attn(x) + x
                x = ff(x) + x
            else:
                _, att = attn.fn(attn.norm(x), return_attention=True)  # b, h, i, j
                return att

class CustomConvInputModel(nn.Module):
    def __init__(self, strides=None, channels=None):
        super(CustomConvInputModel, self).__init__()

        self.conv1 = nn.Conv2d(3, channels[0], 3, stride=strides[0], padding=1)
        self.batchNorm1 = nn.BatchNorm2d(channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=strides[1], padding=1)
        self.batchNorm2 = nn.BatchNorm2d(channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=strides[2], padding=1)
        self.batchNorm3 = nn.BatchNorm2d(channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=strides[3], padding=1)
        self.batchNorm4 = nn.BatchNorm2d(channels[3])

    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = self.batchNorm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.batchNorm3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.batchNorm4(x)
        x = F.relu(x)

        x = x.view(x.shape[0], x.shape[1], -1)  # B x C x W*H
        x = x.permute(0, 2, 1)  # B x W*H x C
        return x


class EquivariantConvModel(nn.Module):
    def __init__(self, strides=None, channels=None):
        super().__init__()

        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.Rot2dOnR2(N=8)

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = enn.FieldType(self.r2_act, 3*[self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C8
        out_type = enn.FieldType(self.r2_act, channels[0] * [self.r2_act.regular_repr])
        self.block1 = enn.SequentialModule(
            # enn.MaskModule(in_type, 29, margin=1),
            enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, stride=strides[0]),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True)
        )

        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = enn.FieldType(self.r2_act, channels[1] * [self.r2_act.regular_repr])
        self.block2 = enn.SequentialModule(
            # enn.MaskModule(in_type, 29, margin=1),
            enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, stride=strides[1]),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True)
        )
        # self.pool1 = nn.SequentialModule(
        #     nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        # )

        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = enn.FieldType(self.r2_act, channels[2] * [self.r2_act.regular_repr])
        self.block3 = enn.SequentialModule(
            # enn.MaskModule(in_type, 29, margin=1),
            enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, stride=strides[2]),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True)
        )

        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = enn.FieldType(self.r2_act, channels[3] * [self.r2_act.regular_repr])
        self.block4 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, stride=strides[3]),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True)
        )

        # self.gpool = enn.GroupPooling(out_type)

        # number of output channels
        # c = self.gpool.out_type.size

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = enn.GeometricTensor(input, self.input_type)

        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # pool over the group
        # x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        x = x.view(x.shape[0], x.shape[1], -1)  # B x C x W*H
        x = x.permute(0, 2, 1)  # B x W*H x C

        return x

class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.perm = args

    def forward(self, x):
        return x.permute(self.perm)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., use_conv = False, conv_strides = None, conv_channels = None, equivariant = False, pretrained = None):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        if use_conv:
            if conv_strides is None:
                conv_strides = [2, 2, 2, 2]
            if conv_channels is None:
                conv_channels = [32, 64, 128, 256]
            self.patch_res = 4 * 2 ** (8 - sum(conv_strides) + 1)
            print('Conv strides: {}; Num patches: {} x {}'.format(conv_strides, self.patch_res, self.patch_res))
            num_patches = self.patch_res ** 2
            patch_dim = conv_channels[-1] if not equivariant else conv_channels[-1] * 8
            if pretrained == 'rn':
                # self.patch_res = 8
                conv = CustomConvInputModel(strides=conv_strides, channels=[24, 24, 24, 24])
                self.to_patch_embedding = nn.Sequential(
                    conv,
                    nn.Linear(24, dim)
                )
                path = 'pretrained_models/original_fp_epoch_493.pth'
                pretrained_model = torch.load(path, map_location='cpu')
                conv_pretrained_dict = {k.replace('module.conv.', '', 1): v for k, v in pretrained_model.items() if
                                        '.conv.' in k}
                conv.load_state_dict(conv_pretrained_dict)
                print('Using pre-trained RN ConvNet')
            elif isinstance(pretrained, str) and 'resnet' in pretrained:
                self.patch_res = 8  # TODO HEREEEEE. Discover what is the shape in output from the cut resnet
                cut_info = pretrained.split('-')
                cut_point = int(cut_info[1])
                if len(cut_info) == 3:
                    self.patch_res = int(cut_info[2])
                assert (cut_point == 3 and self.patch_res == 8) or cut_point == 2
                num_patches = self.patch_res ** 2
                resnet = torchvision.models.resnet50(pretrained=True)
                conv = nn.Sequential(*list(resnet.children())[:cut_point + 4])  # cut the resnet to the first "cut_point" bottlenecks
                if cut_point == 3:
                    conv_dim = 1024
                elif cut_point == 2:
                    conv_dim = 512
                self.to_patch_embedding = nn.Sequential(
                    conv,
                    torch.nn.AvgPool2d(2, stride=2) if cut_point == 2 and self.patch_res == 8 else torch.nn.Identity(),
                    nn.Flatten(start_dim=2, end_dim=3),
                    Permute(0, 2, 1),
                    nn.Linear(conv_dim, dim)
                )
                print('Using pre-trained ResNet (sliced to first {} modules)'.format(cut_point))
            else:
                self.to_patch_embedding = nn.Sequential(
                    CustomConvInputModel(strides=conv_strides,
                                         channels=conv_channels) if not equivariant else EquivariantConvModel(
                        strides=conv_strides, channels=conv_channels),
                    nn.Linear(patch_dim, dim)
                )

            print('Conv Model: {}'.format(type(self.to_patch_embedding[0])))
        else:
            num_patches = (image_height // patch_height) * (image_width // patch_width)
            patch_dim = channels * patch_height * patch_width
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                nn.Linear(patch_dim, dim),
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if depth > 0:
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = None

        self.pool = pool
        self.to_latent = nn.Identity()
        self.num_classes = num_classes

        if num_classes > 0:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )

    def forward(self, img, return_last_attention=False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        if self.transformer is not None:
            if return_last_attention:
                x = self.transformer.get_last_selfattention(x)
                return x
            else:
                x = self.transformer(x)

        if self.num_classes > 0:
            x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

            x = self.to_latent(x)
            return self.mlp_head(x)
        else:
            return x    # B x S x dim

    def get_last_selfattention(self, img):
        return self.forward(img, True)

def ViTModel(depth=6):
    model = ViT(
        image_size=128,
        patch_size=16,
        num_classes=2,
        dim=1024,
        depth=depth,
        heads=8,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1
    )

    return model

def ConvViTModel(depth=6, equivariant=False, strides=None):
    model = ViT(
        image_size=128,
        patch_size=16,
        num_classes=2,
        dim=512,
        depth=depth,
        heads=16,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1,
        use_conv=True,
        conv_strides=strides,
        equivariant=equivariant
    )

    return model

