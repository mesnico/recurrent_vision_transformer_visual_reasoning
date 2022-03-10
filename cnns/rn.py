import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import pdb

DEBUG = False


def debug_print(msg):
    if DEBUG:
        print(msg)


class CustomConvInputModel(nn.Module):
    def __init__(self):
        super(CustomConvInputModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

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
        return x


class RelationalLayerBase(nn.Module):
    def __init__(self, in_size, out_size, hyp):
        super().__init__()

        self.on_gpu = False
        self.hyp = hyp
        self.in_size = in_size
        self.out_size = out_size

    def cuda(self):
        self.on_gpu = True
        super().cuda()

    def forward(self, *input):
        return NotImplementedError


class RelationalLayer(RelationalLayerBase):
    def __init__(self, in_size, out_size, hyp):
        super().__init__(in_size, out_size, hyp)

        self.aggreg_position = hyp["aggregation_position"]
        # dropouts
        self.dropouts = {int(k): nn.Dropout(p=v) for k, v in hyp["dropouts"].items()}

        self.in_size = in_size

        # create aggregation weights
        if 'weighted' in hyp['aggregation']:
            self.aggreg_weights = nn.Parameter(torch.Tensor(12 ** 2 if hyp['state_description'] else 64 ** 2))
            nn.init.uniform_(self.aggreg_weights, -1, 1)

        # create all g layers
        self.g_layers = []
        self.g_layers_size = hyp["g_layers"]
        for idx, g_layer_size in enumerate(hyp["g_layers"]):
            in_s = in_size if idx == 0 else hyp["g_layers"][idx - 1]
            out_s = g_layer_size
            lin = nn.Linear(in_s, out_s)
            self.g_layers.append(lin)
        self.g_layers.append(nn.Linear(self.g_layers_size[-1], out_size)) 
        self.g_layers_size.append(out_size)
        self.g_layers = nn.ModuleList(self.g_layers)
        self.aggregation = hyp['aggregation']

    def forward(self, x):
        # x = (B x 8*8 x 24)
        # qst = (B x 128)
        """g"""
        b, d, k = x.size()

        # cast all pairs against each other
        x_i = torch.unsqueeze(x, 1)  # (B x 1 x 64 x 26)
        x_i = x_i.repeat(1, d, 1, 1)  # (B x 64 x 64 x 26)
        x_j = torch.unsqueeze(x, 2)  # (B x 64 x 1 x 26)
        # x_j = torch.cat([x_j, qst], 3)
        x_j = x_j.repeat(1, 1, d, 1)  # (B x 64 x 64 x 26)

        # concatenate all together
        x_full = torch.cat([x_i, x_j], 3)  # (B x 64 x 64 x 2*26)

        # reshape for passing through network
        x_ = x_full.view(b * d ** 2, self.in_size)

        # create g and inject the question at the position pointed by quest_inject_position.
        for idx, (g_layer, g_layer_size) in enumerate(zip(self.g_layers, self.g_layers_size)):
            in_size = self.in_size if idx == 0 else self.g_layers_size[idx - 1]
            if idx == self.aggreg_position:
                debug_print('{} - Aggregation'.format(idx))
                x_ = x_.view(b, -1, in_size)
                if self.aggregation == 'sum':
                    x_ = x_.sum(1)
                elif self.aggregation == 'avg':
                    x_ = x_.mean(1)
                elif self.aggregation == 'weighted_sum':
                    x_ = torch.matmul(x_.permute(0, 2, 1), self.aggreg_weights)
                else:
                    raise ValueError('Aggregation not recognized: {}'.format(self.aggregation))

            x_ = g_layer(x_)

            debug_print('{} - Layer. Output dim: {}'.format(idx, x_.size()))

            if idx in self.dropouts:
                debug_print('{} - Dropout p={}'.format(idx, self.dropouts[idx].p))
                x_ = self.dropouts[idx](x_)

            # apply ReLU after every layer except the last
            if idx != len(self.g_layers_size) - 1:
                debug_print('{} - ReLU'.format(idx))
                x_ = F.relu(x_)

        if DEBUG:
            pdb.set_trace()

        return x_


hyp = {
        "rl_in_size": 52,
        "state_description": False,
        "aggregation": "sum",
        "aggregation_position": 4,
        "dropouts": {"5": 0.5},
        "g_layers": [256, 256, 256, 256, 256, 256]
}

class RN(nn.Module):
    def __init__(self):
        super(RN, self).__init__()
        self.coord_tensor = None
        self.on_gpu = False

        out_size = 2 #hyp['img_feat_dim']
        # dataset_name = hyp['dataset_name']

        # hyp = hyp['rn_module']

        # CNN
        self.conv = CustomConvInputModel()

        self.state_desc = hyp['state_description']

        # RELATIONAL LAYER
        self.rl_in_size = hyp["rl_in_size"]
        self.rl = RelationalLayer(self.rl_in_size, out_size, hyp)

    def forward(self, img):
        if self.state_desc:
            x = img  # (B x 12 x 8)
        else:
            x = self.conv(img)  # (B x 24 x 8 x 8)
            b, k, d, _ = x.size()
            x = x.view(b, k, d * d)  # (B x 24 x 8*8)

            # add coordinates
            if self.coord_tensor is None or torch.cuda.device_count() == 1:
                self.build_coord_tensor(b, d)  # (B x 2 x 8 x 8)
                self.coord_tensor = self.coord_tensor.to(img.device)
                self.coord_tensor = self.coord_tensor.view(b, 2, d * d)  # (B x 2 x 8*8)

            x = torch.cat([x, self.coord_tensor], 1)  # (B x 24+2 x 8*8)
            x = x.permute(0, 2, 1)  # (B x 64 x 24+2)

        y = self.rl(x)
        return y

    # prepare coord tensor
    def build_coord_tensor(self, b, d):
        coords = torch.linspace(-d / 2., d / 2., d)
        x = coords.unsqueeze(0).repeat(d, 1)
        y = coords.unsqueeze(1).repeat(1, d)
        self.coord_tensor = torch.stack((x, y))

        # broadcast to all batches
        self.coord_tensor = self.coord_tensor.unsqueeze(0).repeat(b, 1, 1, 1)
        self.coord_tensor.requires_grad_(False)
        
