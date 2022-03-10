import torch
import torch.utils.model_zoo

from cornet.cornet_z import CORnet_Z
from cornet.cornet_z import HASH as HASH_Z
from cornet.cornet_r import CORnet_R
from cornet.cornet_r import HASH as HASH_R
from cornet.cornet_s import CORnet_S, CORnet_S_NoRecurrent, CORnet_S_NoResidual, CORnet_S_NoResidualNoRecurrent
from cornet.cornet_s import HASH as HASH_S
from cornet.standards import Resnet101, Resnet34, Resnet18
from cornet.nonresnets import NonResnet101, NonResnet34, NonResnet18
from cornet.standards import VGG19, VGG19BN
from cornet.standards import AlexNet
from cornet.standards import DenseNet121, DenseNet161, DenseNet201
from cornet.rn import RN

import pdb


def get_model(model_letter, pretrained=False, map_location=None):
    model_letter = model_letter.upper()
    model_hash = globals()[f'HASH_{model_letter}']
    model = globals()[f'CORnet_{model_letter}']()
    model = torch.nn.DataParallel(model)
    if pretrained:
        url = f'https://s3.amazonaws.com/cornet-models/cornet_{model_letter.lower()}-{model_hash}.pth'
        ckpt_data = torch.utils.model_zoo.load_url(url, map_location=map_location)
        model.load_state_dict(ckpt_data['state_dict'])
    return model


def cornet_z(pretrained=False, map_location=None):
    return get_model('z', pretrained=pretrained, map_location=map_location)


def cornet_r(pretrained=False, map_location=None):
    return get_model('r', pretrained=pretrained, map_location=map_location)


def cornet_s(pretrained=False, map_location=None):
    return get_model('s', pretrained=pretrained, map_location=map_location)


def cornet_s_norecurrent(pretrained=False, map_location=None):
    model = globals()['CORnet_S_NoRecurrent']()
    model = torch.nn.DataParallel(model)
    return model

def cornet_s_noresidual(pretrained=False, map_location=None):
    model = globals()['CORnet_S_NoResidual']()
    model = torch.nn.DataParallel(model)
    return model

def cornet_s_noresidual_norecurrent(pretrained=False, map_location=None):
    model = globals()['CORnet_S_NoResidualNoRecurrent']()
    model = torch.nn.DataParallel(model)
    return model

def resnet101(pretrained=False, map_location=None):
    model = globals()['Resnet101']()
    model = torch.nn.DataParallel(model)
    return model

def resnet34(pretrained=False, map_location=None):
    model = globals()['Resnet34']()
    model = torch.nn.DataParallel(model)
    return model

def resnet18(pretrained=False, map_location=None):
    model = globals()['Resnet18']()
    model = torch.nn.DataParallel(model)
    return model

def non_resnet101(pretrained=False, map_location=None):
    model = globals()['NonResnet101']()
    model = torch.nn.DataParallel(model)
    return model

def non_resnet34(pretrained=False, map_location=None):
    model = globals()['NonResnet34']()
    model = torch.nn.DataParallel(model)
    return model

def non_resnet18(pretrained=False, map_location=None):
    model = globals()['NonResnet18']()
    model = torch.nn.DataParallel(model)
    return model

def vgg19(pretrained=False, map_location=None):
    model = globals()['VGG19']()
    model = torch.nn.DataParallel(model)
    return model

def vgg19bn(pretrained=False, map_location=None):
    model = globals()['VGG19BN']()
    model = torch.nn.DataParallel(model)
    return model

def alexnet(pretrained=False, map_location=None):
    model = globals()['AlexNet']()
    model = torch.nn.DataParallel(model)
    return model

def rn(pretrained=False, map_location=None):
    model = globals()['RN']()
    model = torch.nn.DataParallel(model)
    return model

def densenet121(pretrained=False, map_location=None):
    model = globals()['DenseNet121']()
    model = torch.nn.DataParallel(model)
    return model

def densenet161(pretrained=False, map_location=None):
    model = globals()['DenseNet161']()
    model = torch.nn.DataParallel(model)
    return model

def densenet201(pretrained=False, map_location=None):
    model = globals()['DenseNet201']()
    model = torch.nn.DataParallel(model)
    return model
