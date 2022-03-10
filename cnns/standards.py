import torchvision.models as models
from torch import nn

import pdb

def Resnet101():
    model = models.resnet101(pretrained=False)
    model.fc = nn.Linear(2048, 2)
    return model


def Resnet34():
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(512, 2)
    return model


def Resnet18():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 2)
    return model


def VGG19():
    model = models.vgg19(pretrained=False, num_classes=2)
    return model


def VGG19BN():
    model = models.vgg19_bn(pretrained=False, num_classes=2)
    return model


def AlexNet():
    model = models.alexnet(pretrained=False, num_classes=2)
    return model


'''class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        vgg19_model = models.vgg19(pretrained=False)

        # get layers from VGG except the last classifier layer
        self.conv = vgg19_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = vgg19_model.classifier[:6]
        self.last = nn.Linear(4096, 2)

    def forward(self, img):
        x = self.conv(img)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.last(x)
        return x


def AlexNet():
    model = models.alexnet(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 2)
    return model
'''

def DenseNet121():
    model = models.densenet121(pretrained=False, num_classes=2)
    return model


def DenseNet161():
    model = models.densenet161(pretrained=False, num_classes=2)
    return model


def DenseNet201():
    model = models.densenet201(pretrained=False, num_classes=2)
    return model
