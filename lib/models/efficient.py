from efficientnet_pytorch import EfficientNet
import random
import torch
import timm
import torch.nn as nn
from collections import OrderedDict

def set_value(val=0.3):
    if val != 0.5:
        val = val
    return val

def get_net(config, **kwargs):
    if config.MODEL.NAME=='efficient':
        in_channels = 1 if config.DATASET.GRAY else 3
        if hasattr(config.MODEL.EXTRA, 'TYPE'):
            t = config.MODEL.EXTRA.TYPE
        else:
            t = 'b0'
        model = EfficientNet.from_pretrained(f'efficientnet-{t}',
                                            num_classes=9,
                                            in_channels=in_channels)
        if hasattr(config.MODEL, 'DROPOUT'):
            model._dropout = torch.nn.Dropout(p=config.MODEL.DROPOUT)
    if config.MODEL.NAME=='resnet':
        in_channels = 1 if config.DATASET.GRAY else 3
        t = config.MODEL.EXTRA.TYPE
        model = timm.create_model(t,in_chans=in_channels,
            pretrained=True,num_classes=9)  
        dropout_value = set_value()
        if hasattr(config.MODEL, 'DROPOUT'):
            dropout_value = config.MODEL.DROPOUT
        fc_features = model.fc.in_features
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(fc_features, 128)),
                                ('relu1', nn.ReLU()), 
                                ('dropout1',nn.Dropout(dropout_value)),
                                ('fc2', nn.Linear(128, 9))
                                ]))
        model.fc = classifier
        
    if config.MODEL.NAME=='repvgg':
        in_channels = 1 if config.DATASET.GRAY else 3
        t = config.MODEL.EXTRA.TYPE
        model = timm.create_model(t,in_chans=in_channels,
            pretrained=True,num_classes=9)   
        dropout_value = set_value()
        if hasattr(config.MODEL, 'DROPOUT'):
            dropout_value = config.MODEL.DROPOUT
        fc_features = model.head.fc.in_features
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(fc_features, 128)),
                                ('relu1', nn.ReLU()), 
                                ('dropout1',nn.Dropout(dropout_value)),
                                ('fc2', nn.Linear(128, 9))
                                ]))
        model.head.fc = classifier
    return model
