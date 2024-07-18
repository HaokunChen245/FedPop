import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
import time
import random
import torchvision
from collections import defaultdict
import string
from torch.nn.utils import rnn

CHARMAP = defaultdict(lambda: 1)
CHARMAP.update({char: i+2 for i, char in enumerate(string.printable)})
VOCAB = len(set(CHARMAP.values())) + 1

#############
# MobileNet #
#############
# https://github.com/jmjeon94/MobileNet-Pytorch
def efficient_conv(builder, seed, in_channels, out_channels, stride):
    return nn.Sequential(
        builder.conv(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=0, groups=in_channels, bias=False, seed=seed),
        builder.bn(in_channels, seed=seed),
        nn.ReLU(True),
        
        builder.conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False, seed=seed),
        builder.bn(out_channels, seed=seed),
        nn.ReLU(True)
    )

class MobileNet(nn.Module):
    def __init__(self, builder, args, seed, block):
        super(MobileNet, self).__init__()
        self._seed = seed
        self.builder = builder
        
        # input layer
        self.in_conv = nn.Sequential(
            builder.conv(in_channels=args.in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False, seed=seed),
            builder.bn(32, seed=seed),
            nn.ReLU(True)
        )
        
        # intermediate layers
        self.layers = nn.Sequential(
            efficient_conv(builder, seed, 32, 64, 1),
            efficient_conv(builder, seed, 64, 128, 2),
            efficient_conv(builder, seed, 128, 256, 2),
            efficient_conv(builder, seed, 256, 512, 1),
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            builder.linear(in_features=512, out_features=256, bias=False, seed=seed),
            nn.ReLU(True),
            builder.linear(in_features=256, out_features=args.num_classes, bias=False, seed=seed)
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.layers(x)
        x = self.classifier(x)
        return x
        
''' Network from FedEx for Shakespeare'''
class CharLSTM(nn.Module):

    def __init__(self, input_size=8, hidden_size=256, **kwargs):

        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=VOCAB, 
                                      embedding_dim=input_size, padding_idx=0)
        self.lstm= nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                           batch_first=True, bidirectional=False, **kwargs)
        self.linear = nn.Linear(hidden_size, VOCAB)

    def forward(self, X, lengths):

        X = self.embedding(X)
        X = rnn.pack_padded_sequence(X, lengths, batch_first=True)
        self.lstm.flatten_parameters()
        X, _ = self.lstm(X)
        X, _ = rnn.pad_packed_sequence(X, batch_first=True)
        return self.linear(X[:,-1])


''' CNN for CelebA'''
class CNN_CelebA(nn.Module):
    def __init__(self):
        super(CNN_CelebA, self).__init__()
        dims_in = [3, 32, 32, 32]
        dims_out = [32, 32, 32, 32]
        self.convs = []
        for dim_in, dim_out in zip(dims_in, dims_out):
            self.convs += [
                nn.Conv2d(dim_in, dim_out, 3, padding=1),
                nn.BatchNorm2d(dim_out),
                nn.MaxPool2d(2),
                nn.ReLU(),
            ]
        self.dropout = nn.Dropout2d(0.0)
        self.convs = nn.Sequential(*self.convs)
        self.fc = nn.Sequential(
                    nn.Linear(4576, 1024),
                    nn.ReLU(),
                )
        self.clf = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = self.clf(x)
        return x

''' CNN from FedEx for CIFAR10'''
class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super(CNN_CIFAR10, self).__init__()
        self.conv1 = nn.Sequential(
                                   nn.Conv2d(3, 32, 3, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   )
        self.conv2 = nn.Sequential(
                                   nn.Conv2d(32, 64, 3, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   )
        self.conv3 = nn.Sequential(
                                   nn.Conv2d(64, 64, 3, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2),
                                   )
        self.dropout = nn.Dropout(0.0)
        self.fc = nn.Sequential(
                                nn.Linear(1024, 64),
                                nn.ReLU(),
                                )
        self.clf = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(self.dropout(x.flatten(1)))
        return self.clf(self.dropout(x))


''' CNN from FedEx for CIFAR10'''
class CNN_FEMNIST(nn.Module):
    def __init__(self):
        super(CNN_FEMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout2d(0.0)
        self.fc1 = nn.Linear(9216, 1024)
        self.fc2 = nn.Linear(1024, 62)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, img_size, img_channel):
        super(ConvNet, self).__init__()

        self.featurizer, self.feature_shape = self._make_layers(net_width, net_depth, net_norm, net_act, net_pooling, img_size, img_channel)
        num_feat = self.feature_shape[0]*self.feature_shape[1]*self.feature_shape[2]
        self.classifier = nn.Linear(num_feat, num_classes)

    @property
    def device(self):
        return next(self.parameters()).device
        
    def forward(self, x):
        out = self.featurizer(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        if net_norm == 'batchnorm':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, net_width, net_depth, net_norm, net_act, net_pooling, img_size, img_channel):
        layers = []
        in_channels = img_channel
        shape_feat = [img_channel, img_size, img_size]
        for d in range(net_depth):
            layers += [(f'conv{d}', nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if img_channel == 1 and d == 0 else 1))]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [(f'norm{d}', self._get_normlayer(net_norm, shape_feat))]
            layers += [(f'actv{d}', self._get_activation(net_act))]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [(f'pool{d}', self._get_pooling(net_pooling))]
                shape_feat[1] //= 2
                shape_feat[2] //= 2        

        return nn.Sequential(OrderedDict(layers)), shape_feat
