#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class VAE_GEN(nn.Model):
    def __init__(self, cfg):
        super(VAE_GEN, self).__init__()
        self.enc_input_size = cfg.MODEL.ENC.INPUT_SIZE
        self.enc_hidden_size = cfg.MODEL.ENC.HIDDEN_SIZE
        self.latent_size = cfg.MODEL.LATENT_SIZE
        self.dec_hidden_size = cfg.MODEL.DEC.HIDDEN_SIZE
        layer = [nn.Linear(self.enc_input_size, self.enc_hidden_size),
                      nn.ReLU(True),
                      nn.DropOut(p=cfg.MODEL.DROP_OUT_PROB)]
        for i in range(cfg.MODEL.GEN_DEPT):
            layer += [nn.Linear(self.enc_hidden_size, self.enc_hidden_size),
                      nn.ReLU(True),
                      nn.DropOut(p=cfg.MODEL.DROP_OUT_PROB)]
        self.encoder = nn.Sequential(*layer)
        self.mu = nn.Linear(self.enc_hidden_size, self.latent_size)
        self.logvar = nn.Linear(self.enc_hidden_size, self.latent_size)

        layer = [nn.Linear(self.latent_size, self.dec_hidden_size),
                 nn.ReLU(True),
                 nn.DropOut(p=cfg.MODEL.DROP_OUT.PROB)]
        for i in range(0, cfg.MODEL.GEN_DEPT):
            layer += [nn.Linear(self.dec_hidden_size, self.dec_hidden_size),
                      nn.ReLU(True),
                      nn.DropOut(p=cfg.MODEL.DROP_OUT.PROB)]
        layer += nn.Linear(self.dec_hidden_size, self.enc_input_size)
        self.decoder = nn.Sequential(*layer)

    def enc_forward(self, x):
        x = self.encoder(x)
        mu = self.mu(x)
        logvar = F.softplus(self.logvar(x))
        return mu, logvar

    def sample(self, mu, logvar):
        eps = torch.randn(logvar.shape).to(logvar.device)
        return mu + eps * logvar

    def dec_forward(self, z):
        z = self.decoder(z)
        return nn.functional.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.enc_forward(x)
        z = self.sample(mu, logvar)
        x_prime = self.dec_forward(z)
        return mu, logvar, z, x_prime


class VAE_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(VAE_Classifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, out_size)
        # init_weights(self.layer1)
        # init_weights(self.layer2)
        # init_weights(self.layer3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf,bias):
        super(ResNet, self).__init__()
        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes, bias=bias)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(nclasses, nf=20,bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf,bias)

def get_model(type, **kwargs):
    if type=='resnet':
        return ResNet18(kwargs['n_cls'])
    elif type=='mlp':
        return Model(kwargs['input_size'], kwargs['hidden_size'], kwargs['out_size'])
