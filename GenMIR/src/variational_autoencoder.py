#!/usr/bin/env python

import argparse
import logging
import os
import shutil
import sys
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
from torch.utils import tensorboard
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transform import to_tensor
from torchvision.datasets import MNIST


################################################################################
# Utils
################################################################################
def setup_logger(name, save_dir, distributed_rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def setup_exp(save_dir, exp_name, clean_run='False'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    runs_dir = os.path.join(save_dir, 'runs', exp_name)
    chkpt_dir = os.path.join(save_dir, exp_name, 'chkpt')
    log_dir = os.path.join(save_dir, exp_name, 'logs')

    def mkdir(dir_name):
        print(dir_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    if clean_run:
        if os.path.exists(log_dir) or os.path.exists(chkpt_dir):
            shutil.rmtree(os.path.join(save_dir, exp_name))
        if os.path.exists(runs_dir):
            shutil.rmtree(runs_dir)
        time.sleep(5)
    mkdir(log_dir)
    mkdir(runs_dir)
    mkdir(chkpt_dir)

    return log_dir, runs_dir, chkpt_dir


################################################################################
# Model
################################################################################
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.enc_hid_sz = args.enc_hid_sz
        self.embed_sz = args.latent_sz

        self.enc_layer1 = nn.Linear(args.input_sz, 256)
        self.enc_layer2 = nn.Linear(256, 128)
        self.enc_layer3 = nn.Linear(128, 64)
        self.enc_layer_mu = nn.Linear(64, 20)
        self.enc_layer_var = nn.Linear(64, self.embed_sz)

        self.dec_layer1 = nn.Linear(self.embed_sz, 28)
        self.dec_layer2 = nn.Linear(28, 64)
        self.dec_layer3 = nn.Linear(64, 128)
        self.dec_layer4 = nn.Linear(128, 256)
        self.dec_layer_out = nn.Linear(256, args.input_sz)

    def enc_forward(self, x):
        x = nn.functional.relu(self.enc_layer1(x))
        x = nn.functional.relu(self.enc_layer2(x))
        x = nn.functional.relu(self.enc_layer3(x))
        mu = self.enc_layer_mu(x)
        logvar = self.enc_layer_var(x)
        return mu, logvar

    def sample(self, mu, logvar):
        eps = torch.randn(logvar.shape).to(logvar.device)
        return mu + eps * logvar

    def dec_forward(self, z):
        z = nn.functional.relu(self.dec_layer1(z))
        z = nn.functional.relu(self.dec_layer2(z))
        z = nn.functional.relu(self.dec_layer3(z))
        z = nn.functional.relu(self.dec_layer4(z))
        return nn.functional.sigmoid(self.dec_layer_out(z))

    def forward(self, x):
        mu, logvar = self.enc_forward(x)
        z = self.sample(mu, logvar)
        x_prime = self.dec_forward(z)
        return mu, logvar, z, x_prime


################################################################################
# Dataloader
################################################################################
def get_loader(args, train):
    # transform_fnc = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transform_fnc = transforms.Compose([transforms.ToTensor()])
    if train:
        loader = DataLoader(MNIST('./.torch/data', train=True, download=True, transform=transform_fnc),
                            batch_size=args.batch_size, shuffle=True)
    else:
        loader = DataLoader(MNIST('./.torch/data', train=False, download=True, transform=transform_fnc),
                            batch_size=args.batch_size, shuffle=True)

    return loader


################################################################################
# Trainer
################################################################################
def kl_criterion(mu, logvar, reduce=None):
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
    return torch.mean(kl_loss)


def visualize(x, writer, writer_idx, args):
    img = x.view((-1,) + args.img_shape)
    img = torch.unsqueeze(img[:, :], 1)
    writer.add_images('Recon Output', img, writer_idx, dataformats='NCHW')

def latent_visualize(model, writer, writer_ix, args):
    a = torch.arange(0, 1, 10)
    b = torch.arange(0, 1, 10)
    res = torch.zeros(100, 28, 28)
    for i in range(0, 10):
        for j in range(0, 10):
            z = [i, j]
            res[i*10 + j] = model.dec_forward((z))



def train(args, model, loader, logger, writer, device):
    model.train()
    model = model.to(device)
    batch_size = args.batch_size
    num_batches = args.num_samples // batch_size
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
    num_epochs = args.num_epochs
    # cls_criterion = nn.functional.binary_cross_entropy (reduction='mean')
    kl_factor = 1.0
    train_loader, test_loader = loader
    writer_idx = 0
    for eid in range(num_epochs):
        for bid, data in enumerate(train_loader):
            x, y = data
            x = x.view(min(x.shape[0], args.batch_size), -1).to(device)
            y = y.to(device)
            mu, logvar , z, x_prime = model.forward(x)
            kl_loss = kl_criterion(mu, logvar)
            # Important See the defination of bce loss at https://pytorch.org/docs/stable/nn.html#loss-functions
            cls_loss = torch.mean(torch.sum(nn.functional.binary_cross_entropy(x_prime.view(x.shape[0], -1), x.view(x.shape[0], -1), reduction='none'), 1))
            loss = cls_loss + kl_factor * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if bid % args.log_freq == 0:
                writer_idx += 1
                logger.info(f'Writer Idx:{writer_idx%args.log_freq}, Epoch Idx:{eid}, cls_loss: {cls_loss}, kl_loss: {kl_loss}, Loss: {kl_factor * kl_loss + cls_loss}')
                writer.add_scalar('kl_loss', kl_loss, writer_idx)
                writer.add_scalar('cls_loss', cls_loss, writer_idx)
                writer.add_scalar('kl_factor', kl_factor, writer_idx)
                visualize(x_prime, writer, writer_idx, args)
        test(args, model, test_loader, logger, writer, device, eid, kl_factor)
        sample = torch.randn(64, args.latent_sz).to(device)
        res = model.dec_forward(sample)
        img = res.view((-1,) + args.img_shape)
        img = torch.unsqueeze(img[:, :], 1)
        save_image(img, 'final_ouput.png')
        writer.add_images('Final Output', img, eid, dataformats='NCHW')
        kl_factor = 1.0 / (1+np.exp(-eid+3))


def test(args, model, test_loader, logger, writer, device, eid, kl_factor):
    model.eval()
    model = model.to(device)
    batch_size = args.batch_size
    num_batches = args.test_num_samples // batch_size
    cls_criterion = nn.BCEWithLogitsLoss()
    for bid, data in enumerate(test_loader):
        writer_idx = bid * batch_size + (eid * num_batches * batch_size)
        x, y = data
        x = x.view(min(args.batch_size, x.shape[0]), -1).to(device)
        y = y.to(device)
        mu, logvar, z, x_prime = model.forward(x)
        kl_loss = kl_criterion(mu, logvar, reduce='mean')
        cls_loss = cls_criterion(x_prime, x)
        loss = cls_loss + kl_factor * kl_loss

    logger.info(f'Epoch: {eid}, Test Loss: {loss}')
    writer.add_scalar('test_kl_loss', kl_loss, writer_idx)
    writer.add_scalar('test_cls_loss', cls_loss, writer_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Save Directory', default='./variational', type=str)
    parser.add_argument('--clean_run', help='Start From scratch', default=True, type=bool)
    parser.add_argument('--exp_name', help='Exp Name', default='default', type=str)
    parser.add_argument('--batch_size', help='Batch Size', default=64, type=int)
    parser.add_argument('--num_epochs', help='Number of Epochs', default=100, type=int)
    parser.add_argument('--num_samples', help='Number of samples to use', default=-1, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-4, type=float)
    parser.add_argument('--device_id', help='Device number', default=-1, type=int)
    parser.add_argument('--enc_hid_sz', help='Encoder Hidden Size', default=100, type=int)
    parser.add_argument('--latent_sz', help='Latent Embedding Size', default=20, type=int)
    parser.add_argument('--log_freq', help='Logging Frequency', default=100, type=int)
    parser.add_argument('--img_shape', help='Image Shape', default=(28, 28), type=tuple)
    parser.add_argument('--input_sz', help='Input Size', default=784, type=int)

    args = parser.parse_args()
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    log_dir, runs_dir, chkpt_dir = setup_exp(args.save_dir, args.exp_name, args.clean_run)
    writer = tensorboard.SummaryWriter(runs_dir)
    logger = setup_logger(args.exp_name, log_dir)
    model = Model(args)
    train_loader = get_loader(args, train=True)
    test_loader = get_loader(args, train=False)
    if args.num_samples == -1:
        args.num_samples = len(train_loader.dataset)
        args.test_num_samples = len(test_loader.dataset)
    train(args, model, [train_loader, test_loader], logger, writer, device)


if __name__ == '__main__':
    main()
