import numpy as np
import torch
from torch.utils.data import random_split, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision import datasets
import hydra
import os
from utils.distributed_training import get_rank, mpi_size, is_main_process

from datasets.dct import DCT_dataset
# fix mnist download problem (https://github.com/pytorch/vision/issues/1938)
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


class BernoulliSample:
    def __call__(self, x):
        return torch.bernoulli(x)


class MNIST:
    def __init__(self, batch_size, test_batch_size, model, ctx_size, root, ddp=False, mpi_size=None, rank=None):
        self.root = root
        self.ddp = ddp
        self.mpi_size = mpi_size
        self.rank = rank
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.model = model
        self.ctx_size = ctx_size
        self.num_workers = 6
        self.prepare_data()

    def prepare_data(self):
        datasets.MNIST(self.root, train=True, download=True)
        datasets.MNIST(self.root, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = datasets.MNIST(self.root, train=True, transform=self.transforms)
            if 'context' in self.model:
                mnist_full = DCT_dataset(mnist_full, self.ctx_size)
            self.train, self.val = random_split(mnist_full, [55000, 5000])

        if stage == 'test' or stage is None:
            self.test = datasets.MNIST(self.root, train=False, transform=self.test_transforms)
            if 'context' in self.model:
                self.test = DCT_dataset(self.test, self.ctx_size)

    def train_dataloader(self):
        params = {
            'pin_memory': True,
            'drop_last': True
        }
        if self.ddp:
            train_sampler = DistributedSampler(self.train, shuffle=True,
                                               num_replicas=mpi_size(),
                                               rank=get_rank())
            params['sampler'] = train_sampler
        else:
            params['shuffle'] = True
            params['num_workers'] = self.num_workers
        return DataLoader(self.train, self.batch_size, **params)

    def val_dataloader(self):
        params = {
            'pin_memory': True,
            'drop_last': False
        }
        if self.ddp:
            val_sampler = DistributedSampler(self.val,
                                             shuffle=False,
                                             num_replicas=mpi_size(),
                                             rank=get_rank())
            params['sampler'] = val_sampler
        else:
            params['shuffle'] = True
            params['num_workers'] = self.num_workers
        return DataLoader(self.val, self.test_batch_size, **params)

    def test_dataloader(self):
        if is_main_process():
            return DataLoader(self.test, self.test_batch_size,
                              num_workers=self.num_workers, 
                              shuffle=False, 
                              pin_memory=True)
        return None


class FashionMNIST(MNIST):
    def __init__(self, batch_size, test_batch_size, model, ctx_size, root):
        super().__init__(batch_size, test_batch_size, model, ctx_size, root)

    def prepare_data(self):
        datasets.FashionMNIST(self.root, train=True, download=True)
        datasets.FashionMNIST(self.root, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            fmnist_full = datasets.FashionMNIST(self.root, train=True,
                                                transform=self.transforms)
            if 'context' in self.model:
                fmnist_full = DCT_dataset(fmnist_full, self.ctx_size)
            self.train, self.val = random_split(fmnist_full, [55000, 5000])

        if stage == 'test' or stage is None:
            self.test = datasets.FashionMNIST(self.root, train=False,
                                              transform=self.test_transforms)
            if 'context' in self.model:
                self.test = DCT_dataset(self.test, self.ctx_size)

