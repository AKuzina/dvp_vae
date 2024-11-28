import numpy as np
import torch
from torch.utils.data import random_split
from torchvision import transforms
from torchvision import datasets
import hydra
import os
from datasets.mnists import MNIST
from datasets.dct import DCT_dataset


class Normalize:
    def __init__(self, dequant=False, num_bits=8, shift=127.5, scale=1./127.5):
        self.dequant = dequant
        self.num_bits = num_bits
        self.shift = shift
        self.scale = scale

    def __call__(self, x):
        x = torch.FloatTensor(np.asarray(x, dtype=np.float32)).permute(2, 0, 1)
        # dequantize and scale to [0, 1]
        if self.dequant:
            x = (x + torch.rand_like(x).detach()) / (2 ** self.num_bits)
            x = 2 * x - 1
        else:
            x = (x - self.shift) * self.scale
        return x


class CIFAR10(MNIST):
    def __init__(self, batch_size, test_batch_size, model, ctx_size, root, ddp=False, mpi_size=None, rank=None, use_augmentation=False):
        super().__init__(
            batch_size=batch_size, 
            test_batch_size=test_batch_size, 
            model=model, 
            ctx_size=ctx_size, 
            root=root, 
            ddp=ddp, 
            mpi_size=mpi_size, 
            rank=rank,
        )
        self.transforms = transforms.Compose([
            Normalize(dequant=False),
            transforms.RandomHorizontalFlip(),
        ])
        if use_augmentation:
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
                transforms.ColorJitter(brightness=0.3, 
                                       contrast=0.3,
                                       saturation=0.1, 
                                       hue=0.1),
                Normalize(dequant=False),
                
            ])
            
        self.test_transforms = transforms.Compose([
            Normalize(dequant=False)
        ])

    def prepare_data(self):
        datasets.CIFAR10(self.root, train=True, download=True)
        datasets.CIFAR10(self.root, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            cifar_full = datasets.CIFAR10(self.root, train=True, transform=self.transforms)
            cifar_full.processed_folder = os.path.join(self.root, cifar_full.base_folder)

            if 'context' in self.model:
                cifar_full = DCT_dataset(cifar_full, self.ctx_size)
            N = len(cifar_full)
            self.train, self.val = random_split(cifar_full, [N-5000, 5000])

        if stage == 'test' or stage is None:
            self.test = datasets.CIFAR10(self.root, train=False, transform=self.test_transforms)
            self.test.processed_folder = os.path.join(self.root, self.test.base_folder)
            if 'context' in self.model:
                self.test = DCT_dataset(self.test, self.ctx_size)

