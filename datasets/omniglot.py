import torch
from torchvision import datasets
from torchvision.datasets.mnist import read_image_file, read_label_file
from torch.utils.data import random_split, TensorDataset, Dataset
from PIL import Image
import os
from torchvision import transforms
import urllib
from scipy.io import loadmat

from datasets.mnists import MNIST
from datasets.dct import DCT_dataset


class omniglot_dset(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        super(omniglot_dset, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform

        self.download_omniglot()
        omni = loadmat(os.path.join(self.processed_folder, 'chardata.mat'))
        if self.train:
            self.data = 255 * omni['data'].astype('float32').reshape(
                (28, 28, -1)).transpose((2, 1, 0))
        else:
            self.data = 255 * omni['testdata'].astype('float32').reshape(
                (28, 28, -1)).transpose((2, 1, 0))
        self.data = self.data.astype('uint8')
        print(self.data.shape)

    def download_omniglot(self):
        filename = 'chardata.mat'
        dir = self.processed_folder
        if not os.path.exists(dir):
            os.mkdir(dir)
        url = 'https://raw.github.com/yburda/iwae/master/datasets/OMNIGLOT/chardata.mat'

        filepath = os.path.join(dir, filename)
        if not os.path.exists(filepath):
            filepath, _ = urllib.request.urlretrieve(url, filepath)
            print('Downloaded', filename)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], 0
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.data.shape[0]

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'OMNIGLOT')

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'OMNIGLOT')


class OMNIGLOT(MNIST):
    def __init__(self, batch_size, test_batch_size, model, ctx_size, root, ddp=False, mpi_size=None, rank=None):
        super(OMNIGLOT, self).__init__(batch_size, test_batch_size, model, ctx_size, root, ddp, mpi_size, rank)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def prepare_data(self):
        omniglot_dset(self.root, train=True, download=True)
        omniglot_dset(self.root, train=False, download=True)

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            omniglot_full = omniglot_dset(self.root, train=True, transform=self.transforms)
            N = len(omniglot_full)
            print(f'{N} training images')
            if 'context' in self.model:
                omniglot_full = DCT_dataset(omniglot_full, self.ctx_size)
            self.train, self.val = random_split(omniglot_full, [N-1000, 1000])

        if stage == 'test' or stage is None:
            self.test = omniglot_dset(self.root, train=False, transform=self.transforms)
            print(f'{len(self.test)} test images')
            if 'context' in self.model:
                self.test = DCT_dataset(self.test, self.ctx_size)

