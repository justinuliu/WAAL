import os

from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import utils, ImageFolder
from collections import defaultdict
from shutil import copy
from torchvision.datasets.folder import default_loader

"""
This code mainly tests the redundancy trick, different from only using the smaller one 
to make the batch, here instead we used the max len as the data to make the batch

"""

def get_dataset(name):


    if name == 'FashionMNIST':
        return get_FashionMNIST()
    elif name == 'SVHN':
        return get_SVHN()
    elif name == 'CIFAR10':
        return get_CIFAR10()
    elif name == 'Food101':
        return get_food101()


def get_FashionMNIST():
    raw_tr = datasets.FashionMNIST('data/FashionMNIST', train=True, download=True)
    raw_te = datasets.FashionMNIST('data/FashionMNIST', train=False, download=True)
    X_tr = raw_tr.train_data
    Y_tr = raw_tr.train_labels
    X_te = raw_te.test_data
    Y_te = raw_te.test_labels
    return X_tr, Y_tr, X_te, Y_te


def get_SVHN():
    data_tr = datasets.SVHN('data/SVHN', split='train', download=True)
    data_te = datasets.SVHN('data/SVHN', split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te


def get_CIFAR10():
    data_tr = datasets.CIFAR10('data/CIFAR10', train=True, download=True)
    data_te = datasets.CIFAR10('data/CIFAR10', train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te


def get_food101():
    data_tr = Food101('data', train=True, download=True)
    data_te = Food101('data', train=False, download=True)
    X_tr = np.array([x[0] for x in data_tr.samples])
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = np.array([x[0] for x in data_te.samples])
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te


def get_handler(name):
    if name == 'FashionMNIST':
        return  Wa_datahandler1
    elif name == 'SVHN':
        return Wa_datahandler2
    elif name == 'CIFAR10':
        return Wa_datahandler3
    elif name == 'Food101':
        return Wa_datahandler4


class Wa_datahandler1(Dataset):

    def __init__(self,X_1, Y_1, X_2, Y_2, transform = None):
        """

        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1),len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        if self.transform is not None:

            x_1 = Image.fromarray(x_1.numpy(), mode='L')
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(x_2.numpy(), mode='L')
            x_2 = self.transform(x_2)

        return index,x_1,y_1,x_2,y_2



class Wa_datahandler2(Dataset):

    def __init__(self,X_1, Y_1, X_2, Y_2, transform = None):
        """

        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1),len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        if self.transform is not None:

            x_1 = Image.fromarray(np.transpose(x_1, (1, 2, 0)))
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(np.transpose(x_2, (1, 2, 0)))
            x_2 = self.transform(x_2)

        return index,x_1,y_1,x_2,y_2


class Wa_datahandler3(Dataset):

    def __init__(self,X_1, Y_1, X_2, Y_2, transform = None):
        """

        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1),len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        if self.transform is not None:

            x_1 = Image.fromarray(x_1)
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(x_2)
            x_2 = self.transform(x_2)

        return index,x_1,y_1,x_2,y_2


class Wa_datahandler4(Dataset):

    def __init__(self, X_1, Y_1, X_2, Y_2, transform=None):
        """

        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform
        self.loader = default_loader

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1), len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        x_1 = self.loader(x_1)
        x_2 = self.loader(x_2)

        if self.transform is not None:
            x_1 = self.transform(x_1)
            x_2 = self.transform(x_2)

        return index, x_1, y_1, x_2, y_2


class Food101(ImageFolder):
    base_folder = 'food-101'
    url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    filename = "food-101.tar.gz"
    tgz_md5 = '85eeb15f3717b99a5da872d97d918f87'
    raw_folder = 'images'
    meta = {
        'folder': 'meta',
        'classes': 'classes.txt',
        'labels': 'labels.txt',
        'train': 'train.txt',
        'test': 'test.txt',
    }

    def _check_validation(self, file):
        return any(x in file for x in self.sample_paths)

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        self.root = root
        self.train = train
        self.split = 'train' if self.train else 'test'
        self._load_meta()
        if download:
            self._download()
        self._prepare_data(os.path.join(self.root, self.base_folder, self.meta["folder"], self.meta[self.split]),
                           os.path.join(self.root, self.base_folder, self.raw_folder),
                           os.path.join(self.root, self.base_folder, self.split))
        super(Food101, self).__init__(self.data_path, transform=transform, target_transform=target_transform)
        pass

    def _load_meta(self):
        self.data_path = os.path.join(self.root, self.base_folder, self.split)
        self.meta_path = os.path.join(self.root, self.base_folder, self.meta['folder'])
        with open(os.path.join(self.meta_path, self.meta['classes']), 'r') as infile:
            self.classes = infile.read().splitlines()
        with open(os.path.join(self.meta_path, self.meta['labels']), 'r') as infile:
            self.labels = infile.read().splitlines()
        with open(os.path.join(self.meta_path, self.meta[self.split]), 'r') as infile:
            self.sample_paths = infile.read().splitlines()
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _check_integrity(self):
        root = self.root
        fpath = os.path.join(root, self.filename)
        if not utils.check_integrity(fpath, self.tgz_md5):
            return False
        return True

    def _download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        utils.download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def _prepare_data(self, filepath, src, dest):
        if os.path.exists(dest):
            print("%s folder already exists" % dest)
            return
        classes_images = defaultdict(list)
        with open(filepath, 'r') as txt:
            paths = [read.strip() for read in txt.readlines()]
            for p in paths:
                food = p.split('/')
                classes_images[food[0]].append(food[1] + '.jpg')

        for food in classes_images.keys():
            if not os.path.exists(os.path.join(dest, food)):
                os.makedirs(os.path.join(dest, food))
            for i in classes_images[food]:
                copy(os.path.join(src, food, i), os.path.join(dest, food, i))
        print("Copying Done!")



