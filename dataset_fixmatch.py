import torch
from torchvision import transforms
from autoaugment import *


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class TransformFixCIFAR(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.strong = transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
        ])
        self.normalize = transforms.Compose([
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformFixSVHN(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ])
        self.strong = transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
            SVHNPolicy(),
            transforms.ToTensor(),
            Cutout(n_holes=1, length=20),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
        ])
        self.normalize = transforms.Compose([
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformFixFashionMNIST(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.strong = transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Compose([
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)
