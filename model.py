import torch.nn as nn
import torch.nn.functional as F
import torch


def get_net(name):
    if name == 'FashionMNIST':
        return Net1
    elif name == 'SVHN':
        return VGG_svhn
    elif name == 'CIFAR10':
        return VGG_c10



class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

# class Net2(nn.Module):
#     def __init__(self):
#         super(Net2, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
#         self.conv3_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(1152, 400)
#         self.fc2 = nn.Linear(400, 50)
#         self.fc3 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
#         x = x.view(-1, 1152)
#         x = F.relu(self.fc1(x))
#         e1 = F.relu(self.fc2(x))
#         x = F.dropout(e1, training=self.training)
#         x = self.fc3(x)
#         return x, e1
#
#     def get_embedding_dim(self):
#         return 50

# class Net3(nn.Module):
#     def __init__(self):
#         super(Net3, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
#         self.fc1 = nn.Linear(1024, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(F.max_pool2d(self.conv3(x), 2))
#         x = x.view(-1, 1024)
#         e1 = F.relu(self.fc1(x))
#         x = F.dropout(e1, training=self.training)
#         x = self.fc2(x)
#         return x, e1
#
#     def get_embedding_dim(self):
#         return 50


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

## VGG for CIFAR 10 and CIFAR 100

# class VGGFeat(nn.Module):
#
#
#     def __init__(self):
#         super(VGGFeat, self).__init__()
#         self.features = self._make_layers(cfg['VGG16'])
#         self.fc1 = nn.Linear(512, 50)
#         self.fc2 = nn.Linear(50,10)
#
#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         e1  = F.relu(self.fc1(out))
#         out = F.dropout(e1,training=self.training)
#         out = self.fc2(out)
#         return out, e1
#
#     def get_embedding_dim(self):
#
#         return 50
#
#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)


class VGG_svhn(nn.Module):


    def __init__(self):
        super(VGG_svhn, self).__init__()
        self.features = self._make_layers(cfg['VGG11'])
        self.fc1 = nn.Linear(512, 50)
        self.fc2 = nn.Linear(50,10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        e1  = F.relu(self.fc1(out))
        out = F.dropout(e1,training=self.training)
        out = self.fc2(out)
        return out, e1

    def get_embedding_dim(self):

        return 50

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



class VGG_c10(nn.Module):


    def __init__(self):
        super(VGG_c10, self).__init__()
        self.features = self._make_layers(cfg['VGG16'])
        self.fc1 = nn.Linear(512, 50)
        self.fc2 = nn.Linear(50,10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        e1  = F.relu(self.fc1(out))
        out = F.dropout(e1,training=self.training)
        out = self.fc2(out)
        return out, e1

    def get_embedding_dim(self):

        return 50

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



