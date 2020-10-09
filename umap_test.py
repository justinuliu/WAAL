import torchvision
import umap.plot
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

from dataset_WA import get_dataset
import torch

# digits = load_digits()


X_tr, Y_tr, X_te, Y_te = get_dataset('CIFAR10')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)
latent = np.zeros((50000, 3*32*32))
for i, (inputs, targets) in enumerate(dataloader):
    latent[i] = torch.flatten(inputs, 1)

embedding = umap.UMAP(n_neighbors=5, min_dist=0.05).fit_transform(latent)
plt.scatter(embedding[:, 0], embedding[:, 1], c=Y_tr, cmap='Spectral', s=1)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the CIFAR-10 dataset', fontsize=24)
plt.show()
pass
