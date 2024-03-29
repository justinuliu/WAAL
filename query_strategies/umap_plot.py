import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import umap
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt


# setting gradient values
def set_requires_grad(model, requires_grad=True):
    """
    Used in training adversarial approach
    :param model:
    :param requires_grad:
    :return:
    """

    for param in model.parameters():
        param.requires_grad = requires_grad


def learning_rate(init, epoch, total_epoch):
    optimal_factor = 0
    p = 1. * epoch / total_epoch;
    if p >= 0.75:
        optimal_factor = 2
    elif p >= 0.5:
        optimal_factor = 1

    return init * math.pow(0.1, optimal_factor)


class UmapPlot:

    def __init__(self, X, Y, idx_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args):
        """

        :param X:
        :param Y:
        :param idx_lb:
        :param net_fea:
        :param net_clf:
        :param net_dis:
        :param train_handler: generate a dataset in the training procedure, since training requires two datasets, the returning value
                                looks like a (index, x_dis1, y_dis1, x_dis2, y_dis2)
        :param test_handler: generate a dataset for the prediction, only requires one dataset
        :param args:
        """

        self.X = X
        self.Y = Y
        self.idx_lb = idx_lb
        self.net_fea = net_fea
        self.net_clf = net_clf
        self.train_handler = train_handler
        self.test_handler = test_handler
        self.args = args

        self.n_pool = len(Y)
        self.num_class = self.args['num_class']
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def update(self, idx_lb):

        self.idx_lb = idx_lb

    def train(self, alpha, total_epoch):

        """
        Only training samples with labeled and unlabeled data-set
        alpha is the trade-off between the empirical loss and error, the more interaction, the smaller \alpha
        :return:
        """

        print("[Training] labeled and unlabeled data")
        n_epoch = total_epoch

        self.fea = self.net_fea().to(self.device)
        self.clf = self.net_clf().to(self.device)

        # setting idx_lb and idx_ulb
        idx_lb_train = np.arange(self.n_pool)[self.idx_lb]

        # Data-loading (Redundant Trick)

        loader_tr = DataLoader(
            self.test_handler(self.X[idx_lb_train], self.Y[idx_lb_train],
                              transform=self.args['transform_tr']), shuffle=True, **self.args['loader_tr_args'])

        for epoch in range(n_epoch):

            # setting three optimizers
            print("lr=%f" % learning_rate(self.args['optimizer_args']['lr'], epoch, total_epoch))
            opt_fea = optim.SGD(self.fea.parameters(),
                                lr=learning_rate(self.args['optimizer_args']['lr'], epoch, total_epoch),
                                momentum=self.args['optimizer_args']['momentum'])
            opt_clf = optim.SGD(self.clf.parameters(),
                                lr=learning_rate(self.args['optimizer_args']['lr'], epoch, total_epoch),
                                momentum=self.args['optimizer_args']['momentum'])

            # setting the training mode in the beginning of EACH epoch
            # (since we need to compute the training accuracy during the epoch, optional)

            self.fea.train()
            self.clf.train()

            Total_loss = 0
            n_batch = 0
            acc = 0

            for label_x, label_y, index in loader_tr:
                n_batch += 1

                label_x, label_y = label_x.cuda(), label_y.cuda()

                # training feature extractor and predictor

                set_requires_grad(self.fea, requires_grad=True)
                set_requires_grad(self.clf, requires_grad=True)

                lb_z = self.fea(label_x)

                opt_fea.zero_grad()
                opt_clf.zero_grad()

                lb_out, _ = self.clf(lb_z)

                # prediction loss (deafult we use F.cross_entropy)
                loss = torch.mean(F.cross_entropy(lb_out, label_y))
                loss.backward()
                opt_fea.step()
                opt_clf.step()

                # prediction and computing training accuracy and empirical loss under evaluation mode
                P = lb_out.max(1)[1]
                acc += 1.0 * (label_y == P).sum().item() / len(label_y)
                Total_loss += loss.item()

            Total_loss /= n_batch
            acc /= n_batch

            print('==========Inner epoch {:d} ========'.format(epoch))
            print('Training Loss {:.3f}'.format(Total_loss))
            print('Training accuracy {:.3f}'.format(acc * 100))
        self.plot(self.X, self.Y)

    def plot(self, X, Y):
        loader_te = DataLoader(self.test_handler(X, Y, transform=self.args['transform_te']),
                               shuffle=False, **self.args['loader_te_args'])
        self.fea.eval()

        features = None
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                latent = self.fea(x)
                if features is None:
                    features = latent.cpu()
                else:
                    features = torch.cat((features, latent.cpu()))

        features_file_name = 'data/features.npy'
        targets_file_name = 'data/targets.npy'
        np.save(features_file_name, features)
        np.save(targets_file_name, Y)

        embedding = umap.UMAP(n_neighbors=5, min_dist=0.05).fit_transform(features)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=Y, cmap='Spectral', s=1)
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
        plt.title('UMAP projection of the CIFAR-10 dataset', fontsize=24)
        plt.show()

    def predict(self, X, Y):

        loader_te = DataLoader(self.test_handler(X, Y, transform=self.args['transform_te']),
                               shuffle=False, **self.args['loader_te_args'])

        self.fea.eval()
        self.clf.eval()

        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                latent = self.fea(x)
                out, _ = self.clf(latent)
                pred = out.max(1)[1]
                P[idxs] = pred.cpu()

        return P

    def predict_prob(self, X, Y):

        """
        prediction output score probability
        :param X:
        :param Y: NEVER USE the Y information for direct prediction
        :return:
        """

        loader_te = DataLoader(self.test_handler(X, Y, transform=self.args['transform_te']),
                               shuffle=False, **self.args['loader_te_args'])

        self.fea.eval()
        self.clf.eval()

        probs = torch.zeros([len(Y), self.num_class])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                latent = self.fea(x)
                out, _ = self.clf(latent)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()

        return probs

    def query(self, query_num):
        return []

