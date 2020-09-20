import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import gc
import torch.nn as nn


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


class DisModel(nn.Module):

    def __init__(self):
        super(DisModel, self).__init__()
        self.fc1 = nn.Linear(512, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)

        return x


def get_discriminative_model(input_shape):
    """
    The MLP model for discriminative active learning, without any regularization techniques.
    """

    layers = []
    width = 512
    layers += [nn.Linear(width, width), nn.ReLU()]
    layers += [nn.Linear(width, width), nn.ReLU()]
    layers += [nn.Linear(width, width), nn.ReLU()]
    layers += [nn.Linear(width, 2), nn.Softmax()]

    return nn.Sequential(*layers)
    # return DisModel()



def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='long')[y]


class DiscriminativeRepresentationSampling:
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
        self.sub_batches = 20

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

    def train_discriminative_model(self, labeled, unlabeled, input_shape):
        """
        A function that trains and returns a discriminative model on the labeled and unlabaled data.
        """

        # create the binary dataset:
        y_L = np.zeros((labeled.shape[0], 1), dtype='int')
        y_U = np.ones((unlabeled.shape[0], 1), dtype='int')
        X_train = np.vstack((labeled, unlabeled))
        Y_train = np.vstack((y_L, y_U))
        # Y_train = to_categorical(Y_train, 2)
        Y_train = np.squeeze(Y_train, axis=1)

        # build the model:
        model = get_discriminative_model(input_shape)
        model.train()
        model = model.to(self.device)

        # train the model:
        batch_size = 1024
        if np.max(input_shape) == 28:
            optimizer = optim.Adam(model.parameters(), lr=0.0003)
            epochs = 200
        elif np.max(input_shape) == 128:
            # optimizer = optimizers.Adam(lr=0.0003)
            # epochs = 200
            batch_size = 128
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            epochs = 1000  # TODO: was 200
        elif np.max(input_shape) == 512:
            optimizer = optim.Adam(model.parameters(), lr=0.0002)
            # optimizer = optimizers.RMSprop()
            epochs = 500
        elif np.max(input_shape) == 32:
            optimizer = optim.Adam(model.parameters(), lr=0.0003)
            epochs = 500
        else:
            optimizer = optim.Adam()
            # optimizer = optimizers.RMSprop()
            epochs = 1000
            batch_size = 32

        criterion = nn.CrossEntropyLoss(weight=torch.tensor([24.0, 1.0]).to(self.device))
        loader_te = DataLoader(self.test_handler(X_train, Y_train, transform=None),
                               shuffle=True, batch_size=batch_size)
        for e in range(epochs):
            acc = 0
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                # for param in model.parameters():
                #     print(param.grad.data.sum())

                # start debugger
                # import pdb; pdb.set_trace()
                optimizer.step()
                P = out.max(1)[1]
                acc += 1.0 * (P == y).sum().item() / len(y)
            acc /= len(loader_te)
            print('Epoch {}/{}'.format(e, epochs))
            print('Embedded model training accuracy {:.3f}'.format(acc * 100))
            if acc > 0.98:
                break

        return model

    def query(self, query_num):
        loader_te = DataLoader(self.test_handler(self.X, self.Y, transform=self.args['transform_te']),
                               shuffle=False, **self.args['loader_te_args'])
        self.fea.eval()

        # subsample from the unlabeled set:
        unlabeled_idx = np.arange(self.n_pool)[~self.idx_lb]
        # unlabeled_idx = np.random.choice(unlabeled_idx, np.min([labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)

        features = None
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                latent = self.fea(x)
                if features is None:
                    features = latent.cpu()
                else:
                    features = torch.cat((features, latent.cpu()))

        # iteratively sub-sample using the discriminative sampling routine:
        labeled_so_far = 0
        sub_sample_size = int(query_num / self.sub_batches)
        while labeled_so_far < query_num:
            if labeled_so_far + sub_sample_size > query_num:
                sub_sample_size = query_num - labeled_so_far

            model = self.train_discriminative_model(features[self.idx_lb], features[~self.idx_lb],
                                                    features[self.idx_lb][0].shape)

            loader_te = DataLoader(self.test_handler(features[~self.idx_lb], self.Y[~self.idx_lb],
                                                     transform=None),
                                   shuffle=False, **self.args['loader_te_args'])

            predictions = torch.zeros([len(self.Y[~self.idx_lb]), 2])
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    prob = model(x)
                    predictions[idxs] = prob.cpu()

            selected_indices = np.argpartition(predictions[:, 1], -sub_sample_size)[-sub_sample_size:]
            for i in selected_indices:
                if not self.idx_lb[unlabeled_idx[i]]:
                    self.idx_lb[unlabeled_idx[i]] = True
            labeled_so_far += sub_sample_size
            unlabeled_idx = np.arange(self.n_pool)[~self.idx_lb]
            # unlabeled_idx = np.random.choice(unlabeled_idx, np.min([labeled_idx.shape[0] * 10, unlabeled_idx.size]), replace=False)

            # delete the model to free GPU memory:
            del model
            gc.collect()
        gc.collect()

        return []
