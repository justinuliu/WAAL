import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


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


class Strategy:
    def __init__(self, X, Y, idxs_lb, net_class, handler, args):
        """

        :param X:
        :param Y:
        :param idx_lb:
        :param net_fea:
        :param net_clf:
        :param net_dis:
        :param train_handler:
        :param test_handler:
        :param args:
        """
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.net_class = net_class
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.num_class = self.args['num_class']

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def query(self, n):

        pass

    def update(self, idxs_lb):

        self.idxs_lb = idxs_lb

    def train(self, total_epoch):

        """
        Using gans loss for adversarial training
        :param total_epoch:
        :return:
        """

        print("[Training] labeled and unlabeled data")
        n_epoch = total_epoch

        self.net = self.net_class().to(self.device)

        # setting three optimizers
        opt = optim.SGD(self.net.parameters(), **self.args['optimizer_args'])

        # Data-loading (Redundant Trick)

        loader_tr = DataLoader(
            self.handler(self.X[self.idxs_lb], self.Y[self.idxs_lb], transform=self.args['transform_tr']),
            shuffle=True, **self.args['loader_tr_args'])

        for epoch in range(n_epoch):

            self.net.train()

            # Total_loss = 0
            # n_batch    = 0
            # acc        = 0

            for label_x, label_y, _ in loader_tr:

                # n_batch += 1

                label_x, label_y = label_x.to(self.device), label_y.to(self.device)

                """
                There many ways to train the system, finally we used the GANs style approach for training the system

                """
                # training feature extractor and predictor

                set_requires_grad(self.net, requires_grad=True)

                lb_out, _ = self.net(label_x)

                opt.zero_grad()

                # prediction loss
                # pred_loss = torch.mean(F.cross_entropy(lb_out,label_y))

                loss = F.cross_entropy(lb_out, label_y)

                loss.backward()
                opt.step()

                # Total_loss /= n_batch
                # acc        /= n_batch

            print('==========Inner epoch {:d} ========'.format(epoch))
            # print('Training Loss {:.3f}'.format(Total_loss))
            # print('Training accuracy {:.3f}'.format(acc*100))

    def predict(self, X, Y):

        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform_te']),
                               shuffle=False, **self.args['loader_te_args'])

        self.net.eval()

        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.net(x)
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

        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform_te']),
                               shuffle=False, **self.args['loader_te_args'])

        self.net.eval()

        probs = torch.zeros([len(Y), self.num_class])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, _ = self.net(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()

        return probs

    def predict_prob_dropout(self, X, Y, n_drop):

        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform_te']),
                               shuffle=False, **self.args['loader_te_args'])

        self.net.train()

        probs = torch.zeros([len(Y), self.num_class])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i + 1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, _ = self.net(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop

        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform_te']),
                               shuffle=False, **self.args['loader_te_args'])

        self.net.train()

        probs = torch.zeros([n_drop, len(Y), self.num_class])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i + 1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    out, _ = self.net(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()

        return probs

    def get_embedding(self, X, Y):

        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transform_te']),
                               shuffle=False, **self.args['loader_te_args'])

        self.net.eval()

        embedding = torch.zeros([len(Y), self.net.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                _, latent = self.net(x)
                embedding[idxs] = latent.cpu()

        return embedding
