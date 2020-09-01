import numpy as np
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


class Semi_Strategy:

    def __init__(self,X,Y,idx_lb,net_fea,net_clf,net_dis,train_handler,test_handler,args):
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
        self.idxs_lb  = idx_lb
        self.net_fea = net_fea
        self.net_clf = net_clf
        self.net_dis = net_dis
        self.train_handler = train_handler
        self.test_handler = test_handler
        self.args    = args
        self.n_pool  = len(Y)
        self.num_class = self.args['num_class']

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def query(self, n):

            pass

    def update(self, idxs_lb):

            self.idxs_lb = idxs_lb


    def train(self,alpha,total_epoch):

        """
        Using gans loss for adversarial training
        :param total_epoch:
        :return:
        """

        print("[Training] labeled and unlabeled data")
        n_epoch = total_epoch

        self.fea = self.net_fea().to(self.device)
        self.clf = self.net_clf().to(self.device)
        self.dis = self.net_dis().to(self.device)

        # setting three optimizers
        opt_fea = optim.SGD(self.fea.parameters(), **self.args['optimizer_args'])
        opt_clf = optim.SGD(self.clf.parameters(), **self.args['optimizer_args'])
        opt_dis = optim.SGD(self.dis.parameters(), **self.args['optimizer_args'])

        # setting idx_lb and idx_ulb
        idx_lb_train = np.arange(self.n_pool)[self.idxs_lb]
        idx_ulb_train = np.arange(self.n_pool)[~self.idxs_lb]

        # computing the unbalancing ratio, a value betwwen [0,1], generally 0.1 - 0.5
        gamma_ratio = len(idx_lb_train) / len(idx_ulb_train)

        # Data-loading (Redundant Trick)

        loader_tr = DataLoader(
            self.train_handler(self.X[idx_lb_train], self.Y[idx_lb_train], self.X[idx_ulb_train], self.Y[idx_ulb_train],
                               transform=self.args['transform_tr']), shuffle=True, **self.args['loader_tr_args'])


        for epoch in range(n_epoch):

            # setting the training mode in the beginning of EACH epoch
            # adversarial use BINARY ENTROPY (JS divergence) Loss for training
            # (since we need to compute the training accuracy during the epoch, optional)
            # The clf outputs two results, the final result, as well as the latent variable in the last layer


            self.fea.train()
            self.clf.train()
            self.dis.train()


            # Total_loss = 0
            # n_batch    = 0
            # acc        = 0

            for index, label_x, label_y, unlabel_x, _ in loader_tr:

                # n_batch += 1

                label_x, label_y = label_x.cuda(), label_y.cuda()
                unlabel_x = unlabel_x.cuda()
                btch_size = len(label_y)

                """
                There many ways to train the system, finally we used the GANs style approach for training the system

                """
                # training feature extractor and predictor

                set_requires_grad(self.fea, requires_grad=True)
                set_requires_grad(self.clf, requires_grad=True)
                set_requires_grad(self.dis, requires_grad=False)

                lb_z = self.fea(label_x)
                unlb_z = self.fea(unlabel_x)

                opt_fea.zero_grad()
                opt_clf.zero_grad()

                lb_out, _ = self.clf(lb_z)

                # prediction loss
                # pred_loss = torch.mean(F.cross_entropy(lb_out,label_y))

                pred_loss = F.cross_entropy(lb_out, label_y)


                # Discriminator loss (w.r.t. JS divergence in gans)
                # Minimize over unlabeled 1 vs labeled 0



                domain_pred_0 = self.dis(unlb_z)
                domain_pred_1 = self.dis(lb_z)

                domain_y_0 = torch.zeros_like(domain_pred_0, dtype = torch.float)
                domain_y_1 = torch.ones_like(domain_pred_1,  dtype=torch.float)

                domain_y_0 = domain_y_0.to(self.device)
                domain_y_1 = domain_y_1.to(self.device)


                JS_divergence = F.binary_cross_entropy(domain_pred_0,domain_y_0) + F.binary_cross_entropy(domain_pred_1,domain_y_1)


                loss = pred_loss + alpha * JS_divergence

                loss.backward()
                opt_fea.step()
                opt_clf.step()


                # secondly training the discriminator (maximize the loss, gd over -1 * JS divergence)

                set_requires_grad(self.fea, requires_grad=False)
                set_requires_grad(self.clf, requires_grad=False)
                set_requires_grad(self.dis, requires_grad=True)

                with torch.no_grad():

                    lb_z = self.fea(label_x)
                    unlb_z = self.fea(unlabel_x)

                for _ in range(1):

                    # gradient ascent for multiple times like GANS training
                    # there are better coding writing, here we only write the naive approach

                    domain_pred_0 = self.dis(unlb_z)
                    domain_pred_1 = self.dis(lb_z)

                    domain_y_0 = torch.zeros_like(domain_pred_0, dtype=torch.float)
                    domain_y_1 = torch.ones_like(domain_pred_1, dtype=torch.float)

                    domain_y_0 = domain_y_0.to(self.device)
                    domain_y_1 = domain_y_1.to(self.device)


                    dis_loss = -1 * alpha * (F.binary_cross_entropy(domain_pred_0, domain_y_0) + F.binary_cross_entropy(
                        domain_pred_1, domain_y_1))


                    opt_dis.zero_grad()
                    dis_loss.backward()
                    opt_dis.step()

                    # prediction and computing training accuracy and empirical loss under evaluation mode
                    # P = lb_out.max(1)[1]
                    # acc += 1.0 * (label_y == P).sum().item() / len(label_y)
                    # Total_loss += loss.item()

                # Total_loss /= n_batch
                # acc        /= n_batch

            print('==========Inner epoch {:d} ========'.format(epoch))
            # print('Training Loss {:.3f}'.format(Total_loss))
            # print('Training accuracy {:.3f}'.format(acc*100))

    def predict(self,X,Y):

        loader_te = DataLoader(self.test_handler(X, Y, transform=self.args['transform_te']),
                               shuffle=False, **self.args['loader_te_args'])

        self.fea.eval()
        self.clf.eval()

        P = torch.zeros(len(Y), dtype=Y.dtype)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                latent  = self.fea(x)
                out, _  = self.clf(latent)
                pred    = out.max(1)[1]
                P[idxs] = pred.cpu()

        return P


    def predict_prob(self,X,Y):

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
                out,_ = self.clf(latent)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()

        return probs


    def predict_prob_dropout(self,X,Y,n_drop):

        loader_te = DataLoader(self.test_handler(X, Y, transform=self.args['transform_te']),
                               shuffle=False, **self.args['loader_te_args'])

        self.fea.train()
        self.clf.train()

        probs = torch.zeros([len(Y), self.num_class])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i + 1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    e1  = self.fea(x)
                    out, _ = self.clf(e1)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop

        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(self.test_handler(X, Y, transform=self.args['transform_te']),
                               shuffle=False, **self.args['loader_te_args'])

        self.fea.train()
        self.clf.train()

        probs = torch.zeros([n_drop, len(Y), self.num_class])
        for i in range(n_drop):
            print('n_drop {}/{}'.format(i + 1, n_drop))
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device)
                    e1 = self.fea(x)
                    out, _ = self.clf(e1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()

        return probs

    def get_embedding(self, X, Y):

        loader_te = DataLoader(self.test_handler(X, Y, transform=self.args['transform_te']),
                               shuffle=False, **self.args['loader_te_args'])

        self.fea.eval()
        self.clf.eval()

        embedding = torch.zeros([len(Y), self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                e1 = self.fea(x)
                _, latent = self.clf(e1)
                embedding[idxs] = latent.cpu()

        return embedding









