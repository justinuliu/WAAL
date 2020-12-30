import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import sys

# setting gradient values
from model_WA import Discriminator
from query_strategies.fixmatch import FixMatch


class FixMatchDis(FixMatch):

    def __init__(self, X, Y, idx_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args):
        super(FixMatchDis, self).__init__(X, Y, idx_lb, net_fea, net_clf, net_dis, train_handler,
                                              test_handler, args)

    def query(self, query_num):
        # setting idx_lb and idx_ulb
        idx_lb_train = np.arange(self.n_pool)[self.idx_lb]
        idx_ulb_train = np.arange(self.n_pool)[~self.idx_lb]
        discriminator = Discriminator(self.fea.fea_out).to(self.device)
        loader_tr = DataLoader(self.train_handler(self.X[idx_lb_train],self.Y[idx_lb_train],self.X[idx_ulb_train],
                                                  self.Y[idx_ulb_train], transform=self.args['transform_tr']),
                               shuffle=True, **self.args['loader_tr_args'])
        bce_loss = nn.BCELoss()
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
        self.fea.eval()
        if self.net_clf is not None:
            self.clf.eval()

        discriminator.train()
        # Training Discriminator
        for e in range(self.args['epochs_dis']):
            total_loss = 0.
            for index, label_x, _, unlabel_x, _ in loader_tr:
                label_x, unlabel_x = label_x.to(self.device), unlabel_x.to(self.device)
                mu = self.fea(label_x)
                unlab_mu = self.fea(unlabel_x)

                labeled_preds = torch.squeeze(discriminator(mu))
                unlabeled_preds = torch.squeeze(discriminator(unlab_mu))

                lab_real_preds = torch.ones(label_x.size(0))
                unlab_fake_preds = torch.zeros(unlabel_x.size(0))

                lab_real_preds = lab_real_preds.to(self.device)
                unlab_fake_preds = unlab_fake_preds.to(self.device)

                dsc_loss = bce_loss(labeled_preds, lab_real_preds) + bce_loss(unlabeled_preds, unlab_fake_preds)
                optim_discriminator.zero_grad()
                dsc_loss.backward()
                optim_discriminator.step()
            sys.stdout.write('\r')
            sys.stdout.write('Current discriminator model loss: {:.8f}'.format(total_loss/len(loader_tr)))
            sys.stdout.write('\n')

        # Querying
        discriminator.eval()
        loader_te = DataLoader(self.test_handler(self.X[idx_ulb_train], self.Y[idx_ulb_train], transform=self.args['transform_te']),
                               shuffle=False, **self.args['loader_te_args'])
        repr_probs = torch.zeros(len(idx_ulb_train))
        with torch.no_grad():
            for x, _, idxs in loader_te:
                x = x.to(self.device)
                latent = self.fea(x)
                prob = discriminator(latent)
                prob = torch.squeeze(prob, dim=1)
                repr_probs[idxs] = prob.cpu()

        query_repr = idx_ulb_train[repr_probs.sort()[1][:query_num]]
        return query_repr
