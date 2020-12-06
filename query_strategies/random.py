import numpy as np
import torch
from query_strategies.supervised import Supervised


class Random(Supervised):

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
        super(Random, self).__init__(X, Y, idx_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)

    def query(self, query_num):

        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]
        return idxs_unlabeled[torch.randperm(idxs_unlabeled.shape[0])[:query_num]]
