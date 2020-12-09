import numpy as np
import torch
import math


# setting gradient values
from query_strategies.fixmatch import FixMatch


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


class FixMatchRandom(FixMatch):

    def __init__(self, X, Y, idx_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args):
        super(FixMatchRandom, self).__init__(X, Y, idx_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)

    def query(self, query_num):

        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]
        return idxs_unlabeled[torch.randperm(idxs_unlabeled.shape[0])[:query_num]]
