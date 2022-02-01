import numpy as np
from query_strategies.fixmatch import FixMatch
import torch


class FixMatchMargin(FixMatch):

    def __init__(self, X, Y, idx_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args):
        super(FixMatchMargin, self).__init__(X, Y, idx_lb, net_fea, net_clf, net_dis, train_handler,
                                                   test_handler, args)

    def query(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]

        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        top2, _ = torch.topk(probs, 2, dim=1)
        score = 1 - (top2[:, 0] - top2[:, 1])
        idxs = score.sort(descending=True)[1][:query_num]
        return idxs_unlabeled[idxs]
