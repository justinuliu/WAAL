import numpy as np
import torch.distributions as distributions
from query_strategies.fixmatch import FixMatch


class FixMatchEntropy(FixMatch):

    def __init__(self, X, Y, idx_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args):
        super(FixMatchEntropy, self).__init__(X, Y, idx_lb, net_fea, net_clf, net_dis, train_handler,
                                              test_handler, args)

    def query(self, query_num):

        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]

        # prediction output probability
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        entropy_score = distributions.categorical.Categorical(probs).entropy()
        return idxs_unlabeled[entropy_score.sort(descending=True)[1][:query_num]]
