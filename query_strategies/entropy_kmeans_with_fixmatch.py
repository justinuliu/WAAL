import numpy as np
from query_strategies.fixmatch import FixMatch
import torch.distributions as distributions
from sklearn.cluster import KMeans


class FixMatchEntropyKMeans(FixMatch):

    def __init__(self, X, Y, idx_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args):
        super(FixMatchEntropyKMeans, self).__init__(X, Y, idx_lb, net_fea, net_clf, net_dis, train_handler,
                                                   test_handler, args)

    def query(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]
        num_per_cluster = self.args['num_per_cluster']
        num_clusters = query_num//num_per_cluster

        model = KMeans(n_clusters=num_clusters, max_iter=1)
        if self.args['use_raw_pixel']:
            points = np.reshape(self.X[idxs_unlabeled], (len(idxs_unlabeled), -1))
        else:
            points = self.extract_feature(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        model.fit(points)

        query_list = []
        for i in range(num_clusters):
            idxs_cluster=idxs_unlabeled[model.labels_==i]
            probs = self.predict_prob(self.X[idxs_cluster], self.Y[idxs_cluster])
            entropy_score = distributions.categorical.Categorical(probs).entropy()
            idx = entropy_score.sort(descending=True)[1][:num_per_cluster]
            q = idxs_cluster[idx.tolist()]
            query_list += list(q)

        if len(query_list) < query_num:
            n = query_num - len(query_list)
            idxs_unlabeled = np.setdiff1d(idxs_unlabeled, np.array(query_list))
            probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
            entropy_score = distributions.categorical.Categorical(probs).entropy()
            idx = entropy_score.sort(descending=True)[1][:n]
            q = idxs_unlabeled[idx.tolist()]
            query_list += list(q)

        return query_list
