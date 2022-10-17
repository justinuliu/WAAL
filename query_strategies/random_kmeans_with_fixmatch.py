import numpy as np
import numpy.random

from query_strategies.fixmatch import FixMatch
from sklearn.cluster import KMeans

import collections

class FixMatchRandomKMeans(FixMatch):

    def __init__(self, X, Y, idx_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args):
        super(FixMatchRandomKMeans, self).__init__(X, Y, idx_lb, net_fea, net_clf, net_dis, train_handler,
                                                   test_handler, args)

    def query(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]
        num_per_cluster = self.args['num_per_cluster']
        num_clusters = query_num//num_per_cluster

        model = KMeans(n_clusters=num_clusters)
        if self.args['use_raw_pixel']:
            points = np.reshape(self.X[idxs_unlabeled], (len(idxs_unlabeled), -1))
        else:
            points = self.extract_feature(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        model.fit(points)

        query_list = []
        for i in range(num_clusters):
            idxs_cluster = idxs_unlabeled[model.labels_ == i]
            idxs = numpy.random.choice(idxs_cluster, num_per_cluster, replace=False)
            query_list += list(idxs)

        print(f'size of query_list: {len(query_list)}')
        print(f'size of idxs_unlabeled: {len(idxs_unlabeled)}')
        print('Duplicate index: ')
        print([item for item, count in collections.Counter(query_list).items() if count > 1])

        if len(query_list) < query_num:
            n = query_num - len(query_list)
            idxs = numpy.random.choice([i for i in idxs_unlabeled if i not in query_list], n)
            query_list += list(idxs)

        return query_list
