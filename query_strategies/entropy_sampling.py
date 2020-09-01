import numpy as np
import torch
from .semi_strategy import Semi_Strategy
from .strategy import Strategy


class EntropySampling(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args):
		super(EntropySampling, self).__init__(X, Y, idxs_lb, net, handler, args)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)
		return idxs_unlabeled[U.sort()[1][:n]]


class Semi_EntropySampling(Semi_Strategy):
	def __init__(self, X,Y,idx_lb,net_fea,net_clf,net_dis,train_handler,test_handler,args):
		super(Semi_EntropySampling, self).__init__(X,Y,idx_lb,net_fea,net_clf,net_dis,train_handler,test_handler,args)

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
		probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
		log_probs = torch.log(probs)
		U = (probs*log_probs).sum(1)
		return idxs_unlabeled[U.sort()[1][:n]]
