import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# setting gradient values
from dataset_fixmatch import TransformMultipleTimes
from query_strategies.fixmatch import FixMatch


class FixMatchFarthestFirst(FixMatch):

    def __init__(self, X, Y, idx_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args):
        super(FixMatchFarthestFirst, self).__init__(X, Y, idx_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)

    def cross_entropy(self, predicted, target):
        return -(target * torch.log(predicted)).sum(dim=1)

    def weak_to_orignal_cross_entropy(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]

        loader_orig = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                   transform=self.args['transform_te']),
                                 shuffle=False, **self.args['loader_te_args'])
        loader_aug = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                  transform=TransformMultipleTimes(self.args['transform_w'], self.args['K'])),
                                shuffle=False, **self.args['loader_te_args'])

        loader = zip(loader_orig, loader_aug)

        self.fea.eval()
        if self.net_clf is not None:
            self.clf.eval()

        with torch.no_grad():
            score = torch.zeros(len(idxs_unlabeled), device=self.device)
            for (input_orig, _, idxs_orig), (inputs_aug, _, idxs_aug) in loader:
                input_orig = input_orig.to(self.device)
                latent_orig = self.fea(input_orig)
                out_orig, _ = self.clf(latent_orig) if self.net_clf is not None else (latent_orig, None)
                probs_orig = F.softmax(out_orig, dim=1)
                for input_aug in inputs_aug:
                    input_aug = input_aug.to(self.device)
                    latent_aug = self.fea(input_aug)
                    out_aug, _ = self.clf(latent_aug) if self.net_clf is not None else (latent_aug, None)
                    probs_aug = F.softmax(out_aug, dim=1)
                    ce = self.cross_entropy(probs_aug, probs_orig)
                    score[idxs_orig] += ce

        score = score.cpu()
        return idxs_unlabeled[score.sort(descending=True)[1][:query_num]]

    def weak_to_orignal_distance(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]

        loader_orig = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                   transform=self.args['transform_te']),
                                 shuffle=False, **self.args['loader_te_args'])
        loader_aug = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                  transform=TransformMultipleTimes(self.args['transform_w'], self.args['K'])),
                                shuffle=False, **self.args['loader_te_args'])

        loader = zip(loader_orig, loader_aug)

        self.fea.eval()
        if self.net_clf is not None:
            self.clf.eval()

        pdist = torch.nn.PairwiseDistance(p=2)
        with torch.no_grad():
            score = torch.zeros(len(idxs_unlabeled), device=self.device)
            for (input_orig, _, idxs_orig), (inputs_aug, _, idxs_aug) in loader:
                input_orig = input_orig.to(self.device)
                latent_orig = self.fea(input_orig)
                out_orig, _ = self.clf(latent_orig) if self.net_clf is not None else (latent_orig, None)
                probs_orig = F.softmax(out_orig, dim=1)
                for input_aug in inputs_aug:
                    input_aug = input_aug.to(self.device)
                    latent_aug = self.fea(input_aug)
                    out_aug, _ = self.clf(latent_aug) if self.net_clf is not None else (latent_aug, None)
                    probs_aug = F.softmax(out_aug, dim=1)
                    dist = pdist(probs_aug, probs_orig)
                    score[idxs_orig] += dist

        score = score.cpu()
        return idxs_unlabeled[score.sort(descending=True)[1][:query_num]]

    def weak_to_orignal_distance_max(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]

        loader_orig = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                   transform=self.args['transform_te']),
                                 shuffle=False, **self.args['loader_te_args'])
        loader_aug = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                  transform=TransformMultipleTimes(self.args['transform_w'], self.args['K'])),
                                shuffle=False, **self.args['loader_te_args'])

        loader = zip(loader_orig, loader_aug)

        self.fea.eval()
        if self.net_clf is not None:
            self.clf.eval()

        pdist = torch.nn.PairwiseDistance(p=2)
        with torch.no_grad():
            score = torch.zeros(len(idxs_unlabeled), device=self.device)
            for (input_orig, _, idxs_orig), (inputs_aug, _, idxs_aug) in loader:
                input_orig = input_orig.to(self.device)
                latent_orig = self.fea(input_orig)
                out_orig, _ = self.clf(latent_orig) if self.net_clf is not None else (latent_orig, None)
                probs_orig = F.softmax(out_orig, dim=1)
                for input_aug in inputs_aug:
                    input_aug = input_aug.to(self.device)
                    latent_aug = self.fea(input_aug)
                    out_aug, _ = self.clf(latent_aug) if self.net_clf is not None else (latent_aug, None)
                    probs_aug = F.softmax(out_aug, dim=1)
                    dist = pdist(probs_aug, probs_orig)
                    score[idxs_orig] = torch.max(score[idxs_orig], dist)

        score = score.cpu()
        return idxs_unlabeled[score.sort(descending=True)[1][:query_num]]

    def weak_internal_variance(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]

        loader_aug = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                  transform=TransformMultipleTimes(self.args['transform_w'], self.args['K'])),
                                shuffle=False, **self.args['loader_te_args'])

        self.fea.eval()
        if self.net_clf is not None:
            self.clf.eval()

        with torch.no_grad():
            score = torch.zeros(len(idxs_unlabeled), device=self.device)
            for inputs_aug, _, idxs_aug in loader_aug:
                probs = torch.zeros((len(inputs_aug), len(idxs_aug), self.args['num_class']), device=self.device)
                for input_aug, i in zip(inputs_aug, range(len(inputs_aug))):
                    input_aug = input_aug.to(self.device)
                    latent_aug = self.fea(input_aug)
                    out_aug, _ = self.clf(latent_aug) if self.net_clf is not None else (latent_aug, None)
                    probs_aug = F.softmax(out_aug, dim=1)
                    probs[i] = probs_aug
                score[idxs_aug] = torch.var(probs, dim=0).sum(dim=1)

        score = score.cpu()
        return idxs_unlabeled[score.sort(descending=True)[1][:query_num]]

    def strong_internal_variance(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]

        loader_aug = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                  transform=TransformMultipleTimes(self.args['transform_s'], self.args['K'])),
                                shuffle=False, **self.args['loader_te_args'])

        self.fea.eval()
        if self.net_clf is not None:
            self.clf.eval()

        with torch.no_grad():
            score = torch.zeros(len(idxs_unlabeled), device=self.device)
            for inputs_aug, _, idxs_aug in loader_aug:
                probs = torch.zeros((len(inputs_aug), len(idxs_aug), self.args['num_class']), device=self.device)
                for input_aug, i in zip(inputs_aug, range(len(inputs_aug))):
                    input_aug = input_aug.to(self.device)
                    latent_aug = self.fea(input_aug)
                    out_aug, _ = self.clf(latent_aug) if self.net_clf is not None else (latent_aug, None)
                    probs_aug = F.softmax(out_aug, dim=1)
                    probs[i] = probs_aug
                score[idxs_aug] = torch.var(probs, dim=0).sum(dim=1)

        score = score.cpu()
        return idxs_unlabeled[score.sort(descending=True)[1][:query_num]]

    def strong_to_original_distance(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]

        loader_orig = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                   transform=self.args['transform_te']),
                                 shuffle=False, **self.args['loader_te_args'])
        loader_aug = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                  transform=TransformMultipleTimes(self.args['transform_s'], self.args['K'])),
                                shuffle=False, **self.args['loader_te_args'])

        loader = zip(loader_orig, loader_aug)

        self.fea.eval()
        if self.net_clf is not None:
            self.clf.eval()

        pdist = torch.nn.PairwiseDistance(p=2)
        with torch.no_grad():
            score = torch.zeros(len(idxs_unlabeled), device=self.device)
            for (input_orig, _, idxs_orig), (inputs_aug, _, idxs_aug) in loader:
                input_orig = input_orig.to(self.device)
                latent_orig = self.fea(input_orig)
                out_orig, _ = self.clf(latent_orig) if self.net_clf is not None else (latent_orig, None)
                probs_orig = F.softmax(out_orig, dim=1)
                for input_aug in inputs_aug:
                    input_aug = input_aug.to(self.device)
                    latent_aug = self.fea(input_aug)
                    out_aug, _ = self.clf(latent_aug) if self.net_clf is not None else (latent_aug, None)
                    probs_aug = F.softmax(out_aug, dim=1)
                    dist = pdist(probs_aug, probs_orig)
                    score[idxs_orig] += dist

        score = score.cpu()
        return idxs_unlabeled[score.sort(descending=True)[1][:query_num]]

    def strong_to_original_distance_max(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]

        loader_orig = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                   transform=self.args['transform_te']),
                                 shuffle=False, **self.args['loader_te_args'])
        loader_aug = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                  transform=TransformMultipleTimes(self.args['transform_s'], self.args['K'])),
                                shuffle=False, **self.args['loader_te_args'])

        loader = zip(loader_orig, loader_aug)

        self.fea.eval()
        if self.net_clf is not None:
            self.clf.eval()

        pdist = torch.nn.PairwiseDistance(p=2)
        with torch.no_grad():
            score = torch.zeros(len(idxs_unlabeled), device=self.device)
            for (input_orig, _, idxs_orig), (inputs_aug, _, idxs_aug) in loader:
                input_orig = input_orig.to(self.device)
                latent_orig = self.fea(input_orig)
                out_orig, _ = self.clf(latent_orig)
                probs_orig = F.softmax(out_orig, dim=1)
                for input_aug in inputs_aug:
                    input_aug = input_aug.to(self.device)
                    latent_aug = self.fea(input_aug)
                    out_aug, _ = self.clf(latent_aug) if self.net_clf is not None else (latent_aug, None)
                    probs_aug = F.softmax(out_aug, dim=1)
                    dist = pdist(probs_aug, probs_orig)
                    score[idxs_orig] = torch.max(score[idxs_orig], dist)

        score = score.cpu()
        return idxs_unlabeled[score.sort(descending=True)[1][:query_num]]

    def strong_to_original_cross_entropy(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]

        loader_orig = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                   transform=self.args['transform_te']),
                                 shuffle=False, **self.args['loader_te_args'])
        loader_aug = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                  transform=TransformMultipleTimes(self.args['transform_s'], self.args['K'])),
                                shuffle=False, **self.args['loader_te_args'])

        loader = zip(loader_orig, loader_aug)

        self.fea.eval()
        if self.net_clf is not None:
            self.clf.eval()

        with torch.no_grad():
            score = torch.zeros(len(idxs_unlabeled), device=self.device)
            for (input_orig, _, idxs_orig), (inputs_aug, _, idxs_aug) in loader:
                input_orig = input_orig.to(self.device)
                latent_orig = self.fea(input_orig)
                out_orig, _ = self.clf(latent_orig) if self.net_clf is not None else (latent_orig, None)
                probs_orig = F.softmax(out_orig, dim=1)
                for input_aug in inputs_aug:
                    input_aug = input_aug.to(self.device)
                    latent_aug = self.fea(input_aug)
                    out_aug, _ = self.clf(latent_aug) if self.net_clf is not None else (latent_aug, None)
                    probs_aug = F.softmax(out_aug, dim=1)
                    ce = self.cross_entropy(probs_aug, probs_orig)
                    score[idxs_orig] += ce

        score = score.cpu()
        return idxs_unlabeled[score.sort(descending=True)[1][:query_num]]

    def strong_to_weak_cross_entropy(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]

        loader_orig = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                   transform=self.args['transform_w']),
                                 shuffle=False, **self.args['loader_te_args'])
        loader_aug = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                  transform=TransformMultipleTimes(self.args['transform_s'], self.args['K'])),
                                shuffle=False, **self.args['loader_te_args'])

        loader = zip(loader_orig, loader_aug)

        self.fea.eval()
        if self.net_clf is not None:
            self.clf.eval()

        with torch.no_grad():
            score = torch.zeros(len(idxs_unlabeled), device=self.device)
            for (input_orig, _, idxs_orig), (inputs_aug, _, idxs_aug) in loader:
                input_orig = input_orig.to(self.device)
                latent_orig = self.fea(input_orig)
                out_orig, _ = self.clf(latent_orig) if self.net_clf is not None else (latent_orig, None)
                probs_orig = F.softmax(out_orig, dim=1)
                for input_aug in inputs_aug:
                    input_aug = input_aug.to(self.device)
                    latent_aug = self.fea(input_aug)
                    out_aug, _ = self.clf(latent_aug) if self.net_clf is not None else (latent_aug, None)
                    probs_aug = F.softmax(out_aug, dim=1)
                    ce = self.cross_entropy(probs_aug, probs_orig)
                    score[idxs_orig] += ce

        score = score.cpu()
        return idxs_unlabeled[score.sort(descending=True)[1][:query_num]]

    def strong_to_weak_distance(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]

        loader_orig = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                   transform=self.args['transform_w']),
                                 shuffle=False, **self.args['loader_te_args'])
        loader_aug = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                  transform=TransformMultipleTimes(self.args['transform_s'], self.args['K'])),
                                shuffle=False, **self.args['loader_te_args'])

        loader = zip(loader_orig, loader_aug)

        self.fea.eval()
        if self.net_clf is not None:
            self.clf.eval()

        pdist = torch.nn.PairwiseDistance(p=2)
        with torch.no_grad():
            score = torch.zeros(len(idxs_unlabeled), device=self.device)
            for (input_orig, _, idxs_orig), (inputs_aug, _, idxs_aug) in loader:
                input_orig = input_orig.to(self.device)
                latent_orig = self.fea(input_orig)
                out_orig, _ = self.clf(latent_orig) if self.net_clf is not None else (latent_orig, None)
                probs_orig = F.softmax(out_orig, dim=1)
                for input_aug in inputs_aug:
                    input_aug = input_aug.to(self.device)
                    latent_aug = self.fea(input_aug)
                    out_aug, _ = self.clf(latent_aug) if self.net_clf is not None else (latent_aug, None)
                    probs_aug = F.softmax(out_aug, dim=1)
                    dist = pdist(probs_aug, probs_orig)
                    score[idxs_orig] += dist

        score = score.cpu()
        return idxs_unlabeled[score.sort(descending=True)[1][:query_num]]

    def strong_to_weak_distance_max(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]

        loader_orig = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                   transform=self.args['transform_w']),
                                 shuffle=False, **self.args['loader_te_args'])
        loader_aug = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                  transform=TransformMultipleTimes(self.args['transform_s'], self.args['K'])),
                                shuffle=False, **self.args['loader_te_args'])

        loader = zip(loader_orig, loader_aug)

        self.fea.eval()
        if self.net_clf is not None:
            self.clf.eval()

        pdist = torch.nn.PairwiseDistance(p=2)
        with torch.no_grad():
            score = torch.zeros(len(idxs_unlabeled), device=self.device)
            for (input_orig, _, idxs_orig), (inputs_aug, _, idxs_aug) in loader:
                input_orig = input_orig.to(self.device)
                latent_orig = self.fea(input_orig)
                out_orig, _ = self.clf(latent_orig) if self.net_clf is not None else (latent_orig, None)
                probs_orig = F.softmax(out_orig, dim=1)
                for input_aug in inputs_aug:
                    input_aug = input_aug.to(self.device)
                    latent_aug = self.fea(input_aug)
                    out_aug, _ = self.clf(latent_aug) if self.net_clf is not None else (latent_aug, None)
                    probs_aug = F.softmax(out_aug, dim=1)
                    dist = pdist(probs_aug, probs_orig)
                    score[idxs_orig] = torch.max(score[idxs_orig], dist)

        score = score.cpu()
        return idxs_unlabeled[score.sort(descending=True)[1][:query_num]]

    def weak_to_orignal_variance(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]

        loader_orig = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                   transform=self.args['transform_te']),
                                 shuffle=False, **self.args['loader_te_args'])
        loader_aug = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                  transform=TransformMultipleTimes(self.args['transform_w'], self.args['K'])),
                                shuffle=False, **self.args['loader_te_args'])

        loader = zip(loader_orig, loader_aug)

        self.fea.eval()
        if self.net_clf is not None:
            self.clf.eval()

        with torch.no_grad():
            score = torch.zeros(len(idxs_unlabeled), device=self.device)
            for (input_orig, _, idxs_orig), (inputs_aug, _, idxs_aug) in loader:
                probs = torch.zeros((self.args['K'] + 1, len(idxs_aug), self.args['num_class']), device=self.device)
                input_orig = input_orig.to(self.device)
                latent_orig = self.fea(input_orig)
                out_orig, _ = self.clf(latent_orig) if self.net_clf is not None else (latent_orig, None)
                probs_orig = F.softmax(out_orig, dim=1)
                probs[0] = probs_orig
                for input_aug, i in zip(inputs_aug, range(len(inputs_aug))):
                    input_aug = input_aug.to(self.device)
                    latent_aug = self.fea(input_aug)
                    out_aug, _ = self.clf(latent_aug) if self.net_clf is not None else (latent_aug, None)
                    probs_aug = F.softmax(out_aug, dim=1)
                    probs[i+1] = probs_aug
                score[idxs_aug] = torch.var(probs, dim=0).sum(dim=1)

        score = score.cpu()
        return idxs_unlabeled[score.sort(descending=True)[1][:query_num]]

    def strong_to_orignal_variance(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]

        loader_orig = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                   transform=self.args['transform_te']),
                                 shuffle=False, **self.args['loader_te_args'])
        loader_aug = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                  transform=TransformMultipleTimes(self.args['transform_s'], self.args['K'])),
                                shuffle=False, **self.args['loader_te_args'])

        loader = zip(loader_orig, loader_aug)

        self.fea.eval()
        if self.net_clf is not None:
            self.clf.eval()

        with torch.no_grad():
            score = torch.zeros(len(idxs_unlabeled), device=self.device)
            for (input_orig, _, idxs_orig), (inputs_aug, _, idxs_aug) in loader:
                probs = torch.zeros((self.args['K'] + 1, len(idxs_aug), self.args['num_class']), device=self.device)
                input_orig = input_orig.to(self.device)
                latent_orig = self.fea(input_orig)
                out_orig, _ = self.clf(latent_orig) if self.net_clf is not None else (latent_orig, None)
                probs_orig = F.softmax(out_orig, dim=1)
                probs[0] = probs_orig
                for input_aug, i in zip(inputs_aug, range(len(inputs_aug))):
                    input_aug = input_aug.to(self.device)
                    latent_aug = self.fea(input_aug)
                    out_aug, _ = self.clf(latent_aug) if self.net_clf is not None else (latent_aug, None)
                    probs_aug = F.softmax(out_aug, dim=1)
                    probs[i+1] = probs_aug
                score[idxs_aug] = torch.var(probs, dim=0).sum(dim=1)

        score = score.cpu()
        return idxs_unlabeled[score.sort(descending=True)[1][:query_num]]

    def strong_to_weak_variance(self, query_num):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idx_lb]

        loader_orig = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                   transform=self.args['transform_w']),
                                 shuffle=False, **self.args['loader_te_args'])
        loader_aug = DataLoader(self.test_handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled],
                                                  transform=TransformMultipleTimes(self.args['transform_s'], self.args['K'])),
                                shuffle=False, **self.args['loader_te_args'])

        loader = zip(loader_orig, loader_aug)

        self.fea.eval()
        if self.net_clf is not None:
            self.clf.eval()

        with torch.no_grad():
            score = torch.zeros(len(idxs_unlabeled), device=self.device)
            for (input_orig, _, idxs_orig), (inputs_aug, _, idxs_aug) in loader:
                probs = torch.zeros((self.args['K'] + 1, len(idxs_aug), self.args['num_class']), device=self.device)
                input_orig = input_orig.to(self.device)
                latent_orig = self.fea(input_orig)
                out_orig, _ = self.clf(latent_orig) if self.net_clf is not None else (latent_orig, None)
                probs_orig = F.softmax(out_orig, dim=1)
                probs[0] = probs_orig
                for input_aug, i in zip(inputs_aug, range(len(inputs_aug))):
                    input_aug = input_aug.to(self.device)
                    latent_aug = self.fea(input_aug)
                    out_aug, _ = self.clf(latent_aug) if self.net_clf is not None else (latent_aug, None)
                    probs_aug = F.softmax(out_aug, dim=1)
                    probs[i+1] = probs_aug
                score[idxs_aug] = torch.var(probs, dim=0).sum(dim=1)

        score = score.cpu()
        return idxs_unlabeled[score.sort(descending=True)[1][:query_num]]

    def query(self, query_num):
        if self.args['farthest_first_criterion'] == 'w_to_o_ce':
            return self.weak_to_orignal_cross_entropy(query_num)
        elif self.args['farthest_first_criterion'] == 'w_to_o_dist':
            return self.weak_to_orignal_distance(query_num)
        elif self.args['farthest_first_criterion'] == 'w_to_o_dist_m':
            return self.weak_to_orignal_distance_max(query_num)
        elif self.args['farthest_first_criterion'] == 'w_i_var':
            return self.weak_internal_variance(query_num)
        elif self.args['farthest_first_criterion'] == 's_to_o_dist':
            return self.strong_to_original_distance(query_num)
        elif self.args['farthest_first_criterion'] == 's_to_o_dist_m':
            return self.strong_to_original_distance_max(query_num)
        elif self.args['farthest_first_criterion'] == 's_to_o_ce':
            return self.strong_to_original_cross_entropy(query_num)
        elif self.args['farthest_first_criterion'] == 's_i_var':
            return self.strong_internal_variance(query_num)
        elif self.args['farthest_first_criterion'] == 's_to_w_dist':
            return self.strong_to_weak_distance(query_num)
        elif self.args['farthest_first_criterion'] == 's_to_w_dist_m':
            return self.strong_to_weak_distance_max(query_num)
        elif self.args['farthest_first_criterion'] == 's_to_w_ce':
            return self.strong_to_weak_cross_entropy(query_num)
        elif self.args['farthest_first_criterion'] == 'w_to_o_var':
            return self.weak_to_orignal_variance(query_num)
        elif self.args['farthest_first_criterion'] == 's_to_o_var':
            return self.strong_to_orignal_variance(query_num)
        elif self.args['farthest_first_criterion'] == 's_to_w_var':
            return self.strong_to_weak_variance(query_num)
        else:
            raise Exception()
