import numpy as np
import torch
import argparse
from autoaugment import SVHNPolicy, CIFAR10Policy, ImageNetPolicy
from dataset_WA import get_dataset, get_handler
import dataset
from model_WA import get_net
from torchvision import transforms
from query_strategies import WAAL, Entropy, Random, SWAAL, WAALFixMatch, WAALUncertainty, FarthestFirst, \
    FixMatchEntropy, FixMatchRandom, EntropySelfTraining, FarthestFirstEntropy, DiscriminativeRepresentationSampling, \
    LeastConfidence, FixMatchLeastConfidence, UmapPlot, KLDiv, FixMatchKLDiv, Discriminate, DisEntropyMixture, \
    FixMatchDisEntropyMixture, FixMatchDis, DisEntropyCombined, FixMatchDisEntropyCombined, FixMatchFarthestFirst
from dataset_fixmatch import TransformFixCIFAR, TransformFixSVHN, TransformFixFashionMNIST, Cutout, TransformFixFood101
from query_strategies.entropy_kmeans_with_fixmatch import FixMatchEntropyKMeans
from query_strategies.margin_kmeans_with_fixmatch import FixMatchMarginKMeans
from query_strategies.margin_with_fixmatch import FixMatchMargin
from query_strategies.random_kmeans_with_fixmatch import FixMatchRandomKMeans
from query_strategies.sup_entropy_with_fixmatch import FixMatchSupEntropy
from query_strategies.sup_least_confidence_with_fixmatch import FixMatchSupLeastConfidence


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True, type=str)
parser.add_argument("-d", "--dataset", required=True, type=str)
parser.add_argument("-s", "--seed", required=True, type=int)
parser.add_argument("-i", "--num_init", required=True, type=int)
parser.add_argument("-q", "--num_query", required=True, type=int)
parser.add_argument("-r", "--num_round", required=False, type=int, default=5)
parser.add_argument("-a", "--algorithm", required=True, type=str)
parser.add_argument("--lr", required=False, type=float, default=1e-2)
parser.add_argument("-e", "--epoch", required=True, type=int)

input_args = parser.parse_args()

args_pool = {
    'FashionMNIST':
        {
            'transform_tr': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            'transform_te': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
            'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
            'optimizer_args': {'lr': 0.01, 'momentum': 0.5, 'weight_decay': 5e-4},
            'num_class': 10,
            'transform_fixmatch': TransformFixFashionMNIST((0.1307,), (0.3081,)),
            'threshold': 0.95,
            'seed': 1,
            'epochs_dis': 10,
            'repr_portion': .4,
        },
    'SVHN':
        {
            'transform_tr': transforms.Compose([
                transforms.RandomCrop(size = 32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ]),
            'transform_te': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ]),
            'transform_w': transforms.Compose([
                transforms.RandomCrop(size=32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ]),
            'transform_s': transforms.Compose([
                transforms.RandomCrop(32, padding=4, fill=128),
                # fill parameter needs torchvision installed from source
                SVHNPolicy(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=20),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ]),
            'loader_tr_args': {'batch_size': 10, 'num_workers': 1},
            'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
            # 'optimizer_args': {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-3},  # for WRN-28-2, supervised
            # 'optimizer_args': {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4},  # for WRN-28-2, fixmatch
            'optimizer_args': {'lr': 0.01, 'momentum': 0.5, 'weight_decay': 0},  # for VGG16
            'num_class': 10,
            'transform_fixmatch': TransformFixSVHN((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            'threshold': 0.95,
            'seed': 5,
            'epochs_dis': 10,
            'repr_portion': .4,
            'farthest_first_criterion': 'w_i_var',
            'K': 2,
            'num_per_cluster': 2,
            'use_raw_pixel': False,
            # epochs for VGG16 for fixmatch 80
            # epochs for VGG16 for supervised 4000
            # epochs for WRN-28-2 for fixmatch 100
            # epochs for WRN-28-2 for supervised 1000
        },
    'CIFAR10':
        {
            'transform_tr': transforms.Compose([
                transforms.RandomCrop(size = 32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ]),
            'transform_te': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ]),
            'transform_w': transforms.Compose([
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ]),
            'transform_s': transforms.Compose([
                transforms.RandomCrop(32, padding=4, fill=128),
                # fill parameter needs torchvision installed from source
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                transforms.ToTensor(),
                Cutout(n_holes=1, length=16),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ]),
            'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
            'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
            # 'optimizer_args': {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-3},  # for WRN-28-2, supervised
            # 'optimizer_args': {'lr': 0.1, 'momentum': 0.3, 'weight_decay': 5e-4},  # for WRN-28-2, FixMatch
            'optimizer_args': {'lr': 0.01, 'momentum': 0.3, 'weight_decay': 0},  # for VGG16
            'num_class': 10,
            'transform_fixmatch': TransformFixCIFAR((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            'threshold': 0.95,
            'seed': 4,
            'epochs_dis': 5,
            'repr_portion': .4,
            'farthest_first_criterion': 's_to_o_var',
            'K': 2,
            'num_per_cluster': 4,
            'use_raw_pixel': False,
            # epochs for VGG16 for fixmatch 80
            # epochs for VGG16 for supervised 300
            # epochs for WRN-28-2 for fixmatch 100
            # epochs for WRN-28-2 for supervised 500
        },
    'Food101':
        {
            'transform_tr': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
            'transform_te': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
            'transform_w': transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
            'transform_s': transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
            'loader_tr_args': {'batch_size': 100, 'num_workers': 16},
            'loader_te_args': {'batch_size': 200, 'num_workers': 16},
            'optimizer_args': {'lr': 0.1, 'momentum': 0.3, 'weight_decay': 0},  # for EfficientNet-b0
            'num_class': 101,
            'transform_fixmatch': TransformFixFood101((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            'threshold': 0.95,
            'seed': 1,
            'epochs_dis': 5,
            'repr_portion': .4,
            'farthest_first_criterion': 'w_i_var',
            'K': 2,
            'num_per_cluster': 1,
            'use_raw_pixel': False,
            # epochs for fixmatch 110
            # epochs for supervised 300
        },
}

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def stratified_split_dataset(targets, num_labelled_samples, num_classes, seed=None, random=False):
    labelled_indices = []

    lenth = len(targets)
    idxs_lb = np.zeros(lenth, dtype=bool)

    if seed is not None:
        np.random.seed(seed)

    if random:
        idxs_tmp = np.arange(lenth)
        np.random.shuffle(idxs_tmp)
        idxs_lb[idxs_tmp[:num_labelled_samples]] = True
        return idxs_lb

    indices = np.random.permutation(len(targets)).tolist()

    if num_labelled_samples < len(targets):
        class_counters = list([0] * num_classes)
        max_counter = num_labelled_samples // num_classes
        for i in indices:
            dp = targets[i]
            if num_labelled_samples > sum(class_counters):
                y = dp
                c = class_counters[y]
                if c < max_counter:
                    class_counters[y] += 1
                    labelled_indices.append(i)
    else:
        labelled_indices = indices

    idxs_lb[labelled_indices] = True

    return idxs_lb

NUM_INIT_LB = input_args.num_init
NUM_QUERY = input_args.num_query
NUM_ROUND = input_args.num_round
DATA_NAME = input_args.dataset
QUERY_STRATEGY = input_args.algorithm  # Could be WAAL, SWAAL (WAAL without semi-supervised manner), Random, Entropy
# setting training parameters
MODEL_NAME = input_args.model
alpha = input_args.lr
epoch = input_args.epoch

args = args_pool[DATA_NAME]

args["seed"] = input_args.seed

# load dataset (Only using the first 50K)
X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME)
if DATA_NAME != 'Food101':
    X_tr = X_tr[:50000, :]
    Y_tr = Y_tr[:50000]

# loading neural network
net_fea, net_clf, net_dis = get_net(MODEL_NAME)

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(NUM_INIT_LB))
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
print('number of testing pool: {}'.format(n_test))
print('Model class: {}'.format(net_fea))

# Generate the initial labeled pool
idxs_lb = stratified_split_dataset(Y_tr, NUM_INIT_LB, args['num_class'], seed=args['seed'])

# here the training handlers and testing handlers are different
train_handler = get_handler(DATA_NAME)
test_handler = dataset.get_handler(DATA_NAME)

if QUERY_STRATEGY == 'WAAL':
    strategy = WAAL(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'Entropy':
    strategy = Entropy(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'Random':
    strategy = Random(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'SWAAL':
    strategy = SWAAL(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'WAALFix':
    strategy = WAALFixMatch(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'WAALUncertainty':
    strategy = WAALUncertainty(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'FF':
    strategy = FarthestFirst(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'FFF':
    strategy = FixMatchFarthestFirst(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'FixMatchEntropy':
    strategy = FixMatchEntropy(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'FixMatchRandom':
    strategy = FixMatchRandom(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'SelfTrainingEntropy':
    strategy = EntropySelfTraining(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'FFEntropy':
    strategy = FarthestFirstEntropy(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'DAL':
    strategy = DiscriminativeRepresentationSampling(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler,
                                                    test_handler, args)
elif QUERY_STRATEGY == 'LeastConfidence':
    strategy = LeastConfidence(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'FixMatchLeastConfidence':
    strategy = FixMatchLeastConfidence(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler,
                                       args)
elif QUERY_STRATEGY == 'Umap':
    strategy = UmapPlot(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'KLDiv':
    strategy = KLDiv(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'FixMatchKLDiv':
    strategy = FixMatchKLDiv(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'Dis':
    strategy = Discriminate(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'FixMatchDis':
    strategy = FixMatchDis(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'DisEntropyMixture':
    strategy = DisEntropyMixture(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'FixMatchDisEntropyMixture':
    strategy = FixMatchDisEntropyMixture(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler,
                                         args)
elif QUERY_STRATEGY == 'DisEntropyCombined':
    strategy = DisEntropyCombined(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'FixMatchDisEntropyCombined':
    strategy = FixMatchDisEntropyCombined(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler,
                                          args)
elif QUERY_STRATEGY == 'FixMatchSupEntropy':
    strategy = FixMatchSupEntropy(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'FixMatchSupLeastConfidence':
    strategy = FixMatchSupLeastConfidence(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler,
                                          args)
elif QUERY_STRATEGY == 'FixMatchMarginKMeans':
    strategy = FixMatchMarginKMeans(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'FixMatchEntropyKMeans':
    strategy = FixMatchEntropyKMeans(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'FixMatchMargin':
    strategy = FixMatchMargin(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
elif QUERY_STRATEGY == 'FixMatchRandomKMeans':
    strategy = FixMatchRandomKMeans(X_tr, Y_tr, idxs_lb, net_fea, net_clf, net_dis, train_handler, test_handler, args)
else:
    raise Exception('Unknown query strategy: {}'.format(QUERY_STRATEGY))

# print information
print(DATA_NAME)
# print('SEED {}'.format(SEED))
print(type(strategy).__name__)

# round 0 accuracy
torch.manual_seed(args['seed'])
strategy.train(alpha=alpha, total_epoch=epoch)
P = strategy.predict(X_te, Y_te)
acc = np.zeros(NUM_ROUND + 1)
acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
print('Round 0\ntesting accuracy {:.3f}'.format(acc[0]))

query_count = np.zeros(args['num_class'])
query_list = np.zeros((NUM_ROUND, NUM_QUERY), dtype=int)

for rd in range(1, NUM_ROUND + 1):
    print('================Round {:d}==============='.format(rd))

    # epoch += 5
    q_idxs = strategy.query(NUM_QUERY)
    idxs_lb[q_idxs] = True
    count = np.count_nonzero(idxs_lb)
    print("count: {}, NUM_INIT_LB: {}, NUM_QUERY: {}, rd: {}".format(count, NUM_INIT_LB, NUM_QUERY, rd))
    assert count == NUM_INIT_LB + NUM_QUERY * rd
    query_list[rd - 1] = q_idxs
    # update
    strategy.update(idxs_lb)
    # torch.manual_seed(args['seed'])
    strategy.train(alpha=alpha, total_epoch=epoch)

    # compute accuracy at each round
    P = strategy.predict(X_te, Y_te)
    acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    print('Accuracy {:.3f}'.format(acc[rd]))

for list in query_list:
    count = np.unique(Y_tr[list], return_counts=True)
    query_count[count[0]] += count[1]
# print final results for each round
# print('SEED {}'.format(SEED))
print(type(strategy).__name__)
if QUERY_STRATEGY == 'FFF' or QUERY_STRATEGY == 'FF':
    print('{}: {}'.format(args['farthest_first_criterion'], args['K']))
print('seed: {:d}'.format(args['seed']))
print(acc)
print(query_count)
np.save('data/query_list_' + QUERY_STRATEGY + '.npy', np.array(query_list))
