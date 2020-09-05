import numpy as np
from dataset import get_dataset, get_handler
from model import get_net
from torchvision import transforms
from query_strategies import EntropySampling
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dataname", type=str, help="dataname")
parser.add_argument("str_indx", type=int, help="baseline index")

args = parser.parse_args()

DATA_NAME = args.dataname
str_indx = args.str_indx

# DATA_NAME   = 'FashionMNIST'
# DATA_NAME   = 'SVHN'
# DATA_NAME   = 'CIFAR10'


if DATA_NAME == "FashionMNIST":
    NUM_INIT_LB = 500
    NUM_QUERY = 500
    NUM_ROUND = 5
    epoch = 10
    incre = 5

elif DATA_NAME == "SVHN":
    NUM_INIT_LB = 1000
    NUM_QUERY = 1000
    NUM_ROUND = 5
    epoch = 20
    incre = 5

elif DATA_NAME == "CIFAR10":
    NUM_INIT_LB = 2000
    NUM_QUERY = 2000
    NUM_ROUND = 5
    epoch = 50
    incre = 5

args_pool = {
    'FashionMNIST':
        {'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
         'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
         'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
         'optimizer_args': {'lr': 0.01, 'momentum': 0.5},
         'num_class': 10},
    'SVHN':
        {'transform_tr': transforms.Compose([
            # transforms.RandomCrop(size=32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
            'transform_te': transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4377, 0.4438, 0.4728),
                                                                     (0.1980, 0.2010, 0.1970))]),
            'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
            'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
            'optimizer_args': {'lr': 0.01, 'momentum': 0.3},
            'num_class': 10},

    'CIFAR10':
        {'transform_tr': transforms.Compose([
            # transforms.RandomCrop(size=32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
            'transform_te': transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                     (0.2470, 0.2435, 0.2616))]),
            'loader_tr_args': {'batch_size': 64, 'num_workers': 1},
            'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
            'optimizer_args': {'lr': 0.01, 'momentum': 0.3},
            'num_class': 10},
}

args = args_pool[DATA_NAME]

# Load dataset (Only using the first 40K, svhn and cifar 10)
# cifar 100 using 50K
X_tr, Y_tr, X_te, Y_te = get_dataset(DATA_NAME)
X_tr = X_tr[:50000, :]
Y_tr = Y_tr[:50000]

# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)
print('number of labeled pool: {}'.format(NUM_INIT_LB))
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB))
print('number of testing pool: {}'.format(n_test))

# Generate the initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True

# loading neural network
net = get_net(DATA_NAME)

# here the training handlers and testing handlers are different
train_handler = get_handler(DATA_NAME)

strategy = EntropySampling(X_tr, Y_tr, idxs_lb, net, train_handler, args)

# print information
print(DATA_NAME)
# print('SEED {}'.format(SEED))
print(type(strategy).__name__)

# round 0 accuracy
strategy.train(total_epoch=epoch)
P = strategy.predict(X_te, Y_te)
acc = np.zeros(NUM_ROUND + 1)
acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
print('Round 0\ntesting accuracy {:.3f}'.format(acc[0]))

for rd in range(1, NUM_ROUND + 1):
    print('================Round {:d}==============='.format(rd))

    epoch += incre

    # query step (need to estimate time)
    q_idxs = strategy.query(NUM_QUERY)
    idxs_lb[q_idxs] = True

    # update
    strategy.update(idxs_lb)
    strategy.train(total_epoch=epoch)

    # compute accuracy at each round
    P = strategy.predict(X_te, Y_te)
    acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    print('testing accuracy {:.3f}'.format(acc[rd]))

# print final results for each round
# print('SEED {}'.format(SEED))
print(type(strategy).__name__)
print(acc)
