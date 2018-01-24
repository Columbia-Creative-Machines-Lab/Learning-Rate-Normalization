from __future__ import print_function
import argparse
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from sgd_lr_norm import *
from adam_lr_norm import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#######################################################################################
# Configuration
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--optimizer', type=str, default='Adam_lr_norm', metavar='M',
                    help='Adam_lr_norm|Adam')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD_lr_norm momentum (default: 0.5)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', type=str, default='ConvolutionalNet',
                        help='ConvolutionalNet|MLPNet')
parser.add_argument('--dataset', type=str, default='cifar10',
                        help='mnist|cifar10|cifar100|lsun|svhn')
parser.add_argument('--data_path', type=str, default='./data',
                        help='where to save data (if any).')
parser.add_argument('--image_size', type=int, default=28,
                        help='preprocesses data into the specified image size')
parser.add_argument('--save_checkpoint', type=str, default='./checkpoint',
                        help='where to save checkpoints (if any).')
parser.add_argument('--load_checkpoint', type=str,
                        help='where to load checkpoint (if any).')
parser.add_argument('--lr_scheduler', type=str, default='linear',
                        help='type of learning rate scheduler (exponential or linear or none)')
parser.add_argument('--gamma', type=float, default=0.5,
                        help='learning rate scheduler decay parameter')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#####################################################################################
# Load dataset
ALLOWABLE_DATASETS = ['mnist', 'cifar10', 'cifar100', 'lsun', 'svhn']
ALLOWABLE_MODELS = ['ConvolutionalNet', 'MLPNet']

assert args.dataset in ALLOWABLE_DATASETS
assert args.model in ALLOWABLE_MODELS

preprocessing = [
    transforms.Scale(args.image_size),
    transforms.CenterCrop(args.image_size),
    transforms.ToTensor(),
]

if args.dataset == 'mnist':
    preprocessing.append(transforms.Normalize((0.1307,), (0.3081,))) 
    train_loader = DataLoader(
        datasets.MNIST(args.data_path, train=True, download=True,
                       transform=transforms.Compose(preprocessing)),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(
        datasets.MNIST(args.data_path, train=False,
                       transform=transforms.Compose(preprocessing)),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    input_channels = 1
    n_class = 10
elif args.dataset == 'cifar10':
    preprocessing.append(transforms.Normalize((0.4914, 0.4822, 0.4465),
                                              (0.2023, 0.1994, 0.2010)))
    train_loader = DataLoader(
        datasets.CIFAR10(args.data_path, train=True, download=True,
                         transform=transforms.Compose(preprocessing)),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(
        datasets.CIFAR10(args.data_path, train=False,
                         transform=transforms.Compose(preprocessing)),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    input_channels = 3
    n_class = 10
elif args.dataset == 'cifar100':
    preprocessing.append(transforms.Normalize((0.4914, 0.4822, 0.4465),
                                              (0.2023, 0.1994, 0.2010)))
    train_loader = DataLoader(
        datasets.CIFAR100(args.data_path, train=True, download=True,
                          transform=transforms.Compose(preprocessing)),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(
        datasets.CIFAR100(args.data_path, train=False,
                          transform=transforms.Compose(preprocessing)),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    input_channels = 3
    n_class = 100
elif args.dataset == 'lsun': # has to be downloaded already
    preprocessing.append(transforms.Normalize((0.5, 0.5, 0.5),
                                              (0.5, 0.5, 0.5)))
    train_loader = DataLoader(
        datasets.LSUN(args.data_path, 'train',
                      [transforms.Compose(preprocessing), None]),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(
        datasets.LSUN(args.data_path, 'test',
                      [transforms.Compose(preprocessing), None]),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    input_channels = 3
    n_class = 10
elif args.dataset == 'svhn':
    preprocessing.append(transforms.Normalize((0.4377, 0.4438, 0.4728),
                                              (0.1980, 0.2010, 0.1970)))
    train_loader = DataLoader(
        datasets.SVHN(args.data_path, split='train', download=True,
                      transform=transforms.Compose(preprocessing)),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(
        datasets.SVHN(args.data_path, split='test',
                      transform=transforms.Compose(preprocessing)),
        batch_size=args.batch_size, shuffle=True, pin_memory=True)
    input_channels = 3
    n_class = 10

assert train_loader
assert test_loader
assert input_channels
assert n_class

#################################################################################
# Models
class ConvolutionalNet(nn.Module):
    def __init__(self, width, height, depth, n_class):
        super(ConvolutionalNet, self).__init__()
        self.conv1 = nn.Conv2d(depth, 20, 5)
        self.bc1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 20, 3)
        self.bc2 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 20, 3)
        self.bc3 = nn.BatchNorm2d(20)
        output_width = width - 8
        output_features = 20 * output_width * output_width
        self.fc1 = nn.Linear(output_features, 50)
        self.bc4 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 50)
        self.bc5 = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50, 50)
        self.bc6 = nn.BatchNorm1d(50)
        self.fc4 = nn.Linear(50, n_class)

    def forward(self, x):
        x = self.bc1(F.relu(self.conv1(x)))
        x = self.bc2(F.relu(self.conv2(x)))
        x = self.bc3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.bc4(F.relu(self.fc1(x)))
        x = self.bc5(F.relu(self.fc2(x)))
        x = self.bc6(F.relu(self.fc3(x)))
        x = F.relu(self.fc4(x))
        return F.log_softmax(x)

class MLPNet(nn.Module):
    def __init__(self, width, height, depth, n_class):
        super(MLPNet, self).__init__()
        self.n_input = width*height*depth
        self.fc1 = nn.Linear(self.n_input, 500)
        self.bc1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 250)
        self.bc2 = nn.BatchNorm1d(250)
        self.fc3 = nn.Linear(250, n_class)

    def forward(self, x):
        x = x.view((-1, self.n_input))
        x = F.relu(self.bc1(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.bc2(self.fc2(x)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x)

if args.model == "ConvolutionalNet":
    model = ConvolutionalNet(args.image_size, args.image_size, input_channels, n_class)
elif args.model == "MLPNet":
    model = MLPNet(args.image_size, args.image_size, input_channels, n_class)
assert model

if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)
#if args.optimizer == "SGD_lr_norm":
if args.optimizer == "Adam_lr_norm":
    #optimizer = SGD_lr_norm(model.parameters(), lr=args.lr, momentum=args.momentum, schedule=args.lr_scheduler, gamma=args.gamma)
    optimizer = Adam_lr_norm(model.parameters(), lr=args.lr, schedule=args.lr_scheduler, gamma=args.gamma)
elif args.optimizer == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
assert optimizer

####################################################################################
# Training and Testing
def train(epoch):
    model.train()
    norm_log = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        # _, norm_log = optimizer.step()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    checkpoint_path = os.path.join(args.save_checkpoint, "%s_epoch_%d.pth" % (args.model, epoch))
    torch.save(model.state_dict(), checkpoint_path)
    return norm_log

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Creates checkpoint directory if it doesn't exist
if not os.path.exists(args.save_checkpoint):
    os.makedirs(args.save_checkpoint)

start_epoch = 1

# Loads checkpoint if specified
if args.load_checkpoint:
    start_epoch += int(args.load_checkpoint.split('_')[-1][:-4])
    model.load_state_dict(torch.load(args.load_checkpoint))

norm_logs = []
epochs = []
for epoch in range(start_epoch, args.epochs + 1):
    epochs.append(epoch)
    norm_log = train(epoch)
    #plt.figure()
    #plt.plot(layer, norm_log[0])
    #plt.show()
    norm_logs.append(norm_log)
    test()
# layers = [i for i in range(len(norm_log[0]))]
# norms = []
# [norms.append(norm[0]) for norm in norm_logs]
# norms = np.array(norms)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(layers, epochs, norms, color='b')
# plt.show()
