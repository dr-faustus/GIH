import torch
import torch.nn as nn
import numpy as np

import argparse
from typing import Union

from tqdm import tqdm
import copy

from src import compute_G, model_gen_fun, load_pruned_cifar_data, load_cifar10_data, train_fixed_iters, test

parser = argparse.ArgumentParser()

parser.add_argument('--n_experiments', type=int, default=5, help="Number of experiments performed.")
parser.add_argument('--n_iterations', type=int, default=50, help="Number of iterations to train the model.")
parser.add_argument('--n_model_samples_G', type=int, default=100000, help="Number of models used for computing G.")
parser.add_argument('--n_data_samples_G', type=int, default=128, help="Number of datapoints used for computing G per model.")
parser.add_argument('--sigma_x', type=float, default=0.001, help="Covariance of the dist used for computing G.")
parser.add_argument('--p', type=lambda x: type(x) is float and x < 1.0 and x > 0.0, default=0.5, help="Protion of the data to drop.")

parser.add_argument('--batch_size', type=int, default=128, help="Batch size used for training/evaluation.")
parser.add_argument('--model_name', choices=['ResNet18', 'ResNet9', 'MLP', 'LeNet', 'ViT'], default='ResNet18', help="The model used for the experiment.")
parser.add_argument('--max_lr', type=float, default=0.1, help="Learning rate used for the experiment.")

parser.add_argument('--data_size', type=int, default=10000, help='Train sample size.')

parser.add_argument('--data_path', type=str, default='./data', help="Path to data.")
parser.add_argument('--gpu', type=int, default=0, help="GPU num.")

args = parser.parse_args()

device = 'cuda:' + str(args.gpu)
INPUT_SHAPE = [3, 32, 32]
activation_function = nn.GELU

G = compute_G(n_model_samples=args.n_model_samples_G, n_data_samples=args.n_data_samples_G, input_shape=INPUT_SHAPE, sigma_x=args.sigma_x,
              model_gen_fun=lambda: model_gen_fun(model_name=args.model_name, activation_function=activation_function), device=device)
G = G.cpu()

trainloader, _, trainset, testset = load_cifar10_data(args.data_path, args.batch_size)
n_iters = len(trainloader)

score = []
for x, y in trainloader:
    x = x.reshape(-1, 32 * 32 * 3)
    x /= x.norm(p=2.0, dim=1, keepdim=True)
    score.append(((x @ G) * x).sum(dim=1))
score = torch.cat(score, dim=0)

test_acc_NADs = []
test_acc_rand = []
for n in range(args.n_experiments):
    init_model = model_gen_fun(model_name=args.model_name, activation_function=activation_function, num_channels=3, num_classes=10).to(device=device)

    trainloader, testloader = load_pruned_cifar_data(score, trainset=trainset, testset=testset, perc=args.p, batch_size=args.batch_size)
    model = train_fixed_iters(copy.deepcopy(init_model), trainloader, testloader, epochs=args.n_iterations, 
                              max_lr=args.max_lr, momentum=0.9, weight_decay=5e-4, n_iters=n_iters, device=device)
    test_acc_NADs.append(test(model, testloader, device=device)[0])

    trainloader, testloader = load_pruned_cifar_data(torch.randn_like(score), trainset=trainset, testset=testset, perc=args.p, batch_size=args.batch_size)
    model = train_fixed_iters(copy.deepcopy(init_model), trainloader, testloader, epochs=args.n_iterations, 
                              max_lr=args.max_lr, momentum=0.9, weight_decay=5e-4, n_iters=n_iters, device=device)
    test_acc_rand.append(test(model, testloader, device=device)[0])

print()
print('Experiment settings: ')
print(args)
print('-' * 10)
print('NADs test accuracy: ')
print(test_acc_NADs)
print('-' * 10)
print('Random test accuracy: ')
print(test_acc_rand)