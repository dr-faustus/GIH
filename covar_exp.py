import torch
import torch.nn as nn
import numpy as np

import argparse
from typing import Union

from tqdm import tqdm

from src import get_cifar10_cov, compute_G, compute_G_t, compute_G_eig, model_gen_fun, load_cifar2_data, test

parser = argparse.ArgumentParser()

parser.add_argument('--n_experiments', type=int, default=25, help="Number of experiments performed.")
parser.add_argument('--n_iterations', type=int, default=100, help="Number of iterations to train the model.")
parser.add_argument('--n_model_samples_G', type=int, default=100000, help="Number of models used for computing G.")
parser.add_argument('--n_data_samples_G', type=int, default=128, help="Number of datapoints used for computing G per model.")
parser.add_argument('--sigma_x', type=float, default=1.0, help="Covariance of the dist used for computing G.")

parser.add_argument('--batch_size', type=int, default=128, help="Batch size used for training/evaluation.")
parser.add_argument('--random_labels', action='store_true', help="Perform the experiment with random labels.")
parser.add_argument('--model_name', choices=['ResNet18', 'ResNet9', 'MLP', 'LeNet', 'ViT'], default='ResNet18', help="The model used for the experiment.")
parser.add_argument('--max_lr', type=Union[float, None], default=None, help="Learning rate used for the experiment - default is the values used in the paper.")

parser.add_argument('--data_size', type=int, default=10000, help='Train sample size.')

parser.add_argument('--data_path', type=str, default='./data', help="Path to data.")
parser.add_argument('--gpu', type=int, default=0, help="GPU num.")

args = parser.parse_args()

device = 'cuda:' + str(args.gpu)
INPUT_SHAPE = [3, 32, 32]
activation_function = nn.ReLU

if args.max_lr is None:
    if args.random_labels is True:
        if args.model_name == 'ResNet18' or args.model_name == 'ViT': 
            max_lr = 0.0001
        else: max_lr = 0.001
    else:
        if args.model_name == 'MLP': 
            max_lr = 0.01
        else:
            max_lr = 0.1
else:
    max_lr = args.max_lr

if args.random_labels is True:
    if args.model_name == 'LeNet' or args.model_name == 'ViT': 
        multiplier = 5
    else: 
        multiplier = 2
else:
    multiplier = 1

if args.model_name == 'ResNet18' or args.model_name == 'ResNet9':
    model_name = args.model_name + '_wo_bn'
elif args.model_name == 'ViT':
    model_name = args.model_name + '_wo_ln'
else:
    model_name = args.model_name

S = get_cifar10_cov(data_path=args.data_path, batch_size=args.batch_size, device=device)
G = compute_G(n_model_samples=args.n_model_samples_G, n_data_samples=args.n_data_samples_G, input_shape=INPUT_SHAPE, sigma_x=args.sigma_x,
              model_gen_fun=lambda: model_gen_fun(model_name=model_name, activation_function=activation_function), device=device)
eigenvalues, NADs = compute_G_eig(G)

GSG = G @ S @ G

G_t = {idx: 0 for idx in range(args.n_iterations)}
test_acc = {idx: 0 for idx in range(args.n_iterations)}
train_acc = {idx: 0 for idx in range(args.n_iterations)}

for n in tqdm(range(args.n_experiments)):
    trainset, testset = load_cifar2_data(data_path=args.data_path, direction=None, 
                                         non_linearity_type='none', data_size=args.data_size)

    if args.random_labels is True: 
        trainset.targets = list(np.random.permutation(trainset.targets))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=2, pin_memory=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=0, pin_memory=True)

    model = model_gen_fun(model_name=model_name, activation_function=activation_function)
    model = model.to(device)

    n_epochs = args.n_iterations * multiplier if args.random_labels else args.n_iterations

    if args.random_labels:
        opt = torch.optim.Adam(model.parameters(), lr=max_lr)
    else:
        lr_schedule = lambda t: np.interp([t], [0, n_epochs], [max_lr, 0])[0]
        opt = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=5e-4)

    loss_fun = nn.CrossEntropyLoss(reduction='sum')

    for iter_idx in range(n_epochs):
        model.train()
        opt.zero_grad()

        if not args.random_labels:
            lr = lr_schedule(iter_idx)
            opt.param_groups[0].update(lr=lr)

        mean_loss = 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            output = model(inputs)
            loss = loss_fun(output, targets)

            loss.backward()

            mean_loss += loss.item() / trainset.data.shape[0]
        
        for param in model.parameters(): param.grad.data.div_(trainset.data.shape[0])

        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if args.random_labels:
            if iter_idx % multiplier == 0:
                G_t[int(iter_idx / multiplier)] += compute_G_t(model) / args.n_experiments
                test_acc[int(iter_idx / multiplier)] += test(model, testloader, device=device)[0] / args.n_experiments
                train_acc[int(iter_idx / multiplier)] += test(model, trainloader, device=device)[0] / args.n_experiments
        else:
            G_t[iter_idx] += compute_G_t(model) / args.n_experiments
            test_acc[iter_idx] += test(model, testloader, device=device)[0] / args.n_experiments
            train_acc[iter_idx] += test(model, trainloader, device=device)[0] / args.n_experiments

corr_S = {}
corr_GSG = {}
for iter_idx in range(args.n_iterations):
    corr_S[iter_idx] = (torch.trace(G_t[iter_idx] @ S) / (torch.trace(G_t[iter_idx] @ G_t[iter_idx]).sqrt() * torch.trace(S @ S).sqrt())).item()
    corr_GSG[iter_idx] = (torch.trace(G_t[iter_idx] @ GSG) / (torch.trace(G_t[iter_idx] @ G_t[iter_idx]).sqrt() * torch.trace(GSG @ GSG).sqrt())).item()

print()
print('Experiment settings: ')
print(args)
print('-' * 10)
print('Corr S: ')
print(corr_S)
print('-' * 10)
print('Corr GSG: ')
print(corr_GSG)
print('-' * 10)
print('Test accuracy: ')
print(test_acc)
print('-' * 10)
print('Train accuracy: ')
print(train_acc)
