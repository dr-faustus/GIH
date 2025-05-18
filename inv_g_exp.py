import torch
import torch.nn as nn
import argparse

import numpy as np
from typing import Union
from tqdm import tqdm

from src import compute_G, compute_G_t, compute_G_eig, model_gen_fun, test, InverseCovarDataset

parser = argparse.ArgumentParser()

parser.add_argument('--n_experiments', type=int, default=25, help="Number of experiments performed.")
parser.add_argument('--n_iterations', type=Union[int, None], default=None, help="Number of iterations to train the model - default is the value used in the paper.")
parser.add_argument('--n_model_samples_G', type=int, default=100000, help="Number of models used for computing G.")
parser.add_argument('--n_data_samples_G', type=int, default=128, help="Number of datapoints used for computing G per model.")
parser.add_argument('--sigma_x', type=float, default=0.001, help="Covariance of the dist used for computing G.")

parser.add_argument('--batch_size', type=int, default=128, help="Batch size used for training/evaluation.")
parser.add_argument('--model_name', required=True, choices=['ResNet18', 'ResNet9', 'MLP', 'LeNet', 'ViT'], help="The model used for the experiment.")
parser.add_argument('--max_lr', type=float, default=0.1, help="Learning rate used for the experiment - default is the value used in the paper.")

parser.add_argument('--data_size', type=int, default=10000, help='Train sample size.')

parser.add_argument('--data_path', type=str, default='./data', help="Path to data.")
parser.add_argument('--gpu', type=int, default=0, help="GPU num.")

args = parser.parse_args()

device = 'cuda:' + str(args.gpu)
INPUT_SHAPE = [1, 32, 32]
activation_function = nn.ReLU

if args.model_name == 'ResNet18' or args.model_name == 'ResNet9':
    model_name = args.model_name + '_wo_bn'
elif args.model_name == 'ViT':
    model_name = args.model_name + '_wo_ln'
else:
    model_name = args.model_name

if args.n_iterations is None:
    if args.model_name == 'LeNet':
        n_iterations = 100
    else:
        n_iterations = 50
else:
    n_iterations = args.n_iterations


def get_velocity(G_list):
    vel = []
    for idx in range(len(G_list) - 1):
        vel.append(1 - (torch.trace(G_list[idx + 1] @ G_list[idx]) / (torch.trace(G_list[idx] @ G_list[idx]) * torch.trace(G_list[idx + 1] @ G_list[idx + 1])).sqrt()).item())
    return vel

G = compute_G(n_model_samples=args.n_model_samples_G, n_data_samples=args.n_data_samples_G, input_shape=INPUT_SHAPE, sigma_x=args.sigma_x,
              model_gen_fun=lambda: model_gen_fun(model_name=model_name, activation_function=activation_function, num_channels=1, num_classes=2), device=device)
eigenvalues, NADs = compute_G_eig(G)
G, eigenvalues, NADs = G.cpu().numpy(), eigenvalues.cpu().numpy(), NADs.cpu().numpy()

G_flipped = NADs.T @ np.diag(np.flip(eigenvalues)) @ NADs

for flip in [True, False]:
    model_cov = {idx: 0 for idx in range(n_iterations)}
    train_acc = {idx: 0 for idx in range(n_iterations)}
    param_dist = {idx: 0 for idx in range(n_iterations)}
    for n in tqdm(range(args.n_experiments)):
        if flip is True:
            trainset = InverseCovarDataset(G_flipped / np.linalg.norm(G_flipped, ord=2), num_samples=args.data_size, shape=INPUT_SHAPE)
            testset = InverseCovarDataset(G_flipped / np.linalg.norm(G_flipped, ord=2), num_samples=args.data_size // 5, shape=INPUT_SHAPE)
        else:
            trainset = InverseCovarDataset(G / np.linalg.norm(G, ord=2), num_samples=args.data_size, shape=INPUT_SHAPE)
            testset = InverseCovarDataset(G / np.linalg.norm(G, ord=2), num_samples=args.data_size // 5, shape=INPUT_SHAPE)
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=2, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=0, pin_memory=True)
        lr_schedule = lambda t: np.interp([t], [0, n_iterations], [args.max_lr, 0])[0]

        model = model_gen_fun(model_name, activation_function=activation_function, num_channels=1, num_classes=2)
        model = model.to(device)

        init_params = torch.cat([p.reshape(-1).clone().detach() for p in model.parameters()], dim=-1)
        loss_fun = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(model.parameters(), lr=args.max_lr, momentum=0.9, weight_decay=0.0)

        for iter_idx in range(n_iterations):
            model.train()

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device=device), targets.to(device=device)
                
                lr = lr_schedule(iter_idx + (batch_idx + 1) / len(trainloader))
                opt.param_groups[0].update(lr=lr)
                
                output = model(inputs)
                loss = loss_fun(output, targets)

                opt.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
            
            current_params = torch.cat([p.reshape(-1).clone().detach() for p in model.parameters()], dim=-1)
            model_cov[iter_idx] += compute_G_t(model, data_shape=INPUT_SHAPE) / args.n_experiments
            train_acc[iter_idx] += test(model, trainloader, device=device)[0] / args.n_experiments
            param_dist[iter_idx] += ((init_params - current_params).norm() / init_params.norm()).item() / args.n_experiments

    velocity = {idx: x for idx, x in enumerate(get_velocity(list(model_cov.values())))}

    print()
    print('Experiment settings: ')
    print(args)
    print('-' * 10)
    print('Flip:', flip)
    print('-' * 10)
    print('Velocity: ')
    print(velocity)
    print('-' * 10)
    print('Train accuracy: ')
    print(train_acc)
    print('-' * 10)
    print('Init parameter distance: ')
    print(param_dist)