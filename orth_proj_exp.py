import torch
import torch.nn as nn
import numpy as np

import argparse
from typing import Union

from tqdm import tqdm
import copy

from src import get_cifar10_cov, compute_G, compute_G_eig, model_gen_fun, load_cifar2_data, train, test

parser = argparse.ArgumentParser()

parser.add_argument('--n_experiments', type=int, default=5, help="Number of experiments performed.")
parser.add_argument('--n_iterations', type=Union[int, None], default=None, help="Number of iterations to train the model - default is the values used in the paper.")
parser.add_argument('--n_model_samples_G', type=int, default=100000, help="Number of models used for computing G.")
parser.add_argument('--n_data_samples_G', type=int, default=128, help="Number of datapoints used for computing G per model.")
parser.add_argument('--sigma_x', type=float, default=0.001, help="Covariance of the dist used for computing G.")
parser.add_argument('--step_size', type=int, default=30, help="Step size for range of indices used for the experiment.")

parser.add_argument('--batch_size', type=int, default=128, help="Batch size used for training/evaluation.")
parser.add_argument('--model_name', choices=['ResNet18', 'ResNet9', 'MLP', 'LeNet', 'ViT'], default='ResNet18', help="The model used for the experiment.")
parser.add_argument('--max_lr', type=Union[float, None], default=None, help="Learning rate used for the experiment - default is the values used in the paper.")

parser.add_argument('--data_size', type=int, default=10000, help='Train sample size.')

parser.add_argument('--data_path', type=str, default='./data', help="Path to data.")
parser.add_argument('--gpu', type=int, default=0, help="GPU num.")

args = parser.parse_args()

device = 'cuda:' + str(args.gpu)
INPUT_SHAPE = [3, 32, 32]
activation_function = nn.GELU

if args.max_lr is None:
    if args.model_name == 'MLP': 
        max_lr = 0.01
    else: 
        max_lr = 0.1
else:
    max_lr = args.max_lr

if args.n_iterations is None:
    if args.model_name == 'LeNet': 
        n_iterations = 50
    else: 
        n_iterations = 20
else:
    n_iterations = args.n_iterations

def orthogonalize(V):
    for idx in range(1, V.shape[0]):
        for v in V[:idx]:
            V[idx] -= np.inner(V[idx], v) * v
        V[idx] /= np.linalg.norm(V[idx], ord=2.0)
    return V

def get_ev(V, S):
    corr = [np.inner(v, S @ v) for v in V]
    ev = [sum(corr[:idx + 1]) / sum(corr) for idx in range(V.shape[0])]
    return ev

def get_rand_basis_vecs(n_dims):
    V = []
    for n in range(n_dims):
        dir = np.random.randn(n_dims)
        dir /= np.linalg.norm(dir)
        V.append(dir)
    return np.stack(V, axis=0)

def get_orth_mat(V, n_components):
    P = np.eye(np.prod(INPUT_SHAPE))
    for idx in range(n_components):
        P -= np.outer(V[idx], V[idx])
    P = torch.from_numpy(P).float().to(device)
    return P

S = get_cifar10_cov(data_path=args.data_path, batch_size=args.batch_size, device=device)
G = compute_G(n_model_samples=args.n_model_samples_G, n_data_samples=args.n_data_samples_G, input_shape=INPUT_SHAPE, sigma_x=args.sigma_x,
              model_gen_fun=lambda: model_gen_fun(model_name=args.model_name, activation_function=activation_function), device=device)

eigenvalues, NADs = compute_G_eig(G @ S @ G)

S, G, eigenvalues, NADs = S.cpu().numpy(), G.cpu().numpy(), eigenvalues.cpu().numpy(), NADs.cpu().numpy()

ev_gsg = get_ev(NADs, S)

test_acc_GSG = {}
test_acc_S = {}

for n_components in tqdm(list(args.step_size * np.arange(np.floor(np.prod(INPUT_SHAPE) / args.step_size) + 1).astype(int))):
    test_acc_GSG[n_components] = []
    test_acc_S[n_components] = []
    for n in range(args.n_experiments):
        V_s = orthogonalize(get_rand_basis_vecs(n_dims=np.prod(INPUT_SHAPE)))
        ev_s = get_ev(V_s, S)
        
        trainset, testset = load_cifar2_data(data_path=args.data_path, non_linearity_type='none', data_size=args.data_size)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=2, pin_memory=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=2, pin_memory=True)

        P = get_orth_mat(NADs, n_components)

        init_model = model_gen_fun(model_name=args.model_name, activation_function=activation_function, num_channels=3, num_classes=2).to(device)

        model = train(model=copy.deepcopy(init_model).to(device), trainloader=trainloader, testloader=testloader,
                      epochs=n_iterations, max_lr=max_lr, momentum=0.9, weight_decay=5e-4, proj_mat=P,
                      device=device, verbose=False)
        
        test_acc_GSG[n_components].append(test(model, testloader, proj_mat=P, device=device)[0])
        
        if ev_gsg[n_components] < ev_s[0]:
            P = None
        else:
            idx = 0
            while ev_s[idx] < ev_gsg[n_components]:
                idx += 1
                if idx == len(ev_s): break
            P = get_orth_mat(V_s, idx)

        model = train(model=copy.deepcopy(init_model).to(device), trainloader=trainloader, testloader=testloader,
                      epochs=n_iterations, max_lr=max_lr, momentum=0.9, weight_decay=5e-4, proj_mat=P,
                      device=device, verbose=False)
        test_acc_S[n_components].append(test(model, testloader, proj_mat=P, device=device)[0])

print()
print('Experiment settings: ')
print(args)
print('-' * 10)
print('Test accuracy GSG: ')
print(test_acc_GSG)
print('-' * 10)
print('Test accuracy random: ')
print(test_acc_S)