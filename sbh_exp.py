import torch
import torch.nn as nn
import numpy as np

import argparse
import copy
from typing import Union
from tqdm import tqdm

from src import compute_G, compute_G_eig, model_gen_fun, train, test, LinearCircleDataset

parser = argparse.ArgumentParser()

parser.add_argument('--n_experiments', type=int, default=5, help="Number of experiments performed.")
parser.add_argument('--n_iterations', type=Union[int, None], default=None, help="Number of iterations to train the model - default is the values used in the paper.")
parser.add_argument('--n_model_samples_G', type=int, default=100000, help="Number of models used for computing G.")
parser.add_argument('--n_data_samples_G', type=int, default=128, help="Number of datapoints used for computing G per model.")
parser.add_argument('--sigma_x', type=float, default=0.0001, help="Covariance of the dist used for computing G.")
parser.add_argument('--noise_std', type=float, default=1.0, help="Noise standard deviation.")
parser.add_argument('--linear_std', type=float, default=1.0, help="Linear component std.")
parser.add_argument('--epsilon', type=float, default=1.0, help="Magnitude of the lindear discriminative feature.")
parser.add_argument('--step_size', type=int, default=10, help="Step size for range of indices used for the experiment.")

parser.add_argument('--batch_size', type=int, default=128, help="Batch size used for training/evaluation.")
parser.add_argument('--random_labels', action='store_true', help="Perform the experiment with random labels.")
parser.add_argument('--model_name', choices=['ResNet18', 'ResNet9', 'MLP', 'LeNet', 'ViT'], default='ResNet18', help="The model used for the experiment.")
parser.add_argument('--max_lr', type=float, default=0.1, help="Learning rate used for the experiment - default is the values used in the paper.")

parser.add_argument('--data_size', type=int, default=10000, help='Train sample size.')

parser.add_argument('--data_path', type=str, default='./data', help="Path to data.")
parser.add_argument('--gpu', type=int, default=0, help="GPU num.")

args = parser.parse_args()

device = 'cuda:' + str(args.gpu)
INPUT_SHAPE = [1, 32, 32]
activation_function = nn.GELU

if args.model_name == 'LeNet':
    epsilon_1, epsilon_2 = 1.5, 0.25
    u_1_idx, u_2_idx = 15, 16
    n_iterations = 50
elif args.model_name == 'ResNet18':
    epsilon_1, epsilon_2 = 3.0, 0.25
    u_1_idx, u_2_idx = 5, 6
    n_iterations = 20
elif args.model_name == 'ViT' or args.model_name == 'MLP':
    epsilon_1, epsilon_2 = 1.5, 0.25
    u_1_idx, u_2_idx = 15, 16
    n_iterations = 20

G = compute_G(n_model_samples=args.n_model_samples_G, n_data_samples=args.n_data_samples_G, input_shape=INPUT_SHAPE, sigma_x=args.sigma_x,
              model_gen_fun=lambda: model_gen_fun(model_name=args.model_name, activation_function=activation_function, num_channels=1, num_classes=2), device=device)
eigenvalues, NADs = compute_G_eig(G)
G, eigenvalues, NADs = G.cpu().numpy(), eigenvalues.cpu().numpy(), NADs.cpu().numpy()

test_acc = {}
test_acc_linear = {}
test_acc_nonlinear = {}

def orthogonal_proj(x, dirs):
    x = x.reshape(-1, np.prod(INPUT_SHAPE))
    P = np.eye(np.prod(INPUT_SHAPE)) - sum([np.outer(dir.reshape(-1), dir.reshape(-1)) for dir in dirs])
    return (x @ P).reshape([x.shape[0]] + INPUT_SHAPE).float()

for linear_idx in tqdm(list(args.step_size * np.arange(np.floor(np.prod(INPUT_SHAPE) / args.step_size) + 1).astype(int))):
    test_acc[linear_idx] = []
    test_acc_linear[linear_idx] = []
    test_acc_nonlinear[linear_idx] = []
    for n in range(args.n_experiments):
        u_1, u_2, u_3 = NADs[u_1_idx].reshape(INPUT_SHAPE), NADs[u_2_idx].reshape(INPUT_SHAPE), NADs[linear_idx].reshape(INPUT_SHAPE)

        trainset = LinearCircleDataset(u_1, u_2, u_3, num_samples=args.data_size, 
                                       sigma=args.noise_std, epsilon_1=epsilon_1, epsilon_2=epsilon_2, epsilon_3=args.epsilon)
        testset = LinearCircleDataset(u_1, u_2, u_3, num_samples=args.data_size // 5, 
                                      sigma=args.noise_std, epsilon_1=epsilon_1, epsilon_2=epsilon_2, epsilon_3=args.epsilon)
        
        testset_linear = copy.deepcopy(testset)
        testset_linear.data = orthogonal_proj(testset_linear.data, [u_1, u_2])

        testset_nonlinear = copy.deepcopy(testset)
        testset_nonlinear.data = orthogonal_proj(testset_nonlinear.data, [u_3])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=2, pin_memory=True)

        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=0, pin_memory=True)
        testloader_linear = torch.utils.data.DataLoader(testset_linear, batch_size=args.batch_size, shuffle=False,
                                                        num_workers=0, pin_memory=True)
        testloader_nonlinear = torch.utils.data.DataLoader(testset_nonlinear, batch_size=args.batch_size, shuffle=False,
                                                           num_workers=0, pin_memory=True)
        
        model = model_gen_fun(model_name=args.model_name, activation_function=activation_function, num_channels=1, num_classes=2)
        model = model.to(device)

        model = train(model, trainloader, testloader, epochs=n_iterations, max_lr=args.max_lr, momentum=0.9, weight_decay=0.0, verbose=False)

        test_acc[linear_idx].append(test(model, testloader, device)[0])
        test_acc_linear[linear_idx].append(test(model, testloader_linear, device)[0])
        test_acc_nonlinear[linear_idx].append(test(model, testloader_nonlinear, device)[0])

print()
print('Experiment settings: ')
print(args)
print('-' * 10)
print('Test accuracy: ')
print(test_acc)
print('-' * 10)
print('Test accuracy - linear: ')
print(test_acc_linear)
print('-' * 10)
print('Test accuracy - non-linear: ')
print(test_acc_nonlinear)