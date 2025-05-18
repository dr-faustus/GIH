import torch
import torch.nn as nn
import numpy as np

import argparse

from typing import Union
from tqdm import tqdm

from src import compute_G, compute_G_eig, model_gen_fun, train, test, StdNormalDataset

parser = argparse.ArgumentParser()

parser.add_argument('--n_experiments', type=int, default=5, help="Number of experiments performed.")
parser.add_argument('--n_iterations', type=Union[int, None], default=None, help="Number of iterations to train the model - default is the values used in the paper.")
parser.add_argument('--n_model_samples_G', type=int, default=100000, help="Number of models used for computing G.")
parser.add_argument('--n_data_samples_G', type=int, default=128, help="Number of datapoints used for computing G per model.")
parser.add_argument('--sigma_x', type=float, default=0.001, help="Covariance of the dist used for computing G.")
parser.add_argument('--non_linearity_type', type=str, choices=['linear', 'quadratic', 'sinusoidal'], help="Type of non-linearity used for labeling.")
parser.add_argument('--noise_std', type=float, default=1.0, help="Noise standard deviation.")
parser.add_argument('--epsilon', type=Union[float, None], default=None, help="Magnitude of discriminative feature - default is the values used in the paper.")
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

if args.epsilon is None:
    if args.non_linearity_type == 'linear':
        epsilon = 0.9
    elif args.non_linearity_type == 'quadratic':
        epsilon = 2.0
    elif args.non_linearity_type == 'sinusoidal':
        if args.model_name == 'ViT' or args.model_name == 'LeNet' or args.model_name == 'SmallLeNet':
            epsilon = 2.2
        elif args.model_name == 'MLP' or args.model_name == 'ResNet18':
            epsilon = 5.0
else:
    epsilon = args.epsilon

if args.n_iterations is None:
    if args.model_name == 'LeNet':
        n_iterations = 50
    else:
        n_iterations = 30

G = compute_G(n_model_samples=args.n_model_samples_G, n_data_samples=args.n_data_samples_G, input_shape=INPUT_SHAPE, sigma_x=args.sigma_x,
              model_gen_fun=lambda: model_gen_fun(model_name=args.model_name, activation_function=activation_function, num_channels=1, num_classes=2), device=device)
eigenvalues, NADs = compute_G_eig(G)
G, eigenvalues, NADs = G.cpu().numpy(), eigenvalues.cpu().numpy(), NADs.cpu().numpy()

test_acc_result = {}
train_acc_result = {}

for idx in tqdm(list(args.step_size * np.arange(np.floor(np.prod(INPUT_SHAPE) / args.step_size) + 1).astype(int))):
    test_acc_result[idx] = []
    train_acc_result[idx] = []

    for n in range(args.n_experiments):
        model = model_gen_fun(model_name=args.model_name, activation_function=activation_function, num_channels=1, num_classes=2)
        model = model.to(device)

        trainset = StdNormalDataset(NADs[idx].reshape(INPUT_SHAPE), 
                                    num_samples=args.data_size, 
                                    non_linearity_type=args.non_linearity_type, 
                                    epsilon=epsilon, sigma=args.noise_std, shape=INPUT_SHAPE)
        testset = StdNormalDataset(NADs[idx].reshape(INPUT_SHAPE), 
                                   num_samples=args.data_size // 5, 
                                   non_linearity_type=args.non_linearity_type, 
                                   epsilon=epsilon, sigma=args.noise_std, shape=INPUT_SHAPE, bias=trainset.bias)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=2, pin_memory=True)

        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=0, pin_memory=True)

        model = train(model, trainloader, testloader, epochs=n_iterations, max_lr=args.max_lr, momentum=0.9, weight_decay=0.0, verbose=False)
        
        test_acc_result[idx].append(test(model, testloader, device=device)[0])
        train_acc_result[idx].append(test(model, trainloader, device=device)[0])
    print(idx, test_acc_result[idx])

print()
print('Experiment settings: ')
print(args)
print('-' * 10)
print('Test accuracy: ')
print(test_acc_result)
print('-' * 10)
print('Train accuracy: ')
print(train_acc_result)
print('-' * 10)
print('G eingevalues: ')
print(eigenvalues)