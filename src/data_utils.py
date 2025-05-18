import torch
import numpy as np

import torchvision
import torchvision.transforms as transforms

import copy
from typing import Union

def get_cifar10_cov(data_path: str, batch_size: int = 512, device: str = 'cuda'):
    CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR_STD = [0.247, 0.243, 0.261]

    tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_path, download=True, 
                                            train=True, transform=tf)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False,
                                              num_workers=2, pin_memory=True)
    
    data_cov = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device=device), targets.to(device=device)
        inputs = inputs.reshape(-1, 32 * 32 * 3)

        for x, t in zip(inputs, targets):
            data_cov += torch.outer(x, x) / trainset.data.shape[0]

    return data_cov

def load_cifar10_data(path: str, batch_size: int = 128):
    tf_train = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])

    tf_test = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])

    trainset = torchvision.datasets.CIFAR10(root=path, download=False, 
                                            train=True, transform=tf_train)
    testset = torchvision.datasets.CIFAR10(root=path, download=False, 
                                           train=False, transform=tf_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=0, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                             num_workers=0, pin_memory=True)

    return trainloader, testloader, trainset, testset

def load_cifar2_data(data_path: str, direction: Union[np.typing.ArrayLike, None] = None, non_linearity_type: str = 'linear', 
                     noise_std: float = 0.2, data_size:int = 10000):
    assert non_linearity_type in ['linear', 'quadratic', 'sinusoidal', 'none']
    if non_linearity_type != 'none': assert direction is not None

    def subsample_dataset_synth_label(dataset, sample_size, mean, std, bias=None):        
        indices = np.random.choice(np.arange(dataset.data.shape[0]), size=sample_size, replace=False)

        dataset.data = dataset.data[indices]

        x_ = dataset.data
        noise = noise_std * np.random.randn(*x_.shape) * (x_.reshape(-1, 32 * 32, 3).max(axis=1).reshape(-1, 1, 1, 3) - 
                                                          x_.reshape(-1, 32 * 32, 3).min(axis=1).reshape(-1, 1, 1, 3))
        x = x_ + noise
        if non_linearity_type == 'linear':
            inner = x.reshape(-1, 32 * 32 * 3) @ direction
        elif non_linearity_type == 'quadratic':
            inner = (x.reshape(-1, 32 * 32 * 3) @ direction) ** 2
        elif non_linearity_type == 'sinusoidal':
            inner = np.sin(((x.reshape(-1, 32 * 32 * 3) @ direction) - mean) / std * np.pi)
        
        if bias is None:
            inner_sorted = np.sort(inner)
            bias = - (inner_sorted[inner.shape[0] // 2 - 1] + inner_sorted[inner.shape[0] // 2]) / 2

        dataset.targets = list(((np.sign(inner + bias) + 1) / 2).astype(int))
        dataset.classes = ['0', '1']
        dataset.class_to_idx = {trainset.classes[0]: 0, trainset.classes[1]: 1}

        return dataset, bias
    
    def subsample_dataset_real_label(dataset, sample_size):
        animal_indices = []
        non_animal_indices = []
        for idx in range(dataset.data.shape[0]):
            if dataset.targets[idx] in [2, 3, 4, 5, 6, 7]:
                animal_indices.append(idx)
            else:
                non_animal_indices.append(idx)
        
        class_1_indices = np.random.choice(animal_indices, size=sample_size, replace=False)
        class_2_indices = np.random.choice(non_animal_indices, size=sample_size, replace=False)

        dataset.data = np.concatenate([dataset.data[class_1_indices], dataset.data[class_2_indices]], axis=0)
        dataset.targets = [0 for _ in range(sample_size)] + [1 for _ in range(sample_size)]
        dataset.classes = ['animal', 'non_animal']
        dataset.class_to_idx = {trainset.classes[0]: 0, trainset.classes[1]: 1}

        return dataset

    CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
    CIFAR_STD = [0.247, 0.243, 0.261]

    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=data_path, download=False, train=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root=data_path, download=False, train=False, transform=test_transform)

    if non_linearity_type == 'none':
        testset = subsample_dataset_real_label(testset, data_size // 5)
        trainset = subsample_dataset_real_label(trainset, data_size)
    else:
        mean = np.mean(trainset.data.reshape(-1, 32 * 32 * 3) @ direction)
        std = np.std(trainset.data.reshape(-1, 32 * 32 * 3) @ direction)

        testset, bias = subsample_dataset_synth_label(testset, data_size // 5, mean, std)
        trainset, _ = subsample_dataset_synth_label(trainset, data_size, mean, std, bias)

    return trainset, testset

def load_pruned_cifar_data(score: np.typing.ArrayLike, trainset: torch.utils.data.Dataset, testset: torch.utils.data.Dataset,
                           perc: float = 0, batch_size: int = 128):
    new_data = []
    new_targets = []

    labels = np.array(trainset.targets)

    for curr_class_idx in list(trainset.class_to_idx.values()):
        raw_data = trainset.data[labels == curr_class_idx]

        class_alignment = score[labels == curr_class_idx].detach().cpu().numpy()
        indices = np.argsort(class_alignment)[int(perc * raw_data.shape[0]):]

        new_data.append(raw_data[indices])
        new_targets += [curr_class_idx for idx in indices]
   
    pruned_trainset = copy.deepcopy(trainset)
    
    pruned_trainset.data = np.concatenate(new_data, axis=0)
    pruned_trainset.targets = new_targets

    trainloader = torch.utils.data.DataLoader(pruned_trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=2, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                             num_workers=2, pin_memory=True)

    return trainloader, testloader