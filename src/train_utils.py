import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, Union

def test(model: nn.Module, testloader: torch.utils.data.DataLoader, device: str = 'cuda', proj_mat: Union[torch.Tensor, None] = None) -> Tuple[float, float]:
    loss_fun = nn.CrossEntropyLoss()
    test_loss_sum = 0
    test_acc_sum = 0
    test_n = 0

    model.eval()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if proj_mat is not None:
                inputs = (inputs.reshape(inputs.shape[0], -1) @ proj_mat).reshape(inputs.shape)
            
            output = model(inputs)
            loss = loss_fun(output, targets)

            test_loss_sum += loss.item() * targets.size(0)
            test_acc_sum += (output.max(1)[1] == targets).sum().item()
            test_n += targets.size(0)

        test_loss = (test_loss_sum / test_n)
        test_acc = (100 * test_acc_sum / test_n)

    return test_acc, test_loss

def train(model: nn.Module, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader, 
          epochs: int, max_lr: float, momentum: float, weight_decay: float, grad_norm_clip: float = 0.5,
          verbose: bool = False, log_interval: int = 100, device: str = 'cuda', proj_mat: Union[torch.Tensor, None] = None):
    lr_schedule = lambda t: np.interp([t], [0, epochs], [max_lr, 0])[0]
    opt = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=momentum, 
                          weight_decay=weight_decay)
    loss_fun = nn.CrossEntropyLoss()

    if verbose:
        print('Starting training...')
        print()

    for epoch in range(epochs):
        if verbose:
            print('Epoch', epoch)
        train_loss_sum = 0
        train_acc_sum = 0
        train_n = 0

        model.train()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            lr = lr_schedule(epoch + (batch_idx + 1) / len(trainloader))
            opt.param_groups[0].update(lr=lr)

            if proj_mat is not None:
                inputs = (inputs.reshape(inputs.shape[0], -1) @ proj_mat).reshape(inputs.shape)
            
            output = model(inputs)
            loss = loss_fun(output, targets)

            opt.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
            opt.step()

            train_loss_sum += loss.item() * targets.size(0)
            train_acc_sum += (output.max(1)[1] == targets).sum().item()
            train_n += targets.size(0)

            if verbose and batch_idx % log_interval == 0:
                print('Batch idx: %d(%d)\tTrain Acc: %.3f%%\tTrain Loss: %.3f' %
                      (batch_idx, epoch, 100. * train_acc_sum / train_n, train_loss_sum / train_n))
       
        if verbose:
            print('\nTrain Summary\tEpoch: %d | Train Acc: %.3f%% | Train Loss: %.3f' %
                (epoch, 100. * train_acc_sum / train_n, train_loss_sum / train_n))

            test_acc, test_loss = test(model, testloader, proj_mat=proj_mat, device=device)
            print('Test  Summary\tEpoch: %d | Test Acc: %.3f%% | Test Loss: %.3f\n' % (epoch, test_acc, test_loss))

    return model

def train_fixed_iters(model: nn.Module, trainloader: torch.utils.data.DataLoader, testloader: torch.utils.data.DataLoader, 
                      epochs: int, max_lr: float, momentum: float, weight_decay: float, grad_norm_clip: float = 0.5, n_iters: int = 391, 
                      verbose: bool = False, log_interval: int = 100, device: str = 'cuda'):
    lr_schedule = lambda t: np.interp([t], [0, epochs], [max_lr, 0])[0]
    opt = torch.optim.SGD(model.parameters(), lr=max_lr, momentum=momentum, 
                          weight_decay=weight_decay)
    loss_fun = nn.CrossEntropyLoss()

    if verbose:
        print('Starting training...')
        print()

    for epoch in range(epochs):
        if verbose:
            print('Epoch', epoch)
        train_loss_sum = 0
        train_acc_sum = 0
        train_n = 0

        model.train()

        trainloader_iter = iter(trainloader)

        for batch_idx in range(n_iters):
            try:
                inputs, targets = next(trainloader_iter)
            except:
                trainloader_iter = iter(trainloader)
                inputs, targets = next(trainloader_iter)

            inputs, targets = inputs.to(device), targets.to(device)

            lr = lr_schedule(epoch + (batch_idx + 1) / n_iters)
            opt.param_groups[0].update(lr=lr)

            output = model(inputs)
            loss = loss_fun(output, targets)

            opt.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
            opt.step()

            train_loss_sum += loss.item() * targets.size(0)
            train_acc_sum += (output.max(1)[1] == targets).sum().item()
            train_n += targets.size(0)

            if verbose and batch_idx % log_interval == 0:
                print('Batch idx: %d(%d)\tTrain Acc: %.3f%%\tTrain Loss: %.3f' %
                      (batch_idx, epoch, 100. * train_acc_sum / train_n, train_loss_sum / train_n))
       
        if verbose:
            print('\nTrain Summary\tEpoch: %d | Train Acc: %.3f%% | Train Loss: %.3f' %
                (epoch, 100. * train_acc_sum / train_n, train_loss_sum / train_n))

            test_acc, test_loss = test(model, testloader, device=device)
            print('Test  Summary\tEpoch: %d | Test Acc: %.3f%% | Test Loss: %.3f\n' % (epoch, test_acc, test_loss))

    return model