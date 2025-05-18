import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Union

from tqdm import tqdm

def compute_G(n_model_samples: int, n_data_samples: int, model_gen_fun: Callable, 
             dataloader: Union[torch.utils.data.DataLoader, None] = None, 
             input_shape: List[int] = [1, 32, 32], 
             sigma_x: float = 1, device='cuda'):
    G = 0
    for n in tqdm(range(n_model_samples)):
        net = model_gen_fun()
        net = net.to(device)

        net.eval()
        if dataloader is None:
            inputs = sigma_x * torch.randn([n_data_samples] + input_shape, device=device)
        else:
            try:
                inputs, _ = next(data_iter)
            except:
                data_iter = iter(dataloader)
                inputs, _ = next(data_iter)
            inputs = inputs.to(device)

        inputs.requires_grad = True

        output = net(inputs)
        
        (grads,) = torch.autograd.grad(output.mean(), inputs)
        grads = grads.reshape(-1, np.prod(input_shape))

        G += grads.transpose(0, 1) @ grads / n_model_samples
    
    return G

def compute_G_t(model: nn.Module, batch_size: int = 128, data_shape: List[int] = [3, 32, 32],
                dataloader: Union[torch.utils.data.DataLoader, None] = None, 
                device: str = 'cuda'):
    if dataloader is None:
        inputs = torch.randn([batch_size] + data_shape, device=device)
        targets = torch.zeros([batch_size], device=device).long()
    else:
        try:
            inputs, targets = next(data_iter)
        except:
            data_iter = iter(dataloader)
            inputs, targets = next(data_iter)
            inputs, targets = inputs.to(device), targets.to(device)

    inputs.requires_grad = True
    
    model.eval()

    output = model(inputs)
    (grads,) = torch.autograd.grad(torch.gather(output, 1, targets.unsqueeze(1)).squeeze().sum(), inputs)
    grads = grads.reshape(-1, np.prod(data_shape))

    cov = grads.transpose(0, 1) @ grads / batch_size

    return cov

def compute_G_eig(G: torch.Tensor):
    # check if G is a symmetric matrix
    assert len(G.shape) == 2
    assert G.shape[0] == G.shape[1]
    assert ((G + G.transpose(0, 1)) / 2 - G).norm() < 1e-4

    L, V = torch.linalg.eigh(G)
    V = V.transpose(0, 1)

    L, V = torch.flip(L, dims=[0]), torch.flip(V, dims=[0])
    L, V = L.detach(), V.detach()

    return L, V