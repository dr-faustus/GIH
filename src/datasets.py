import torch
import numpy as np

from typing import List, Union

class InverseCovarDataset(torch.utils.data.Dataset):

    def __init__(self, cov: np.typing.ArrayLike, num_samples: int = 10000, 
                 shape: List[int] = [1, 32, 32]):
        
        self.cov = cov
        self.num_samples = num_samples
        self.shape = shape
        self.data, self.targets = self._generate_dataset(self.cov, self.num_samples)
        super()

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        return img, target

    def __len__(self):
        return self.num_samples

    def _generate_dataset(self, cov, n_samples):    
        data, labels = self._generate_samples(cov, n_samples)
    
        return torch.from_numpy(data).float(), torch.from_numpy(labels)

    def _generate_samples(self, cov, n_samples):
        data = np.random.multivariate_normal(np.zeros(np.prod(self.shape)), cov, size=n_samples)
        labels = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
        return data.reshape(n_samples, *self.shape), labels

class LinearCircleDataset(torch.utils.data.Dataset):

    def __init__(self, u_1: np.typing.ArrayLike, u_2: np.typing.ArrayLike, u_3: np.typing.ArrayLike, 
                 num_samples: int = 10000, sigma: float = 3, epsilon_1: float = 1, epsilon_2: float = 2, epsilon_3: float = 1, 
                 shape: List[int] = [1, 32, 32]):
        
        self.u_1, self.u_2, self.u_3 = u_1, u_2, u_3
        self.num_samples = num_samples
        self.sigma = sigma
        self.epsilon_1, self.epsilon_2, self.epsilon_3 = epsilon_1, epsilon_2, epsilon_3
        self.shape = shape
        self.data, self.targets = self._generate_dataset(self.num_samples)
        super()

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        return img, target

    def __len__(self):
        return self.num_samples

    def _generate_dataset(self, n_samples):
        if n_samples > 1:
            data_plus = self._generate_samples(n_samples // 2 + n_samples % 2, 0).astype(np.float32)
            labels_plus = np.zeros([n_samples // 2 + n_samples % 2]).astype(int)

            data_minus = self._generate_samples(n_samples // 2, 1).astype(np.float32)
            labels_minus = np.ones([n_samples // 2]).astype(int)

            data = np.r_[data_plus, data_minus]
            labels = np.r_[labels_plus, labels_minus]
        else:
            data = self._generate_samples(1, 0).astype(np.float32)
            labels = np.zeros([1]).astype(int)

        return torch.from_numpy(data), torch.from_numpy(labels)

    def _generate_samples(self, n_samples, label):
        data = self._generate_noise_floor(n_samples)
        sign = 1 if label == 0 else -1

        cirlce = self._generate_circle(n_samples, self.epsilon_1 if sign == 1 else self.epsilon_2)
        linear = sign * self.epsilon_3 / 2 * self.u_3[np.newaxis, :]
        
        data = linear + cirlce + self._project_orthogonal(data)
        return data
    
    def _generate_circle(self, n_samples, norm):
        alpha = np.random.randn(n_samples, 1, 1, 1)
        beta = np.random.randn(n_samples, 1, 1, 1)
        
        circle = alpha * self.u_1[np.newaxis, :] + beta * self.u_2[np.newaxis, :]
        circle /= np.linalg.norm(circle.reshape(n_samples, -1), axis=1).reshape(n_samples, 1, 1, 1)

        return circle * norm

    def _generate_noise_floor(self, n_samples):
        shape = [n_samples] + self.shape
        data = self.sigma * np.random.randn(*shape)

        return data

    def _project(self, x, v):
        proj_x = np.reshape(x, [x.shape[0], -1]) @ np.reshape(v, [-1, 1])
        return proj_x[:, :, np.newaxis, np.newaxis] * v[np.newaxis, :]

    def _project_orthogonal(self, x):
        return x - self._project(x, self.u_1) - self._project(x, self.u_2) - self._project(x, self.u_3)
    
class StdNormalDataset(torch.utils.data.Dataset):

    def __init__(self, u: np.typing.ArrayLike, num_samples: int = 10000, non_linearity_type: str = 'linear', 
                 epsilon: float = 1.0, sigma: float = 1.0, bias: Union[None, np.array] = None, 
                 shape: List[int] = [1, 32, 32]):
        
        assert non_linearity_type in ['linear', 'quadratic', 'sinusoidal']
        
        self.u = u
        self.sigma = sigma
        self.epsilon = epsilon
        self.num_samples = num_samples
        self.shape = shape
        self.non_linearity_type = non_linearity_type
        self.bias = bias

        self.data, self.targets = self._generate_dataset(self.num_samples)
        super()

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        return img, target

    def __len__(self):
        return self.num_samples

    def _generate_dataset(self, n_samples):
        shape = [n_samples] + self.shape

        noise = self._project_orthogonal(self._generate_noise_floor(n_samples))

        alpha = self.epsilon * np.random.randn(n_samples, 1, 1, 1)
        
        if self.non_linearity_type == 'linear':
            inner = alpha.reshape(-1)
        elif self.non_linearity_type == 'quadratic':
            inner = alpha.reshape(-1) ** 2
        elif self.non_linearity_type == 'sinusoidal':
            inner = np.sin((alpha.reshape(-1) / self.epsilon) * np.pi)
        
        if self.bias is None:
            inner_sorted = np.sort(inner)
            self.bias = - (inner_sorted[inner.shape[0] // 2 - 1] + inner_sorted[inner.shape[0] // 2]) / 2

        labels = ((np.sign(inner + self.bias) + 1) / 2).astype(int)

        data = alpha * self.u + noise

        return torch.from_numpy(data).float(), torch.from_numpy(labels)
    
    def _generate_circle(self, n_samples, norm):
        shape = [n_samples] + self.shape

        alpha = np.random.randn(n_samples, 1, 1, 1)
        beta = np.random.randn(n_samples, 1, 1, 1)
        
        circle = alpha * self.u_1[np.newaxis, :] + beta * self.u_2[np.newaxis, :]
        circle /= np.linalg.norm(circle.reshape(n_samples, -1), axis=1).reshape(n_samples, 1, 1, 1)

        return circle * norm

    def _generate_noise_floor(self, n_samples):
        shape = [n_samples] + self.shape
        data = self.sigma * np.random.randn(*shape)

        return data

    def _project(self, x, v):
        proj_x = np.reshape(x, [x.shape[0], -1]) @ np.reshape(v, [-1, 1])
        return proj_x[:, :, np.newaxis, np.newaxis] * v[np.newaxis, :]

    def _project_orthogonal(self, x):
        return x - self._project(x, self.u)