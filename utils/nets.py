from abc import abstractmethod
from typing import List, Any

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
from torch.distributions import Normal
from torch.nn import Parameter


class DummyNet(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ScalerNet(nn.Module):

    def __init__(self, scaler: StandardScaler):
        super().__init__()
        self.scaler = scaler

    def forward(self, x: torch.Tensor):
        input = x[:].cpu()
        if len(x.shape) == 1:
            input = input.reshape(1, -1)
        transformed = self.scaler.transform(input)
        if len(x.shape) == 1:
            transformed = transformed.reshape(-1)
        return torch.as_tensor(transformed).float()


class ScalerNetLoose(ScalerNet):
    """
    Scale only the first k entries of the input, and ignore the rest
    """

    def forward(self, x: torch.Tensor):
        k, = self.scaler.mean_.shape
        input = x[:].cpu()
        if len(x.shape) == 1:
            input = input.reshape(1, -1)
        input_to_scale = input[:, :k]
        input_rest = input[:, k:]
        transformed = self.scaler.transform(input_to_scale)
        ret = torch.cat([transformed, input_rest], dim=1)
        if len(x.shape) == 1:
            ret = ret.reshape(-1)
        return torch.as_tensor(ret).float()


class MultiLayerPerceptron(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], activation: nn.Module,
                 final_layer_activation: nn.Module):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation: nn.Module = activation
        self.final_layer_activation = final_layer_activation
        self.fcs: nn.ModuleList = nn.ModuleList()
        layer_dims: np.ndarray = np.array([input_dim, *hidden_dims, output_dim])
        for dim1, dim2 in zip(layer_dims, layer_dims[1:]):
            fc = nn.Linear(dim1, dim2)
            # randomize the weights
            nn.init.normal(fc.weight, mean=0.0, std=1.0)
            self.fcs.append(fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fc in self.fcs[:-1]:
            x = fc(x)
            x = self.activation.forward(x)
        x = self.fcs[-1](x)
        x = self.final_layer_activation(x)
        return x


class ProbNet(nn.Module):
    """
    Two-handed network whose forward() returns a distribution
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def sample(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        raise NotImplementedError


class ProbMLPConstantLogStd(ProbNet):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], activation: nn.Module,
                 final_layer_activation: nn.Module, log_std: float):
        super().__init__()
        self.mlp = MultiLayerPerceptron(input_dim, output_dim, hidden_dims, activation, final_layer_activation)
        self.log_std = log_std  # fixed log_std is superior for exploration

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        mu: torch.Tensor = self.mlp.forward(x)
        log_std: torch.Tensor = torch.ones_like(mu) * self.log_std
        return mu, log_std

    def sample(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        mu, log_std = self.forward(x)
        sigma = torch.exp(log_std)
        normal_distribution = Normal(mu, sigma)
        action = normal_distribution.sample()
        log_prob = normal_distribution.log_prob(action)
        return action, log_prob

    def get_log_prob(self, input: torch.Tensor, output: torch.Tensor):
        """
        Sampling has produced the output. Based on current distribution, what is the probability?
        """
        mu, log_std = self.forward(input)
        std = torch.exp(log_std)
        normal = Normal(mu, std)
        log_prob = normal.log_prob(output)
        return log_prob


class ProbMLPLearnedLogStd(ProbMLPConstantLogStd):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], activation: nn.Module,
                 final_layer_activation: nn.Module):
        super().__init__(input_dim, output_dim, hidden_dims, activation, final_layer_activation, 0)
        self.log_std = Parameter(torch.zeros(output_dim), True)  # learnable exploration noise

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        mu: torch.Tensor = self.mlp.forward(x)
        log_std: torch.Tensor = self.log_std
        return mu, log_std


class ProbMLPLayeredLogStd(ProbMLPConstantLogStd):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], activation: nn.Module,
                 final_layer_activation: nn.Module):
        super().__init__(input_dim, output_dim, hidden_dims, activation, final_layer_activation, 0)
        self.log_std_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        mu: torch.Tensor = self.mlp.forward(x)
        for fc in self.mlp.fcs[:-1]:
            x = fc.forward(x)
            x = self.mlp.activation.forward(x)
        log_std: torch.Tensor = self.log_std_layer.forward(x)
        log_std = torch.clamp(log_std, -2.0, -1.0)
        return mu, log_std
