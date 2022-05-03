from Connect4 import Connect4
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from math import prod


class Network(nn.Module):
    def __init__(self, input_dim: tuple, output_dim: int, learning_rate: float) -> None:
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(prod(input_dim), 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.optimizer = optim.RMSprop(self.parameters(), lr = learning_rate)
        self.loss = nn.SmoothL1Loss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.float())


if __name__ == "__main__":
    env = Connect4()
    network = Network(env.board.shape, 7, 0.0001)
    print(env.action_space)

    state = env.reset()
    result = network.forward(torch.tensor(state))
    print(result)
    print(result.argmax())
