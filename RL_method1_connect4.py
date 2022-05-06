from typing import Dict
from Connect4 import Connect4
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Network(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float
    ) -> None:
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Mish(),
            nn.Linear(128, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, 128),
            nn.Mish(),
            nn.Linear(128, output_dim)
        )
        self.optimizer = optim.RMSprop(self.parameters(), learning_rate)
        self.loss = nn.SmoothL1Loss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers.forward(x.float())


class Epsilon_Controller:
    def __init__(
        self,
        init_eps: float,
        eps_dec_rate: str,
        min_eps: float
    ) -> None:
        self.eps = init_eps
        self.eps_dec_rate = eps_dec_rate
        self.min_eps = min_eps
        self._deci_place = self._get_deci_place()
    
    def decay(self) -> None:
        self.eps = round(self.eps - float(self.eps_dec_rate), self._deci_place) if self.eps > self.min_eps else self.min_eps

    def _get_deci_place(self) -> int:
        after_dot = False
        count = 0
        for char in self.eps_dec_rate:
            if char == ".":
                after_dot = True
            if after_dot:
                count += 1
        return count


class Replay_Buffer:
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        input_dim: int
    ) -> None:
        self.ptr = 0
        self.cur_size = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_memory = np.zeros((input_dim, batch_size), dtype = np.float32)
        self.action_memory = np.zeros(batch_size, dtype = np.int8)
        self.reward_memory = np.zeros(buffer_size, dtype = np.float32)
        self.next_state_memory = np.zeros((input_dim, buffer_size), dtype = np.float32)

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray
    ) -> None:
        self.state_memory[self.ptr] = state
        self.action_memory[self.ptr] = action
        self.reward_memory[self.ptr] = reward
        self.next_state_memory[self.ptr] = next_state
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.cur_size = min(self.cur_size + 1, self.buffer_size)
    
    def sample(self) -> Dict[str, np.ndarray]:
        indexes = np.random.choice(self.cur_size, self.batch_size, False)
        states = self.state_memory[indexes]
        actions = self.action_memory[indexes]
        rewards = self.reward_memory[indexes]
        next_states = self.next_state_memory[indexes]
        return dict(
            states = states,
            actions = actions,
            rewards = rewards,
            next_states = next_states
        )
    
    def is_ready(self) -> bool:
        return self.cur_size >= self.batch_size


class Agent:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        init_eps: float,
        eps_dec_rate: str, 
        min_eps: float,
        gamma: float,
        c: int
    ) -> None:
        self.network = Network(input_dim, output_dim, learning_rate)
        self.target_network = Network(input_dim, output_dim, learning_rate)
        self.replay_buffer = Replay_Buffer(buffer_size, batch_size, input_dim)
        self.epsilon_controller = Epsilon_Controller(init_eps, eps_dec_rate, min_eps)
        self.output_dim = output_dim
        self.gamma = gamma
        self.c, self.update_count = c, 0
        self.update_target_network()

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())

    def choose_action_train(self, state: torch.Tensor) -> int:
        if np.random.random() < self.epsilon_controller.eps:
            return np.random.choice(self.output_dim)
        else:
            return self.network.forward(torch.tensor(state)).argmax().item()

    def choose_action_test(self, state: torch.Tensor) -> int:
        return self.network.forward(torch.tensor(state)).argmax().item()

    def _compute_loss(self) -> torch.Tensor:
        pass


if __name__ == "__main__":
    env = Connect4()
    network = Network(env.state_dim, env.action_dim, 0.001)

    state = env.reset()
    result = network.forward(torch.tensor(state))
    print(result)
    print(result.argmax())
