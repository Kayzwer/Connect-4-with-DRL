from Connect4 import Connect4
from collections import deque
from typing import Deque, Dict, Tuple
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Network(nn.Module):
    def __init__(self, output_dim: int, learning_rate: float) -> None:
        super(Network, self).__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(1, 64, 4),
            nn.Mish(),
            nn.Conv2d(64, 32, 2),
            nn.Mish(),
            nn.Conv2d(32, 16, 2),
            nn.Flatten(),
        )
        self.advantage_layers = nn.Sequential(
            nn.Linear(32, 16),
            nn.Mish(),
            nn.Linear(16, output_dim)
        )
        self.value_layers = nn.Sequential(
            nn.Linear(32, 8),
            nn.Mish(),
            nn.Linear(8, 1)
        )
        self.optimizer = optim.Adadelta(self.parameters(), lr = learning_rate)
        self.loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.feature_layers(x)
        advantage = self.advantage_layers(feature)
        value = self.value_layers(feature)
        return value + advantage - advantage.mean(dim = -1, keepdim = True)

    def get_prob(self, x: torch.Tensor) -> np.ndarray:
        return F.softmax(self.forward(x), dim = -1).detach().numpy()

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
        input_dim: tuple,
        n_step: int,
        gamma: float,
    ) -> None:
        self.ptr = 0
        self.cur_size = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_memory = np.zeros((buffer_size, *input_dim), dtype = np.float32)
        self.action_memory = np.zeros(buffer_size, dtype = np.longlong)
        self.reward_memory = np.zeros(buffer_size, dtype = np.float32)
        self.next_state_memory = np.zeros((buffer_size, *input_dim), dtype = np.float32)

        self.n_step_buffer = deque(maxlen = n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray
    ) -> Tuple[np.ndarray, int, float, np.ndarray]:
        transition = (state, action, reward, next_state)
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return ()
        reward, next_state = self._get_n_step_info(self.n_step_buffer, self.gamma)
        state, action = self.n_step_buffer[0][:2]
        self.state_memory[self.ptr] = state
        self.action_memory[self.ptr] = action
        self.reward_memory[self.ptr] = reward
        self.next_state_memory[self.ptr] = next_state
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.cur_size = min(self.cur_size + 1, self.buffer_size)
        return self.n_step_buffer[0]
    
    def _get_n_step_info(self, n_step_buffer: Deque, gamma: float) -> Tuple[np.longlong, np.ndarray]:
        reward, next_state = n_step_buffer[-1][-2:]
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_s = transition[-2:]
            reward = r + gamma * reward
            next_state = n_s
        return reward, next_state
    
    def sample(self) -> Dict[str, np.ndarray]:
        indexes = np.random.choice(self.cur_size, self.batch_size, False)
        states = self.state_memory[indexes]
        actions = self.action_memory[indexes]
        rewards = self.reward_memory[indexes]
        next_states = self.next_state_memory[indexes]
        return dict(
            states = torch.from_numpy(states),
            actions = torch.from_numpy(actions),
            rewards = torch.from_numpy(rewards),
            next_states = torch.from_numpy(next_states),
            indexes = torch.from_numpy(indexes)
        )
    
    def sample_from_idxs(self, indexes: torch.Tensor) -> Dict[str, np.ndarray]:
        return dict(
            states = torch.from_numpy(self.state_memory[indexes]),
            actions = torch.from_numpy(self.action_memory[indexes]),
            rewards = torch.from_numpy(self.reward_memory[indexes]),
            next_states = torch.from_numpy(self.next_state_memory[indexes])
        )
    
    def is_ready(self) -> bool:
        return self.cur_size >= self.batch_size


class Agent:
    def __init__(
        self,
        input_dim: tuple,
        output_dim: int,
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        init_eps: float,
        eps_dec_rate: str, 
        min_eps: float,
        gamma: float,
        n_step: int,
        c: int
    ) -> None:
        self.network = Network(output_dim, learning_rate)
        self.target_network = Network(output_dim, learning_rate)
        self.replay_buffer = Replay_Buffer(buffer_size, batch_size, input_dim, 1, gamma)
        self.n_step_replay_buffer = Replay_Buffer(buffer_size, batch_size, input_dim, n_step, gamma)
        self.epsilon_controller = Epsilon_Controller(init_eps, eps_dec_rate, min_eps)
        self.output_dim = output_dim
        self.gamma = gamma
        self.c, self.update_count = c, 0
        self.update_target_network()

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.network.state_dict())
        print("Target Network Updated")

    def choose_action_train(self, state: np.ndarray, env: Connect4) -> int:
        if np.random.random() < self.epsilon_controller.eps:
            action = np.random.choice(self.output_dim)
            while not env._check_valid(action):
                action = np.random.choice(self.output_dim)
        else:
            probs = self.network.get_prob(torch.as_tensor(state, dtype = torch.float32).unsqueeze(0)).squeeze()
            action = np.random.choice(self.output_dim, p = probs)
            while not env._check_valid(action):
                action = np.random.choice(self.output_dim, p = probs)
        return action

    def choose_action_test(self, state: np.ndarray, env: Connect4) -> int:
        state = torch.as_tensor(state, dtype = torch.float32).unsqueeze(0)
        probs = self.network.get_prob(state).squeeze()
        action = self.network.forward(state).argmax().item()
        while not env._check_valid(action):
            action = np.random.choice(self.output_dim, p = probs)
        return action

    def _compute_loss(self, batch: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        states = batch.get("states")
        actions = batch.get("actions")
        rewards = batch.get("rewards")
        next_states = batch.get("next_states")
        batch_index = np.arange(self.replay_buffer.batch_size, dtype = np.longlong)

        q_pred = self.network.forward(states)[batch_index, actions]
        probs = self.network.get_prob(states)
        q_next_state_actions = torch.tensor([np.random.choice(env.action_dim, p = prob) for prob in probs], dtype = torch.long)
        q_next = self.target_network.forward(next_states)[batch_index, q_next_state_actions]
        q_target = rewards + gamma * q_next
        return self.network.loss(q_pred, q_target)
    
    def train(self) -> torch.Tensor:
        batch = self.replay_buffer.sample()
        indexes = batch.get("indexes")
        loss = self._compute_loss(batch, indexes)

        batch = self.n_step_replay_buffer.sample_from_idxs(indexes)
        gamma = self.gamma ** self.n_step_replay_buffer.n_step
        n_loss = self._compute_loss(batch, gamma)
        loss += n_loss
        
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()
        self.update_count += 1
        if self.update_count % self.c == 0:
            self.update_target_network()
        self.epsilon_controller.decay()
        return loss

    def test(self, n_game: int, env: Connect4) -> None:
        for i in range(n_game):
            state = env.reset()
            done = False
            score = 0
            while not done:
                action = self.choose_action_test(state, env)
                state, reward, done = env.step(action)
                print(env)
                score += reward
            print(f"Game: {i + 1}, Score: {score}")


if __name__ == "__main__":
    env = Connect4()
    agent1 = Agent(
        env.state_shape,
        env.action_dim, 
        0.0001, 20000, 1024,
        1.0, "0.0001", 0.001,
        0.99, 6, 2048
    )
    agent2 = Agent(
        env.state_shape,
        env.action_dim, 
        0.0001, 20000, 1024,
        1.0, "0.0001", 0.001,
        0.99, 6, 2048
    )
    iteration = 10000
    for i in range(iteration):
        state = env.reset()
        loss1 = 0
        loss2 = 0
        winner = ""
        done = False
        after_first = False
        agent1_transition = []
        agent2_transition = []

        if i % 2 == 0:
            while not done:
                action1 = agent1.choose_action_train(state, env)
                agent1_transition += [state, action1]
                state, reward_1, reward_2, done = env.step(action1, 1)
                if done:
                    agent1_transition += [reward_1, state]
                    one_step_transition1 = agent1.n_step_replay_buffer.store(*agent1_transition)
                    if one_step_transition1:
                        agent1.replay_buffer.store(*one_step_transition1)
                    agent1_transition.clear()
                    agent2_transition += [reward_2, state]
                    one_step_transition2 = agent2.n_step_replay_buffer.store(*agent2_transition)
                    if one_step_transition2:
                        agent2.replay_buffer.store(*one_step_transition2)
                    agent2_transition.clear()
                    if reward_1 == 0.0:
                        winner = "Draw"
                    elif reward_1 == 1.0:
                        winner = "Player 1"
                    break
                if after_first:
                    agent2_transition += [reward_2, state]
                    one_step_transition2 = agent2.n_step_replay_buffer.store(*agent2_transition)
                    if one_step_transition2:
                        agent2.replay_buffer.store(*one_step_transition2)
                    agent2_transition.clear()
            
                action2 = agent2.choose_action_train(state, env)
                agent2_transition += [state, action2]
                state, reward_1, reward_2, done = env.step(action2, -1)
                if done:
                    agent1_transition += [reward_1, state]
                    one_step_transition1 = agent1.n_step_replay_buffer.store(*agent1_transition)
                    if one_step_transition1:
                        agent1.replay_buffer.store(*one_step_transition1)
                    agent1_transition.clear()
                    agent2_transition += [reward_2, state]
                    one_step_transition2 = agent2.n_step_replay_buffer.store(*agent2_transition)
                    if one_step_transition2:
                        agent2.replay_buffer.store(*one_step_transition2)
                    agent2_transition.clear()
                    if reward_2 == 0.0:
                        winner = "Draw"
                    elif reward_2 == 1.0:
                        winner = "Player 2"
                    break
                agent1_transition += [reward_1, state]
                one_step_transition1 = agent1.n_step_replay_buffer.store(*agent1_transition)
                if one_step_transition1:
                    agent1.replay_buffer.store(*one_step_transition1)
                agent1_transition.clear()
                after_first = True

                if agent1.replay_buffer.is_ready() and agent2.replay_buffer.is_ready():
                    loss1 = agent1.train()
                    loss2 = agent2.train()
        else:
            while not done:
                action2 = agent2.choose_action_train(state, env)
                agent2_transition += [state, action2]
                state, reward_1, reward_2, done = env.step(action2, -1)
                if done:
                    agent2_transition += [reward_2, state]
                    one_step_transition2 = agent2.n_step_replay_buffer.store(*agent2_transition)
                    if one_step_transition2:
                        agent2.replay_buffer.store(*one_step_transition2)
                    agent2_transition.clear()
                    agent1_transition += [reward_1, state]
                    one_step_transition1 = agent1.n_step_replay_buffer.store(*agent1_transition)
                    if one_step_transition1:
                        agent1.replay_buffer.store(*one_step_transition1)
                    agent1_transition.clear()
                    if reward_2 == 0.0:
                        winner = "Draw"
                    elif reward_2 == 1.0:
                        winner = "Player 2"
                    break
                if after_first:
                    agent1_transition += [reward_1, state]
                    one_step_transition1 = agent1.n_step_replay_buffer.store(*agent1_transition)
                    if one_step_transition1:
                        agent1.replay_buffer.store(*one_step_transition1)
                    agent1_transition.clear()
            
                action1 = agent1.choose_action_train(state, env)
                agent1_transition += [state, action1]
                state, reward_1, reward_2, done = env.step(action1, 1)
                if done:
                    agent2_transition += [reward_2, state]
                    one_step_transition2 = agent2.n_step_replay_buffer.store(*agent2_transition)
                    if one_step_transition2:
                        agent2.replay_buffer.store(*one_step_transition2)
                    agent2_transition.clear()
                    agent1_transition += [reward_1, state]
                    one_step_transition1 = agent1.n_step_replay_buffer.store(*agent1_transition)
                    if one_step_transition1:
                        agent1.replay_buffer.store(*one_step_transition1)
                    agent1_transition.clear()
                    if reward_1 == 0.0:
                        winner = "Draw"
                    elif reward_1 == 1.0:
                        winner = "Player 1"
                    break
                agent2_transition += [reward_2, state]
                one_step_transition2 = agent2.n_step_replay_buffer.store(*agent2_transition)
                if one_step_transition2:
                    agent2.replay_buffer.store(*one_step_transition2)
                agent2_transition.clear()
                after_first = True

                if agent1.replay_buffer.is_ready() and agent2.replay_buffer.is_ready():
                    loss1 = agent1.train()
                    loss2 = agent2.train()
        
        print(f"Iteration: {i + 1}, Winner: {winner}, Agent1 Loss: {loss1}, Agent2 Loss: {loss2}, Epsilon: {agent1.epsilon_controller.eps}")

        if (i + 1) % 50 == 0:

            with open("CNN DDQN Connect 4 Agent1.pickle", "wb") as f:
                pickle.dump(agent1, f)
            
            with open("CNN DDQN Connect 4 Agent2.pickle", "wb") as f:
                pickle.dump(agent2, f)
            
            print("Model Saved")
