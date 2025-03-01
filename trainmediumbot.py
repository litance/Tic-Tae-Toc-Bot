import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# 只允许 GPU 运行
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available! Please run on a GPU-enabled machine.")

device = torch.device("cuda")

# 棋盘符号映射
symbol_map = {'X': 1, 'O': -1, ' ': 0}


def board_to_state(board):
    return torch.tensor([symbol_map[s] for s in board], dtype=torch.float32, device=device)


# Dueling DQN 结构
class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(9, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        self.value_stream = nn.Linear(1024, 1)
        self.advantage_stream = nn.Linear(1024, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_values = value + (advantage - advantage.mean())

        return q_values


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


# DQN 训练器
class TicTacToeDQNTrainer:
    def __init__(self, episodes=500000, batch_size=2048, gamma=0.995, lr=0.0001, epsilon_decay=0.99999,
                 min_epsilon=0.01):
        self.device = device
        self.dqn = DuelingDQN().to(self.device)
        self.target_dqn = DuelingDQN().to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.AdamW(self.dqn.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5000)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer()
        self.episodes = episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.update_target_steps = 5000

    def select_action(self, state, valid_moves):
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        with torch.no_grad():
            q_values = self.dqn(state.unsqueeze(0))
        valid_q_values = [(i, q_values[0, i].item()) for i in valid_moves]
        return max(valid_q_values, key=lambda x: x[1])[0]

    def train_step(self):
        if self.buffer.size() < self.batch_size:
            return
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.dqn(states).gather(1, actions)

        # Double DQN 更新
        with torch.no_grad():
            next_q_values = self.dqn(next_states)
            next_actions = torch.argmax(next_q_values, dim=1, keepdim=True)
            target_q_values = rewards + self.gamma * self.target_dqn(next_states).gather(1, next_actions) * (1 - dones)

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(loss)

    def train(self):
        for episode in range(self.episodes):
            board = [' '] * 9
            state = board_to_state(board)
            done = False
            while not done:
                valid_moves = [i for i in range(9) if board[i] == ' ']
                action = self.select_action(state, valid_moves)
                board[action] = 'X'
                reward, done = self.get_reward(board, 'X')

                if not done:
                    opponent_moves = [i for i in range(9) if board[i] == ' ']
                    if opponent_moves:
                        board[random.choice(opponent_moves)] = 'O'
                        reward, done = self.get_reward(board, 'O')

                next_state = board_to_state(board)
                self.buffer.add(state, action, reward, next_state, done)
                state = next_state
                self.train_step()

            if episode % self.update_target_steps == 0:
                self.target_dqn.load_state_dict(self.dqn.state_dict())

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            if episode % 1000 == 0:
                print(f"Episode {episode}: Epsilon = {self.epsilon:.4f}")

        torch.save(self.dqn.state_dict(), "tictactoe_medium.pth")
        print("训练完成，模型已保存！")

    def get_reward(self, board, player):
        if self.check_winner(board, player):
            return 1, True
        elif ' ' not in board:
            return 0.5, True  # 平局
        return 0, False  # 继续游戏

    def check_winner(self, board, player):
        win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                          (0, 3, 6), (1, 4, 7), (2, 5, 8),
                          (0, 4, 8), (2, 4, 6)]
        return any(board[a] == board[b] == board[c] == player for a, b, c in win_conditions)


if __name__ == "__main__":
    trainer = TicTacToeDQNTrainer()
    trainer.train()
