import torch
import torch.nn as nn
import numpy as np
import tkinter as tk
from tkinter import messagebox


# 加载训练好的 DQN 模型
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class TicTacToeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tic Tac Toe - AI 对战")
        self.board = [' '] * 9
        self.buttons = [
            tk.Button(root, text=' ', font=('Arial', 24), width=5, height=2, command=lambda i=i: self.human_move(i)) for
            i in range(9)]
        for i, btn in enumerate(self.buttons):
            btn.grid(row=i // 3, column=i % 3)
        self.load_model()
        self.update_board()

    def load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN().to(self.device)
        self.dqn.load_state_dict(torch.load("tictactoe_easy.pth", map_location=self.device)) #You can choose game difficult here
        self.dqn.eval()

    def board_to_state(self, board):
        symbol_map = {'X': 1, 'O': -1, ' ': 0}
        return np.array([symbol_map[s] for s in board], dtype=np.float32)

    def ai_move(self):
        available = [i for i in range(9) if self.board[i] == ' ']
        state = torch.tensor(self.board_to_state(self.board), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.dqn(state).cpu().numpy()
        best_move = max(available, key=lambda x: q_values[x])
        return best_move

    def human_move(self, index):
        if self.board[index] == ' ':
            self.board[index] = 'O'
            self.update_board()
            if self.check_winner('O'):
                messagebox.showinfo("游戏结束", "你赢了！")
                self.reset()
                return
            if self.is_draw():
                messagebox.showinfo("游戏结束", "平局！")
                self.reset()
                return

            ai_action = self.ai_move()
            self.board[ai_action] = 'X'
            self.update_board()
            if self.check_winner('X'):
                messagebox.showinfo("游戏结束", "AI 赢了！")
                self.reset()
            elif self.is_draw():
                messagebox.showinfo("游戏结束", "平局！")
                self.reset()

    def check_winner(self, player):
        win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                          (0, 3, 6), (1, 4, 7), (2, 5, 8),
                          (0, 4, 8), (2, 4, 6)]
        return any(self.board[a] == self.board[b] == self.board[c] == player for a, b, c in win_conditions)

    def is_draw(self):
        return ' ' not in self.board

    def reset(self):
        self.board = [' '] * 9
        self.update_board()

    def update_board(self):
        for i in range(9):
            self.buttons[i].config(text=self.board[i])


if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToeGUI(root)
    root.mainloop()
