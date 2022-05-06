from typing import List, Tuple
import numpy as np
import os


class Connect4:
    def __init__(self) -> None:
        self.state_tuple = (6, 7)
        self.state_dim = 42
        self.action_space = np.arange(7)
        self.action_dim = len(self.action_space)
        
        self.board = np.full(self.state_tuple, 0)
        self._col_ptr = np.full(self.action_dim, 5)

    def __str__(self) -> str:
        return self.board.__str__()
    
    def reset(self) -> np.ndarray:
        self.__init__()
        return self.board.flatten()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        assert 0 <= action <= 6
        row = self._col_ptr[action]
        if row == -1:
            raise Exception("Column is fulled")
        self.board[row][action] = 1
        self._col_ptr[action] -= 1

        result = self.check_win()
        if result == -1:
            return self.board.flatten(), 0.0, True
        elif result == 1:
            return self.board.flatten(), 1.0, True
        else:
            random_col = self._get_random_valid_col()
            row = self._col_ptr[random_col]
            self.board[row][random_col] = -1
            self._col_ptr[random_col] -= 1
            result = self.check_win()
            if result == -1:
                return self.board.flatten(), 0.0, True
            elif result == 2:
                return self.board.flatten(), -1.0, True
            else:
                return self.board.flatten(), 0.0, False
    
    def _get_random_valid_col(self) -> int:
        while True:
            random_col = np.random.choice(7)
            if self._check_valid(random_col):
                return random_col

    def _get_vers(self) -> List[List[int]]:
        verticals = []
        for index in range(7):
            for i in range(3):
                verticals.append([row[index] for row in self.board][i:i + 4])
        return verticals

    def _get_hors(self) -> List[List[int]]:
        horizontals = []
        for index in range(6):
            horizontals.extend([list(self.board[index])[i:i + 4] for i in range(4)])
        return horizontals

    def _get_diags(self) -> List[List[int]]:
        diags = []
        for i in range(4):
            for j in range(3):
                diags.append([self.board[x + j][x + i] for x in range(4)])
                diags.append([self.board[x + j][6 - x - i] for x in range(4)])
        return diags

    def _check_valid(self, col: int) -> bool:
        return self._col_ptr[col] > -1

    def check_win(self) -> int:
        if self.is_draw():
            return -1
        pattern = [[1, 1, 1, 1], [-1, -1, -1, -1]]
        player = 0
        for ver in self._get_vers():
            if ver in pattern:
                player = pattern.index(ver) + 1
        for hor in self._get_hors():
            if hor in pattern:
                player = pattern.index(hor) + 1
        for diag in self._get_diags():
            if diag in pattern:
                player = pattern.index(diag) + 1
        return player
    
    def is_draw(self) -> bool:
        for ptr in self._col_ptr:
            if ptr != -1:
                return False
        return True
    
    def play_vs_random(self) -> None:
        done = False
        while not done:
            print(self)
            col = int(input("Column: "))
            _, score, done = self.step(col)
            print(score, end = "\n")


if __name__ == "__main__":
    env = Connect4()
    env.play_vs_random()
