from typing import List, Tuple
import numpy as np


class Connect4:
    def __init__(self) -> None:
        self.state_shape = (6, 7)
        self.state_dim = 42
        self.action_space = np.arange(7)
        self.action_dim = len(self.action_space)
        
        self.board = np.full(self.state_shape, 0)
        self._col_ptr = np.full(self.action_dim, 5)

    def __str__(self) -> str:
        return self.board.__str__()
    
    def reset(self) -> np.ndarray:
        self.__init__()
        return self.board.reshape(6, 7)

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

    def _check_win(self) -> int:
        draw = True
        for ptr in self._col_ptr:
            if ptr != -1:
                draw = False
                break
        if draw:
            return -1
        
        pattern = [[1, 1, 1, 1], [-1, -1, -1, -1]]
        for ver in self._get_vers():
            if ver in pattern:
                return pattern.index(ver) + 1
        for hor in self._get_hors():
            if hor in pattern:
                return pattern.index(hor) + 1
        for diag in self._get_diags():
            if diag in pattern:
                return pattern.index(diag) + 1
        return 0

    def step(self, action: int, token: int) -> Tuple[np.ndarray, float, float, bool]:
        assert 0 <= action <= 6
        row = self._col_ptr[action]
        if row == -1:
            raise Exception("Column is fulled")
        self.board[row][action] = token
        self._col_ptr[action] -= 1

        board = self.board.reshape(6, 7)
        result = self._check_win()
        if result == 0:
            return board, 0.0, 0.0, False
        elif result == 1:
            return board, 1.0, -1.0, True
        elif result == 2:
            return board, -1.0, 1.0, True
        elif result == -1:
            return board, 0.0, 0.0, True
