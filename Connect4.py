from typing import List
import numpy as np


class Connect4:
    def __init__(self) -> None:
        self.board = np.full((6, 7), 0)
        self._col_ptr = np.full(7, 5)
    
    def __str__(self) -> str:
        return self.board.__str__()
    
    def step(self, action: int, item: int) -> None:
        assert 0 <= action <= 6
        row = self._col_ptr[action]
        if row == -1:
            raise Exception("Column is full")
        self.board[row][action] = item
        self._col_ptr[action] -= 1
    
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
    
    def check_win(self) -> int:
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

    
    def play(self) -> None:
        i = 0
        while True:
            print(self)
            col = int(input("Column: "))
            self.step(col, 1 if i % 2 == 0 else -1)

            winner = self.check_win()
            if winner in (1, 2):
                print(self)
                print(f"Player {winner} win the game")
                break
            else:
                i += 1


if __name__ == "__main__":
    temp = Connect4()
    temp.play()
    