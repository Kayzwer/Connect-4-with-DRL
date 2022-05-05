from typing import List, Tuple
import numpy as np
import os


class Connect4:
    """
    Connect Four (also known as Connect 4, Four Up, Plot Four, Find Four, Captain's Mistress, Four in a Row,
    Drop Four, and Gravitrips in the Soviet Union) is a two-player connection board game, in which the players choose
    a color and then take turns dropping colored tokens into a seven-column, six-row vertically suspended grid. The
    pieces fall straight down, occupying the lowest available space within the column. The objective of the game is
    to be the first to form a horizontal, vertical, or diagonal line of four of one's own tokens. Connect Four is a
    solved game. The first player can always win by playing the right moves.

    Attributes
    ----------
    board : np.ndarray
        a 2d array that represent the board of the game
    _col_ptr : np.ndarray
        a list of number that keep track of empty slot of each column

    Methods
    -------
    step(action, item)
        Insert item into a column based on action and return information of the game
    reset()
        Reset the game to initial state and return state of the game
    _get_vers()
        Return all possible vertical result
    _get_hors()
        Return all possible horizontal result
    _get_diags()
        Return all possible diagonal result
    check_win()
        Return winner if there is else 0
    play()
        Play the game
    """

    def __init__(self) -> None:
        """
        Create a 7 x 6 board for the regular connect four game.
        Create a list to keep track of the empty slot of each column.
        """
        self.state_dim = (6, 7)
        self.action_space = np.arange(7)
        self.action_dim = len(self.action_space)
        
        self.board = np.full(self.state_dim, 0)
        self._col_ptr = np.full(self.action_dim, 5)

    def __str__(self) -> str:
        return self.board.__str__()
    
    def reset(self) -> np.ndarray:
        """
        Reset the game to initial state

        Returns
        -------
            a 2d array which represent the state of the board
        """
        self.__init__()
        return self.board

    def step(self, action: int, item: int) -> Tuple[np.ndarray, float, bool]:
        """
        Insert item into given column index

        Parameters
        ----------
        action : int
            The index of the column to insert item
        item : int
            The element to insert into the selected column

        Raises
        ------
        Exception
            if selected column is fulled.

        Returns
        -------
            a 2d array which represents the state of the board, 
            a reward in flaot type and, 
            a boolean that represents termination
        """
        assert 0 <= action <= 6
        row = self._col_ptr[action]
        if row == -1:
            raise Exception("Column is fulled")
        self.board[row][action] = item
        self._col_ptr[action] -= 1

        done = False
        reward = 0
        result = self.check_win()
        if result == 1:
            done = True
            reward += 1
        elif result == 2:
            done = True
            reward -= 1
        return self.board, reward, done

    def _get_vers(self) -> List[List[int]]:
        """
        Returns
        -------
        List[List[int]]
            a list of list that each list contain integers, which is all vertical results of the board.
        """
        verticals = []
        for index in range(7):
            for i in range(3):
                verticals.append([row[index] for row in self.board][i:i + 4])
        return verticals

    def _get_hors(self) -> List[List[int]]:
        """
        Returns
        -------
        List[List[int]]
            a list of list that each list contain integers, which is all horizontal results of the board.
        """
        horizontals = []
        for index in range(6):
            horizontals.extend([list(self.board[index])[i:i + 4] for i in range(4)])
        return horizontals

    def _get_diags(self) -> List[List[int]]:
        """
        Returns
        -------
        List[List[int]]
            a list of list that each list contain integers, which is all diagonal results of the board.
        """
        diags = []
        for i in range(4):
            for j in range(3):
                diags.append([self.board[x + j][x + i] for x in range(4)])
                diags.append([self.board[x + j][6 - x - i] for x in range(4)])
        return diags

    def check_win(self) -> int:
        """
        Returns
        -------
        int
            0 if not player win, 1 if player 1 win, 2 if player 2 win.
        """
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
        """
        Play the game
        """
        i = 0
        while True:
            os.system("clear")
            print(self)
            col = int(input("Column: "))
            self.step(col, 1 if i % 2 == 0 else -1)
            result = self.check_win()
            if result == 1:
                os.system("clear")
                print(self)
                print("Player 1 win")
                break
            elif result == 2:
                os.system("clear")
                print(self)
                print("Player 2 win")
                break
            else:
                i += 1
