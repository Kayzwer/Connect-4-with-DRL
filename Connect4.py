from typing import List, Tuple
import numpy as np


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
    reset()
        Reset the game to initial state and return state of the game
    step(action, item)
        Insert item into a column based on action and return information of the game
    _get_random_valid_col()
        Return a random valid action if possible, else -1
    _get_vers()
        Return all possible vertical result
    _get_hors()
        Return all possible horizontal result
    _get_diags()
        Return all possible diagonal result
    _check_valid(col)
        Return a validity of a move
    check_win()
        Return winner if there is else 0
    play()
        Play the game
    play_vs_random()
        Play the game against random
    """

    def __init__(self) -> None:
        """
        Create a 7 x 6 board for the regular connect four game
        Create a list to keep track of the empty slot of each column
        """
        self.state_tuple = (6, 7)
        self.state_dim = 42
        self.action_space = np.arange(7)
        self.action_dim = len(self.action_space)
        
        self.board = np.full(self.state_tuple, 0)
        self._col_ptr = np.full(self.action_dim, 5)

    def __str__(self) -> str:
        return self.board.__str__()
    
    def reset(self) -> np.ndarray:
        """
        Reset the game to initial state

        Returns
        -------
            a flatten 2d array which represent the state of the board
        """
        self.__init__()
        return self.board.flatten()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Insert token into a column based on action

        Parameters
        ----------
        action : int
            The index pf tje column to insert token

        Raises
        ------
        Exception
            If selected column is fulled

        Returns
        -------
            a flatten 2d array which represents the state of the board, 
            a reward in flaot type and, 
            a boolean that represents termination
        """
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
        """
        Returns
        -------
            Return a random valid action if possible, else -1
        """
        while True:
            random_col = np.random.choice(7)
            if self._check_valid(random_col):
                return random_col

    def _get_vers(self) -> List[List[int]]:
        """
        Returns
        -------
        List[List[int]]
            a list of list that each list contain integers, which is all vertical results of the board    
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
            a list of list that each list contain integers, which is all horizontal results of the board
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
            a list of list that each list contain integers, which is all diagonal results of the board
        """
        diags = []
        for i in range(4):
            for j in range(3):
                diags.append([self.board[x + j][x + i] for x in range(4)])
                diags.append([self.board[x + j][6 - x - i] for x in range(4)])
        return diags

    def _check_valid(self, col: int) -> bool:
        """
        Check wether a given action is valid or not

        Returns
        -------
            a boolean to indicate validity of the given action
        """
        return self._col_ptr[col] > -1

    def check_win(self) -> int:
        """
        Returns
        -------
        int
            0 if not player win, 1 if player 1 win, 2 if player 2 win, -1 if draw
        """
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
        """
        Check wether the current state of the game is draw

        Returns
        -------
            a boolean to indicate wether the game is draw
        """
        for ptr in self._col_ptr:
            if ptr != -1:
                return False
        return True
