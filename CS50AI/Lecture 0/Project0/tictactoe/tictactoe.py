"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if sum(row.count(None) for row in board) % 2 == 0:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()
    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col] is EMPTY:
                actions.add((row, col))
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action is None:
        raise Exception("Invalid action")

    row, col = action

    if not (0 <= row < len(board) and 0 <= col < len(board[row])):
        raise Exception("Invalid Move")
    if board[row][col] is not EMPTY:
        raise Exception("Invalid Move")

    new_board = copy.deepcopy(board)
    new_board[row][col] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    winConditions = [
        [(0, 0), (0, 1), (0, 2)],
        [(1, 0), (1, 1), (1, 2)],
        [(2, 0), (2, 1), (2, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 1), (2, 1)],
        [(0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 1), (2, 2)],
        [(0, 2), (1, 1), (2, 0)],
    ]

    for winCondition in winConditions:
        values = [board[row][col] for row, col in winCondition]
        if values == [X, X, X]:
            return X
        elif values == [O, O, O]:
            return O
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None or sum(row.count(None) for row in board) == 0:
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    def max_value(b):
        if terminal(b):
            return utility(b)
        v = -math.inf
        for action in actions(b):
            v = max(v, min_value(result(b, action)))
            if v == 1:
                return v
        return v

    def min_value(b):
        if terminal(b):
            return utility(b)
        v = math.inf
        for action in actions(b):
            v = min(v, max_value(result(b, action)))
            if v == -1:
                return v
        return v

    turn = player(board)
    best_action = None

    if turn == X:
        best_val = -math.inf
        for action in actions(board):
            val = min_value(result(board, action))
            if val > best_val:
                best_val = val
                best_action = action

    else:
        best_val = math.inf
        for action in actions(board):
            val = max_value(result(board, action))
            if val < best_val:
                best_val = val
                best_action = action

    return best_action
