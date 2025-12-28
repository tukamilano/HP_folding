import numpy as np
import time

def piece_move_dict(piece, board_size):
    piece_move_list_dict = dict()
    for piece_place0 in range(board_size[0]):
        for piece_place1 in range(board_size[1]):
            if piece == 'ROOK':
                move_list = [(0, i) for i in range(-board_size[0]+1, board_size[0]) if i != 0] +\
                            [(i, 0) for i in range(-board_size[1]+1, board_size[1]) if i != 0]
            elif piece == 'KING':
                move_list = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            elif piece == 'QUEEN':
                board_min = min(board_size[0], board_size[1])
                move_list = [(0, i) for i in range(-board_size[0]+1, board_size[0]) if i != 0] +\
                            [(i, 0) for i in range(-board_size[1]+1, board_size[1]) if i != 0] +\
                            [(i, i) for i in range(-board_min+1, board_min) if i != 0] +\
                            [(i, -i) for i in range(-board_min+1, board_min) if i != 0]              
            elif piece == 'KNIGHT':
                move_list = [(2, 1), (2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2), (-2, 1), (-2, -1)]
            else:
                raise Exception('unexpected piece')
            
            piece_move_list_dict[(piece_place0, piece_place1)] = [(piece_place0 + move[0], piece_place1 + move[1]) for move in move_list 
                            if (0 <= piece_place0 + move[0]) and (piece_place0 + move[0] < board_size[0]) and (0 <= piece_place1 + move[1]) and (piece_place1 + move[1] < board_size[1])]
            
    return piece_move_list_dict

UNVISITED = 0
VISITED = 1
PIECE = -1

class Board():
    def __init__(self, board, piece):
        # store board as numpy array for shape/fast updates
        self.board = np.array(board, dtype=int)
        self.piece = piece
        self.memo = {}

        self.board_size = self.board.shape
        self.piece_move_dict = piece_move_dict(self.piece, self.board_size)

    def result(self, current_state=1, alpha=-float('inf'), beta=float('inf')):
        memoise_key = (current_state, tuple(map(tuple, self.board)))
        if memoise_key in self.memo:
            return self.memo[memoise_key]
        
        pos = np.argwhere(self.board == PIECE)

        if pos.size == 0:
            piece_pos = None
            piece_valid_move_candid = [(i, j) for i in range(self.board_size[0]) for j in range(self.board_size[1])]
        else:
            piece_pos = tuple(pos[0])
            piece_move_candid = self.piece_move_dict[piece_pos]
            piece_valid_move_candid = [move for move in piece_move_candid if self.board[move] == UNVISITED]
        if not piece_valid_move_candid:
            winner = -current_state
            self.memo[memoise_key] = winner
            return winner

        pruned = False
        if current_state == 1: # Maximizing player
            current_value = -float('inf')
            for i in piece_valid_move_candid:
                if piece_pos is not None:
                    self.board[piece_pos] = VISITED
                self.board[i] = PIECE
                next_state = current_state if piece_pos is None else -current_state
                value = self.result(next_state, alpha, beta)
                self.board[i] = UNVISITED
                if piece_pos is not None:
                    self.board[piece_pos] = PIECE
                current_value = max(current_value, value)
                alpha = max(alpha, current_value)
                if beta <= alpha:
                    pruned = True
                    break # Beta cut-off
        else: # Minimizing player
            current_value = float('inf')
            for i in piece_valid_move_candid:
                self.board[i] = current_state
                next_state = current_state if piece_pos is None else -current_state
                value = self.result(next_state, alpha, beta)
                self.board[i] = 0
                current_value = min(current_value, value)
                beta = min(beta, current_value)
                if beta <= alpha:
                    pruned = True
                    break # Alpha cut-off
        
        # Memoize only when the value is exact (no pruning); otherwise
        # we might store a lower/upper bound and reuse it incorrectly.
        if not pruned:
            self.memo[memoise_key] = current_value
        return current_value

    def winning_initial_positions(self):
        """Return all starting squares (no pieces on board) where the first player wins.

        We reuse the shared memo table. We temporarily clear the board to all
        UNVISITED, place the piece on each cell, and evaluate keeping the same
        player to move (placement does not swap turns). The board is restored
        to its original contents after the search.
        """
        original_board = self.board.copy()
        self.board[:] = UNVISITED

        winning_cells = []
        n = self.board_size[0]
        k = (n + 1) // 2
        for r in range(k):
            for c in range(r, k):
                if self.board[r, c] != UNVISITED:
                    continue
                self.board[r, c] = PIECE
                result = self.result(current_state=1)
                if result == 1:
                    winning_cells.extend([
                        (r, c),
                        (r, n - 1 - c),
                        (n - 1 - r, c),
                        (n - 1 - r, n - 1 - c),
                        (c, r),
                        (n - 1 - c, r),
                        (c, n - 1 - r),
                        (n - 1 - c, n - 1 - r),
                    ])
                self.board[r, c] = UNVISITED

        self.board[:, :] = original_board
        winning_cells = sorted(set(winning_cells))
        return winning_cells
'''
board = [[UNVISITED, UNVISITED, UNVISITED],
         [UNVISITED, UNVISITED, UNVISITED],
         [UNVISITED, UNVISITED, PIECE]]
piece = 'KING'
print(B0.result())
'''
time0 = time.time()
board = [[UNVISITED for _ in range(4)] for _ in range(4)]
piece = 'QUEEN'
B0 = Board(board, piece)
print(B0.winning_initial_positions())
print(time.time() - time0)
