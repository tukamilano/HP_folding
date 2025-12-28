from bitarray import bitarray
import time


def moving_order(board_size):
    def priority(board_size, pos):
        x = min(pos[0], board_size - 1 - pos[0])
        y = min(pos[1], board_size - 1 - pos[1])
        a = min(x, y)
        b = max(x, y)
        return a * board_size + b

    moving_order_list = [priority(board_size, (i, j)) for j in range(board_size) for i in range(board_size)]
    return moving_order_list

def piece_move_dict(piece, board_size):
    piece_move_list_dict = dict()
    for piece_place0 in range(board_size):
        for piece_place1 in range(board_size):
            if piece == 'ROOK':
                move_list = [(0, i) for i in range(-board_size+1, board_size) if i != 0] +\
                            [(i, 0) for i in range(-board_size+1, board_size) if i != 0]
            elif piece == 'KING':
                move_list = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            elif piece == 'QUEEN':
                move_list = [(0, i) for i in range(-board_size+1, board_size) if i != 0] +\
                            [(i, 0) for i in range(-board_size+1, board_size) if i != 0] +\
                            [(i, i) for i in range(-board_size+1, board_size) if i != 0] +\
                            [(i, -i) for i in range(-board_size+1, board_size) if i != 0]              
            elif piece == 'KNIGHT':
                move_list = [(2, 1), (2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2), (-2, 1), (-2, -1)]
            else:
                raise Exception('unexpected piece')
            
            value = [(piece_place0 + move[0], piece_place1 + move[1]) for move in move_list 
                            if (0 <= piece_place0 + move[0]) and (piece_place0 + move[0] < board_size) and (0 <= piece_place1 + move[1]) and (piece_place1 + move[1] < board_size)]
            
            value_to_bitarray = bitarray(board_size * board_size)
            value_to_bitarray.setall(0)
            for v in value:
                value_to_bitarray[v[0] * board_size + v[1]] = 1

            piece_move_list_dict[(piece_place0, piece_place1)] = value_to_bitarray

    return piece_move_list_dict

UNVISITED = 1
VISITED = 0

class Board():
    def __init__(self, board_size, piece):
        # store board as numpy array for shape/fast updates
        self.board_size = board_size
        self.board = bitarray(self.board_size * self.board_size)
        self.board.setall(UNVISITED)
        self.piece = piece
        self.memo = {}
        self.piece_move_dict = piece_move_dict(self.piece, self.board_size)
        self.move_priority = moving_order(self.board_size)
        self.piece_pos = None

    def result(self, current_state=1, alpha=-float('inf'), beta=float('inf')):
        memoise_key = (self.piece_pos, tuple(self.board))
        if memoise_key in self.memo:
            return self.memo[memoise_key]
        
        piece_move_candid = self.piece_move_dict[self.piece_pos]
        legal_moves = piece_move_candid & self.board  # 1 where reachable and unvisited
        n = self.board_size
        b = self.board

        if not any(legal_moves):
            winner = -current_state
            self.memo[memoise_key] = winner
            return winner

        pruned = False
        if current_state == 1: # Maximizing player
            current_value = -float('inf')
            ordered_moves = sorted((i for i, m in enumerate(legal_moves) if m), key=lambda idx: -self.move_priority[idx])
            for i in ordered_moves:
                prev_pos = self.piece_pos
                self.piece_pos = (i // n, i % n)
                b[i] = VISITED
                next_state = -current_state
                value = self.result(next_state, alpha, beta)
                b[i] = UNVISITED
                self.piece_pos = prev_pos
                current_value = max(current_value, value)
                alpha = max(alpha, current_value)
                if beta <= alpha:
                    pruned = True
                    break # Beta cut-off
        else: # Minimizing player
            current_value = float('inf')
            ordered_moves = sorted((i for i, m in enumerate(legal_moves) if m), key=lambda idx: -self.move_priority[idx])
            for i in ordered_moves:
                prev_pos = self.piece_pos
                b[i] = VISITED
                self.piece_pos = (i // n, i % n)
                next_state = -current_state
                value = self.result(next_state, alpha, beta)
                b[i] = UNVISITED
                self.piece_pos = prev_pos
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
        k = (self.board_size + 1) // 2
        for r in range(k):
            for c in range(r,k):
                self.piece_pos = (r, c)
                self.board[r * self.board_size + c] = VISITED
                result = self.result(current_state=1)
                if result == 1:
                    winning_cells.append((r, c))
                    winning_cells.append((r, self.board_size - 1 - c))
                    winning_cells.append((self.board_size - 1 - r, c))
                    winning_cells.append((self.board_size - 1 - r, self.board_size - 1 - c))
                    winning_cells.append((c, r))
                    winning_cells.append((self.board_size - 1 - c, r))
                    winning_cells.append((c, self.board_size - 1 - r))
                    winning_cells.append((self.board_size - 1 - c, self.board_size - 1 - r))
                self.board[r * self.board_size + c] = UNVISITED

        self.board[:] = original_board

        winning_cells = sorted(set(winning_cells))
        return winning_cells

time0 = time.time()
board_size = 5
piece = 'QUEEN'
B0 = Board(board_size, piece)
print(B0.winning_initial_positions())
print(time.time() - time0)