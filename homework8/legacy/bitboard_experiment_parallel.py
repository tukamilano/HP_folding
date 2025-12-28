from bitarray import bitarray
import time
import threading
from concurrent.futures import ThreadPoolExecutor

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
    def __init__(self, board_size, piece, memo=None, lock=None, piece_moves=None):
        # store board as numpy array for shape/fast updates
        self.board_size = board_size
        self.board = bitarray(self.board_size * self.board_size)
        self.board.setall(UNVISITED)
        self.piece = piece
        self.memo = memo if memo is not None else {}
        self.lock = lock if lock is not None else threading.RLock()
        self.piece_move_dict = piece_moves if piece_moves is not None else piece_move_dict(self.piece, self.board_size)
        self.piece_pos = None

    def result(self, current_state=1, alpha=-float('inf'), beta=float('inf')):
        memoise_key = (self.piece_pos, tuple(self.board))
        with self.lock:
            cached = self.memo.get(memoise_key)
        if cached is not None:
            return cached
        
        piece_move_candid = self.piece_move_dict[self.piece_pos]
        legal_moves = piece_move_candid & self.board  # 1 where reachable and unvisited

        if not any(legal_moves):
            winner = -current_state
            with self.lock:
                self.memo[memoise_key] = winner
            return winner

        pruned = False
        if current_state == 1: # Maximizing player
            current_value = -float('inf')
            move_indices = [i for i, b in enumerate(legal_moves) if b]
            for i in move_indices:
                prev_pos = self.piece_pos
                self.piece_pos = (i // self.board_size, i % self.board_size)
                self.board[i] = VISITED
                next_state = -current_state
                value = self.result(next_state, alpha, beta)
                self.board[i] = UNVISITED
                self.piece_pos = prev_pos
                current_value = max(current_value, value)
                alpha = max(alpha, current_value)
                if beta <= alpha:
                    pruned = True
                    break # Beta cut-off
        else: # Minimizing player
            current_value = float('inf')
            move_indices = [i for i, b in enumerate(legal_moves) if b]
            for i in move_indices:
                prev_pos = self.piece_pos
                self.board[i] = VISITED
                self.piece_pos = (i // self.board_size, i % self.board_size)
                next_state = -current_state
                value = self.result(next_state, alpha, beta)
                self.board[i] = UNVISITED
                self.piece_pos = prev_pos
                current_value = min(current_value, value)
                beta = min(beta, current_value)
                if beta <= alpha:
                    pruned = True
                    break # Alpha cut-off
        
        # Memoize only when the value is exact (no pruning); otherwise
        # we might store a lower/upper bound and reuse it incorrectly.
        if not pruned:
            with self.lock:
                self.memo[memoise_key] = current_value
        return current_value

    def winning_initial_positions(self, max_workers=None):
        """Return all starting squares (no pieces on board) where the first player wins.

        We reuse the shared memo table. We temporarily clear the board to all
        UNVISITED, place the piece on each cell, and evaluate keeping the same
        player to move (placement does not swap turns). The board is restored
        to its original contents after the search.
        """
        original_board = self.board.copy()
        self.board[:] = UNVISITED

        tasks = []
        k = (self.board_size + 1) // 2
        for r in range(k):
            for c in range(r, k):
                tasks.append((r, c))

        def eval_start(rc):
            r, c = rc
            worker = Board(self.board_size, self.piece, memo=self.memo, lock=self.lock, piece_moves=self.piece_move_dict)
            worker.board[:] = UNVISITED
            idx = r * self.board_size + c
            worker.piece_pos = (r, c)
            worker.board[idx] = VISITED
            res = worker.result(current_state=1)
            if res != 1:
                return []
            n = self.board_size
            return [
                (r, c),
                (r, n - 1 - c),
                (n - 1 - r, c),
                (n - 1 - r, n - 1 - c),
                (c, r),
                (n - 1 - c, r),
                (c, n - 1 - r),
                (n - 1 - c, n - 1 - r),
            ]

        winning_cells = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for cells in ex.map(eval_start, tasks):
                winning_cells.extend(cells)

        self.board[:] = original_board

        winning_cells = sorted(set(winning_cells))
        return winning_cells

time0 = time.time()
board_size = 5
piece = 'QUEEN'
B0 = Board(board_size, piece)
print(B0.winning_initial_positions(max_workers=None))
print(time.time() - time0)