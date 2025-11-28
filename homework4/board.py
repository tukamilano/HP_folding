from bitarray import bitarray
from collections import deque
import heapq
import itertools

d4 = [
    # 0: identity
    [0, 1, 2, 3,
     4, 5, 6, 7,
     8, 9,10,11,
     12,13,14,15],

    # 1: rotate90
    [12, 8, 4, 0,
     13, 9, 5, 1,
     14,10, 6, 2,
     15,11, 7, 3],

    # 2: rotate180
    [15,14,13,12,
     11,10, 9, 8,
     7, 6, 5, 4,
     3, 2, 1, 0],

    # 3: rotate270
    [3, 7,11,15,
     2, 6,10,14,
     1, 5, 9,13,
     0, 4, 8,12],

    # 4: flip_horizontal
    [3, 2, 1, 0,
     7, 6, 5, 4,
     11,10, 9, 8,
     15,14,13,12],

    # 5: flip_vertical
    [12,13,14,15,
     8, 9,10,11,
     4, 5, 6, 7,
     0, 1, 2, 3],

    # 6: flip_main_diag
    [0, 4, 8,12,
     1, 5, 9,13,
     2, 6,10,14,
     3, 7,11,15],

    # 7: flip_anti_diag
    [15,11, 7, 3,
     14,10, 6, 2,
     13, 9, 5, 1,
     12, 8, 4, 0],
]

def canonicalize(board):
    return min(board[p] for p in d4)

def update(board, flip_pos, symmetry):
    new_board = board ^ flip_pos
    if symmetry:
        new_board = canonicalize(new_board)
    return new_board

def is_terminate(board):
    return not board.any()

def BFS(init_board, flip_candidates, symmetry=False):
    if symmetry:
        init_board = canonicalize(init_board)

    visited_board_set = set()
    init_key = init_board.to01()
    visited_board_set.add(init_key)

    # (board, parent, flip_index) を保存
    candidates = deque([(init_board, None, None)])
    search_num = 0
    parent_map = {}  # board_key -> (parent_key, flip_index)

    while candidates:
        board, parent_key, flip_idx = candidates.popleft() 
        search_num += 1
        board_key = board.to01()
        if parent_key is not None:
            parent_map[board_key] = (parent_key, flip_idx)
        
        new_board_list = []

        for flip_idx, flip_pos in enumerate(flip_candidates):
            new_board = update(board, flip_pos, symmetry)

            if is_terminate(new_board):
                # 経路を構築
                path = []
                current_key = board_key
                while current_key in parent_map:
                    current_key, flip_idx = parent_map[current_key]
                    path.append(flip_idx)
                path.reverse()
                path.append(flip_idx)  # 最後の操作を追加
                return search_num, path

            key = new_board.to01()
            if key not in visited_board_set:
                visited_board_set.add(key)
                new_board_list.append((new_board, board_key, flip_idx))

        new_board_list.sort(key=lambda x: x[0].count())
        candidates.extend(new_board_list)
    return None, None

def DFS(init_board, flip_candidates, symmetry=False):
    if symmetry:
        init_board = canonicalize(init_board)

    visited_board_set = set()
    init_key = init_board.to01()
    visited_board_set.add(init_key)

    # (board, parent, flip_index) を保存
    candidates = deque([(init_board, None, None)])
    search_num = 0
    parent_map = {}  # board_key -> (parent_key, flip_index)

    while candidates:
        board, parent_key, flip_idx = candidates.pop() 
        search_num += 1
        board_key = board.to01()
        if parent_key is not None:
            parent_map[board_key] = (parent_key, flip_idx)
        
        new_board_list = []

        for flip_idx, flip_pos in enumerate(flip_candidates):
            new_board = update(board, flip_pos, symmetry)

            if is_terminate(new_board):
                # 経路を構築
                path = []
                current_key = board_key
                while current_key in parent_map:
                    current_key, flip_idx = parent_map[current_key]
                    path.append(flip_idx)
                path.reverse()
                path.append(flip_idx)  # 最後の操作を追加
                return search_num, path

            key = new_board.to01()
            if key not in visited_board_set:
                visited_board_set.add(key)
                new_board_list.append((new_board, board_key, flip_idx))

        new_board_list.sort(key=lambda x: -x[0].count())
        candidates.extend(new_board_list)
    return None, None

def admissible_A_star(init_board, flip_candidates, max_block_size, symmetry=False):
    if symmetry:
        init_board = canonicalize(init_board)

    searched = set()
    counter = itertools.count()
    candidate_queue = []

    h0 = init_board.count() / max_block_size
    heapq.heappush(candidate_queue, (h0, next(counter), (init_board, 0, None, None)))
    
    parent_map = {}  # board_key -> (parent_key, flip_index)
    search_num = 0

    while candidate_queue:
        _, _, (board, g, parent_key, flip_idx) = heapq.heappop(candidate_queue)
        key = board.to01()

        # すでに展開済みの状態なら skip
        if key in searched:
            continue

        searched.add(key)
        search_num += 1
        if parent_key is not None:
            parent_map[key] = (parent_key, flip_idx)

        for flip_idx, flip_pos in enumerate(flip_candidates):
            new_board = update(board, flip_pos, symmetry)
            new_key = new_board.to01()

            if is_terminate(new_board):
                # 経路を構築
                path = []
                current_key = key
                while current_key in parent_map:
                    current_key, flip_idx = parent_map[current_key]
                    path.append(flip_idx)
                path.reverse()
                path.append(flip_idx)  # 最後の操作を追加
                return search_num, path

            if new_key not in searched:
                h = new_board.count() / max_block_size
                f = g + 1 + h 
                heapq.heappush(candidate_queue,
                                (f, next(counter), (new_board, g+1, key, flip_idx)))

    return None, None


init_board1 = bitarray('1001000000001001')
init_board2 = bitarray('0001001001001000')
init_board3 = bitarray('0100101001011010')
init_board4 = bitarray('0101100000011010')

# Convert coordinates to bitarray index: (i, j) -> i*4 + j
def coords_to_bitarray(coords):
    ba = bitarray(16)
    ba.setall(0)
    for i, j in coords:
        ba[i*4 + j] = 1
    return ba

def max_block_size(flip_candidates):
    a = 0
    for flip_candidate in flip_candidates:
        if a < flip_candidate.count():
            a = flip_candidate.count()
    return a
'''
flip_candidates = (
    [coords_to_bitarray([(i, j), (i+1, j), (i, j+1)])
     for i in range(3) for j in range(3)]
    +
    [coords_to_bitarray([(i, j), (i+1, j), (i+1, j+1)])
     for i in range(3) for j in range(3)]
    +
    [coords_to_bitarray([(i, j), (i, j+1), (i+1, j+1)])
     for i in range(3) for j in range(3)]
    +
    [coords_to_bitarray([(i+1, j), (i, j+1), (i+1, j+1)])
     for i in range(3) for j in range(3)]
)
'''
flip_candidates = (
    [coords_to_bitarray([(i, j), (i+1, j), (i+2, j)])
     for i in range(2) for j in range(4)] +
    [coords_to_bitarray([(i, j), (i, j+1), (i, j+2)])
     for i in range(4) for j in range(2)]   
)

flip_candidates_max_block = max_block_size(flip_candidates)
#flip_candidates_max_block = 1

init_boards = [init_board1, init_board2, init_board3, init_board4]

for i in range(len(init_boards)):  
    search_num, path = admissible_A_star(init_boards[i], flip_candidates, flip_candidates_max_block, symmetry=False)
    print(f"Board {i+1}:")
    print(f"  探索数: {search_num}")
    if path is not None:
        print(f"  探索経路 (flip操作のインデックス): {path}")
        print(f"  経路の長さ: {len(path)}")
    else:
        print(f"  解が見つかりませんでした")
    print()


