import random

def _pos(candidate):
    # 初期位置は(0, 0)、初期方向は右（x軸正方向）
    current_pos = [0, 0]
    # 方向ベクトル: [dx, dy]
    # 右(1,0), 上(0,1), 左(-1,0), 下(0,-1)
    direction = [1, 0]  # 右方向から開始
    pos_list = [tuple(current_pos.copy())]
    
    for r in candidate:
        # 角度に応じて方向を回転
        if r == -1:  # 90度左回転
            # (dx, dy) -> (-dy, dx)
            direction = [-direction[1], direction[0]]
        elif r == 1:  # 90度右回転
            # (dx, dy) -> (dy, -dx)
            direction = [direction[1], -direction[0]]
        
        # 方向に従って次の位置に移動
        current_pos = [current_pos[0] + direction[0], current_pos[1] + direction[1]]
        pos_list.append(tuple(current_pos.copy()))
    
    return pos_list


def pos(candidate):
    """
    Public wrapper so external modules (e.g., visualize scripts) can reuse the
    lattice reconstruction while keeping internal name compatibility.
    """
    return _pos(candidate)

def get_abcd_discrete(L):
    # 1. 区間の長さ w を決める
    # 長さ w (1からL-1まで) の「重み」は (L-w)^2
    # w=1 が一番出やすく、w=L-1 が一番出にくい
    weights = [(L - w)**2 for w in range(1, L)]
    w = random.choices(range(1, L), weights=weights, k=1)[0]
    
    # 2. 決まった長さ w で、場所 a, c をランダムに選ぶ
    # a の範囲は 0 から L-w-1 まで
    max_pos = L - w - 1
    a = random.randint(0, max_pos)
    c = random.randint(0, max_pos)
    
    return a, a + w, c, c + w

def detect_lethal(candidate):
    candidate_pos = _pos(candidate)
    return len(candidate_pos) != len(set(candidate_pos))

def score(candidate, sequence):
    assert len(candidate) == len(sequence) - 1
    assert not detect_lethal(candidate)

    # 候補の角度列から座標リストを構築
    candidate_pos = _pos(candidate)
    assert len(candidate_pos) == len(sequence)

    # 位置 → アミノ酸の辞書を構築
    pos_dict = {}
    for i in range(len(candidate_pos)):
        pos_dict[candidate_pos[i]] = sequence[i]

    # 隣接するH-H結合を探索
    HH_bond = 0
    neighbor_deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for position, acid in pos_dict.items():
        if acid != 'H':
            continue

        px, py = position
        for dx, dy in neighbor_deltas:
            neighbor = (px + dx, py + dy)
            if pos_dict.get(neighbor) == 'H':
                HH_bond += 1

    assert HH_bond % 2 == 0
    HH_bond //= 2

    # 連続するH-H鎖のペナルティ
    HH_chain = sum(
        1
        for i in range(len(sequence) - 1)
        if sequence[i] == 'H' and sequence[i + 1] == 'H'
    )

    return HH_bond - HH_chain