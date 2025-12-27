from bitarray import bitarray

d4 = [
    # 0: identity
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    # 1: rotate90
    [7, 2, 10, 5, 0, 8, 3, 11, 6, 1, 9, 4],
    # 2: rotate180
    [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    # 3: rotate270
    [4, 9, 1, 6, 11, 3, 8, 0, 5, 10, 2, 7],
    # 4: flip_0
    [1, 0, 4, 3, 2, 6, 5, 9, 8, 7, 11, 10],
    # 5: flip_1
    [2, 7, 0, 5, 10, 3, 8, 1, 6, 11, 4, 9],
    # 6: flip_2
    [10, 11, 7, 8, 9, 5, 6, 2, 3, 4, 0, 1],
    # 7: flip_3
    [9, 4, 11, 6, 1, 8, 3, 10, 5, 0, 7, 2]
]

box = [
    [0, 2, 3, 5],
    [1, 3, 4, 6],
    [5, 7, 8, 10],
    [6, 8, 9, 11]
]

def canonicalize(board):
    return min(board[p] for p in d4)

class Dots_and_Boxes():
    def __init__(self, board, score):
        # boardは[bitarray, bitarray]を使う
        # scoreは[my_score, opp_score]を使う
        self.board = board
        self.score = score
        self.memo = {}
    
    def result(self, current_state=1, alpha=-float('inf'), beta=float('inf')):
        memoise_key = (current_state, tuple(canonicalize(self.board[0]), canonicalize(self.board[1])))
        if memoise_key in self.memo:
            return self.memo[memoise_key]
        
        winner = self.won_by()
        if winner != 0:
            self.memo[memoise_key] = winner
            return winner
        
        spaces = [] # 二つのbitarrayの中でどちらも0になっていないもの

        if len(spaces) > 0:
            pruned = False
            if current_state == 1: # Maximaizing player
                current_value = -float('inf')
                for i in spaces:
                    pass

        
    def update(self):
        # 探索する前にとりあえず繰り返しbox_reachとなるboxの残りの一辺を埋める(scoreを出さないと)
        pass

    def box_reach(self):
        box_reach_list = []
        for b in box:
            # self.board[0]とself.board[1]で三辺が埋まっている時に残りの辺のインデックスをbox_reach_listに追加する
            pass
        return box_reach_list
    
    def won_by(self):
        if self.score[0] > 2:
            return 1
        elif self.score[1] > 2:
            return -1
        else:
            return 0

    
