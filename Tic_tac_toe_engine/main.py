class ThreeByThree():
    def __init__(self, board=None):
        if board is None:
            board = [0] * 9
        self.board = board
        self.memo = {}
        self.optimal_order = [8,4,7,3,9,2,6,1,5]
    
    def result(self, current_state=1, alpha=-float('inf'), beta=float('inf')):
        memoise_key = (current_state, tuple(self.board))
        if memoise_key in self.memo:
            return self.memo[memoise_key]
        
        winner = self.won_by()
        if winner != 0:
            self.memo[memoise_key] = winner
            return winner
        
        spaces = [i for (i, v) in enumerate(self.board) if v == 0]

        spaces.sort(
            key=lambda x: self.optimal_order[x] if x < len(self.optimal_order) else -1,
            reverse=True
        )
        if len(spaces) > 0:
            pruned = False
            if current_state == 1: # Maximizing player
                current_value = -float('inf')
                for i in spaces:
                    self.board[i] = current_state
                    value = self.result(-current_state, alpha, beta)
                    self.board[i] = 0
                    current_value = max(current_value, value)
                    alpha = max(alpha, current_value)
                    if beta <= alpha:
                        pruned = True
                        break # Beta cut-off
            else: # Minimizing player
                current_value = float('inf')
                for i in spaces:
                    self.board[i] = current_state
                    value = self.result(-current_state, alpha, beta)
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
        
        return 0 # Draw if no spaces left

    def find_best_move(self, player=1):
        best_move = -1
        alpha = -float('inf')
        beta = float('inf')
        
        # 空きマスを取得
        spaces = [i for (i, v) in enumerate(self.board) if v == 0]
        # 最適な順序に基づいてソート
        spaces.sort(
            key=lambda x: self.optimal_order[x] if x < len(self.optimal_order) else -1,
            reverse=True
        )
        if player == 1:  # 最大化プレイヤー
            best_value = -float('inf')
            for i in spaces:
                self.board[i] = player
                value = self.result(-player, alpha, beta)
                self.board[i] = 0
                
                if value > best_value:
                    best_value = value
                    best_move = i
                    
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break  # ベータカット
                    
        else:  # 最小化プレイヤー
            best_value = float('inf')
            for i in spaces:
                self.board[i] = player
                value = self.result(-player, alpha, beta)
                self.board[i] = 0
                
                if value < best_value:
                    best_value = value
                    best_move = i
                    
                beta = min(beta, best_value)
                if beta <= alpha:
                    break  # アルファカット
        
        return best_move

    def won_by(self):
        for i in range(3):
            if self.board[3*i] == self.board[3*i+1] == self.board[3*i+2]:
                if self.board[3*i] != 0:
                    return self.board[3*i]
            if self.board[i] == self.board[3+i] == self.board[6+i]:
                if self.board[i] != 0:
                    return self.board[i]
        if (self.board[0] == self.board[4] == self.board[8]
            or self.board[2] == self.board[4] == self.board[6]):
            if self.board[4] != 0:
                return self.board[4]
        return 0

engine = input()
if engine == 'X':
    current_state = 1
else:
    current_state = -1


board = []
for _ in range(3):
    row = input()
    for r in row:
        if r == 'X':
            board.append(1)
        elif r == 'O':
            board.append(-1)
        else:
            board.append(0)

tictactoe = ThreeByThree(board = board)
if (tictactoe.won_by() == 0) and (0 in board):
    s = tictactoe.find_best_move(player=current_state)
    board[s] = current_state
board = ['X' if x == 1 else x for x in board]
board = ['O' if x == -1 else x for x in board]
board = ['.' if x == 0 else x for x in board]

print("".join(board[0:3])) 
print("".join(board[3:6])) 
print("".join(board[6:9])) 