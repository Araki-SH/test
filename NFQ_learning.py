# coding: utf-8

import numpy as np
class NFQ:
    def __init__(self):
        self.input_layer = 2
        self.hide_layer = 2
        self.output_layer = 4
        self.input_p = np.array([1,2])#np.zeros(self.input_layer)
        self.w_in2hi = np.array([[1,0],[0,1]])#np.random.rand(self.input_layer,self.hide_layer)
        self.hide_p = np.array(self.hide_layer)
        self.w_hi2ou = np.array([[1,0,0,0],[0,0,0,1]])#np.random.rand(self.hide_layer,self.output_layer)
        self.output_p = np.array(self.output_layer)
        self.ganma = 0.1
    
    def Relu(self,x):
        return np.maximum(x, 0)
    
    def step(self,x):
        return np.array(x > 0, dtype=np.int)
    
    def layer_in2hi(self):
        self.hide_p = self.Relu(np.dot(self.input_p, self.w_in2hi))

    def layer_hi2ou(self):
        self.output_p =  self.Relu(np.dot(self.hide_p, self.w_hi2ou))

    def layer_predict(self):
        self.layer_in2hi()
        self.layer_hi2ou()
        return self.output_p

    def layer_update_hi2in(self):
        y = self.step(self.output_p)
        self.w_hi2ou += np.dot(self.ganma * y,self.input_p)

    def layer_update_ou2hi(self):
        y = self.step(self.output_p)
        self.w_hi2ou += self.ganma * y

    def layer_update(self):
        self.layer_update_ou2hi()
        self.layer_update_hi2in()
        print(self.w_hi2ou)
        print(self.w_in2hi)

nfq = NFQ()

print(nfq.layer_predict())
print(nfq.layer_update())

class NFQ_learning:
    """単純なQ-learning"""

    def __init__(self, maze):
        # 迷路
        self.maze = maze
        # 状態（=エージェントの位置）は迷路の盤面数
        row, col = self.maze.board_size()
        self.num_state = row * col
        # 行動の数は上下左右の4種類
        self.num_action = 4
        # Qは 状態数 x 行動数
        self.Q = np.zeros((self.num_state, self.num_action))
        # 現在の状態
        self.state = self.get_state()

    def select_best_action(self):
        """評価値の最も高い行動を探す"""
        return self.Q[self.state, :].argmax()

    def select_action(self, random_rate=0.01):
        """一定の確率で、ベストでない動きをする"""
        if np.random.rand() < random_rate:
            return np.random.randint(self.num_action)
        return self.select_best_action()

    def get_state(self):
        """状態を取得"""
        row, col = self.maze.board_size()
        x, y = self.maze.get_position()
        return x/row,y/col
        #,x * col + y

    def reward(self):
        """報酬"""
        return 0 if self.maze.is_goal() else -1

    def from_start(self):
        """スタートからやり直す"""
        self.maze.reset()
        self.state = self.get_state()

    def step(self, learning_rate, discount_rate, random_rate):
        # 行動の選択。ベストアクションとは限らない。
        action = self.select_action(random_rate)
        # 選択された行動に従い動く。ただし、壁がある場合は無視される
        self.maze.move(action)
        # 移動後の状態を取得
        next_state = self.get_state()
        # ベストアクションを選択
        next_action = self.select_best_action()
        # Q[s][a] += 学習率 * ( 報酬 + 割引率 * ( max_{s'} Q[s'][a'] ) - Q[s][a] )
        self.Q[self.state][action] += learning_rate * (
            self.reward()
            + discount_rate * self.Q[next_state][next_action]
            - self.Q[self.state][action]
        )
        # 移動後の状態を現在の状態に記録
        self.state = next_state
