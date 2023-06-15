#!/usr/bin/env python
# coding: utf-8

import time
from Maze import Maze
from NFQ_learning import NFQ_learning
from Sarsa_learning import Sarsa_learning
import matplotlib.pyplot  as plt
import numpy as np

# 迷路
# W: Wall, S : Start, G: Goal
'''BOARD = [
    "WWWWWWWWWWWWWWWWWWWWW",
    "WG    W     W       W",
    "W W WWWWW W W W WWWWW",
    "W W       W   W W   W",
    "W W W WWW WWWWW W WWW",
    "W W W       W       W",
    "W W W WWW WWW W WWWWW",
    "W             W    SW",
    "WWWWWWWWWWWWWWWWWWWWW",
]#'''
BOARD = [
    "WWWWWWW",
    "WS W  W",
    "W  W  W",
    "WW    W",
    "W  WW W",
    "W   WGW",
    "WWWWWWW",
]#'''
# 何回ゴールするか
EPISODE_MAX = 1000
# ゴールまでの打ち切りステップ数
STEP_MAX = 3000
# 学習率
LEARNING_RATE = 0.01
# 割引率
DISCOUNT_RATE = 0.95
# 描画スピード
SLEEP_TIME = 0.001

maze = Maze(BOARD)
maze = Maze(BOARD)
nfq_learn = NFQ_learning(maze)
sarsa_learning = Sarsa_learning(maze)

all = []
for episode in range(EPISODE_MAX):
    step = 0
    nfq_learn.from_start()
    #sarsa_learning.from_start()
    # ランダムに最善でない行動を取る
    random_rate = 0.01 + 0.9 / (1 + episode)
    while not maze.is_goal() and step < STEP_MAX:
        # エージェントの1ステップ(行動、評価値の更新)
        nfq_learn.step(LEARNING_RATE, DISCOUNT_RATE, random_rate)
        #sarsa_learning.step(LEARNING_RATE, DISCOUNT_RATE, random_rate)
        # 迷路描画
        maze.draw()
        step += 1
        time.sleep(SLEEP_TIME)
    print("\x1b[K")  # 行末までをクリア
    print(f"episode : {episode} step : {step} ")
    all.append(step)

x = range(0, len(all))
plt.plot(x,all)
plt.show()
