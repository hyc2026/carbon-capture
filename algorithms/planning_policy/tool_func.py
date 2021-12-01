import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import math
import copy
from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, OrderedDict, Tuple

from algorithms.base_policy import BasePolicy
from envs.obs_parser_xinnian import (BaseActions, ObservationParser,
                                     WorkerActions)
from zerosum_env import make, evaluate
from zerosum_env.envs.carbon.helpers import *
from planning_policy import PlanningPolicy

def carbon2map(carbon: List) -> List:
    map = []
    for x in range(15):
        for y in range(15):
            if y == 0:
                map.append([])
            map[x].append(carbon[y + x * 15])
    return map
        
def get_surrounded_cells(board: Board, cur_cell: Cell) -> List[Cell]:
    cell_list = []
    point = cur_cell.position
    cell_list.append(board.cells[Point(point.x - 1 if point.x > 0 else 14, point.y)])
    cell_list.append(board.cells[Point(point.x + 1 if point.x < 14 else 0, point.y)])
    cell_list.append(board.cells[Point(point.x, point.y - 1 if point.y > 0 else 14)])
    cell_list.append(board.cells[Point(point.x, point.y + 1 if point.y < 14 else 0)])
    return cell_list
    
def move(board: Board, cur_cell: Cell, direction: WorkerAction, step: int) -> Cell:
    point = cur_cell.position
    if direction == WorkerAction.UP:
        return board.cells[Point(point.x, point.y - step if point.y - step >= 0 else point.y - step + 15)]
    elif  direction == WorkerAction.DOWN:
        return board.cells[Point(point.x, point.y + step if point.y + step < 15 else (point.y + step) % 15)]
    elif  direction == WorkerAction.RIGHT:
        return board.cells[Point(point.x + step if point.x + step < 15 else (point.x + step) % 15, point.y)]
    elif  direction == WorkerAction.LEFT:
        return board.cells[Point(point.x - step if point.x - step >= 0 else point.x - step + 15, point.y)]
    else:
        return cur_cell

def get_distance(cell1: Cell, cell2: Cell) -> int:
    return math.abs(cell1.position.x, cell2.position.x) + math.abs(cell1.position.y, cell2.position.y)

def get_nearest_enemy(cur_cell: Cell, enemy_list: List[Cell]) -> Cell:
    if len(enemy_list) == 0:
        return None
    max_dis = 50
    r_cell = enemy_list[0]
    for i in enemy_list:
        dis = get_distance(cur_cell, i)
        if dis < max_dis:
            max_dis = dis
            r_cell = i
    return r_cell

def build_vitual_map():
    pass

def get_cell_carbon_after_n_step(board: Board, position: Point, n: int) -> float:
    # 危险区域中如果有树会对position造成影响
    danger_zone = []
    x_left = position.x - 1 if position.x > 0 else 14
    x_right = position.x + 1 if position.x < 14 else 0
    y_up = position.y - 1 if position.y > 0 else 14
    y_down = position.y + 1 if position.y < 14 else 0
    # danger_zone.append(Point(x_left, y_up))
    danger_zone.append(Point(position.x, y_up))
    # danger_zone.append(Point(x_right, y_up))
    danger_zone.append(Point(x_left, position.y))
    danger_zone.append(Point(x_right, position.y))
    # danger_zone.append(Point(x_left, y_down))
    danger_zone.append(Point(position.x, y_down))
    # danger_zone.append(Point(x_right, y_down))
    
    start = 0
    c = board.cells[position].carbon
    if n == 0:
        return c
        # position的位置有树，直接从树暴毙之后开始算
    if board.cells[position].tree is not None:
        start = 50 - board.cells[position].tree.age + 1
        if start <= n:
            c = 30.0
        else:
            return 0
            
    # 对于每一回合，分别计算有几颗树在吸position的碳
    for i in range(start, n):
        tree_count = 0
        for p in danger_zone:
            tree = board.cells[p].tree
            # 树在危险区域内
            if tree is not None:
                # i回合后树还没暴毙
                if tree.age + i <= 50:
                    tree_count += 1
        if tree_count == 0:
            c = c * (1.05)
        else:
            c = c * (1 - 0.0375 * tree_count)
            # c = c * (1 - 0.0375) ** tree_count
        c = min(c, 100)
    return c  

def get_distance_map(board: Board) -> Dict[(Tuple, Tuple), int]:

    # 初始化参数
    # 一个极大值
    INF = int(1e9)
    # 地图边长
    LENGTH = 15
    # 地图格子数
    SIZE = 225
    # 初始化距离地图
    dist = [[INF for i in range(SIZE)] for j in range(SIZE)]
    # 危险位置
    dangerPos = []
    # 返回答案
    ans = {}

    # 两个辅助函数：坐标和序号相互转换
    def xy2pos(x: int, y: int) -> int:
        x = max(x, 0)
        x = min(x, LENGTH-1)
        y = max(y, 0)
        y = min(y, LENGTH-1)
        return y * LENGTH + x

    def pos2xy(pos: int) -> [int, int]:
        x = (pos + 1) % LENGTH - 1
        y = (pos + 1) // LENGTH
        if x < 0:
            x += LENGTH
            y -= 1
        return x, y

    # 初始化环境：将每一格上下左右挨着的4格的距离设为1
    for i in range(SIZE):
        dist[i][i] = 0
        if i - LENGTH >= 0:
            dist[i][i - LENGTH] = 1
        if not (i + 1) % LENGTH == 1:
            dist[i][i - 1] = 1
        if i + LENGTH < SIZE:
            dist[i][i + LENGTH] = 1
        if not (i + 1) % LENGTH == 0:
            dist[i][i + 1] = 1

    # 记录危险区域
    for tree in board.trees.values():
        if tree.player_id != board.current_player_id:
            x = tree.position.x
            y = tree.position.y
            dangerPos.append(xy2pos(x-1, y-1))
            dangerPos.append(xy2pos(x-1, y))
            dangerPos.append(xy2pos(x-1, y+1))
            dangerPos.append(xy2pos(x, y-1))
            dangerPos.append(xy2pos(x, y))
            dangerPos.append(xy2pos(x, y+1))
            dangerPos.append(xy2pos(x+1, y-1))
            dangerPos.append(xy2pos(x+1, y))
            dangerPos.append(xy2pos(x+1, y+1))

    # 将该区域设为不可通行
    for pos in dangerPos:
        for i in range(SIZE):
            dist[i][pos] = -1
            dist[pos][i] = -1

    # 计算距离
    for k in range(SIZE):
        for i in range(SIZE):
            for j in range(SIZE):
                if dist[i][k] != -1 and dist[k][j] != -1:
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

    # 按格式返回
    for i in range(SIZE):
        for j in range(SIZE):
            ans[(pos2xy(i)), (pos2xy(j))] = dist[i][j] if dist[i][j] != -1 else INF

    return ans



policy=PlanningPolicy()
logs = []
env = make("carbon", configuration={"randomSeed":1}, logs=logs)
env.reset()

