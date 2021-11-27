import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
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
    point = cur_cell.position
    pass
    
policy=PlanningPolicy()
logs = []
env = make("carbon", configuration={"randomSeed":1}, logs=logs)
env.reset()
print(carbon2map(env.states[0]['observation']['carbon']))


