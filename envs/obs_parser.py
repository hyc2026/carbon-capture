from typing import Dict, Tuple
import numpy as np

from zerosum_env import make, evaluate
# from zerosum_env.envs.carbon.helpers import *
from zerosum_env.envs.carbon.helpers import RecrtCenterAction, WorkerAction, Board


BaseActions = [None,
               RecrtCenterAction.RECCOLLECTOR,
               RecrtCenterAction.RECPLANTER]

WorkerActions = [None,
                 WorkerAction.UP,
                 WorkerAction.RIGHT,
                 WorkerAction.DOWN,
                 WorkerAction.LEFT]

WorkerDirections = np.stack([np.array((0, 0)),
                             np.array((0, 1)),
                             np.array((1, 0)),
                             np.array((0, -1)),
                             np.array((-1, 0))])  # 与WorkerActions相对应


BaseActionsByName = {action.name: action for action in BaseActions if action is not None}

WorkerActionsByName = {action.name: action for action in WorkerActions if action is not None}


def one_hot_np(value: int, num_cls: int):
    ret = np.zeros(num_cls)
    ret[value] = 1
    return ret

class ObservationParser:
    """
    ObservationParser class is used to parse observation dict data and converted to observation tensor for training.

    The features are included as follows:

    """
    def __init__(self, grid_size=15,
                 max_step=300,
                 max_cell_carbon=100,
                 tree_lifespan=50,
                 action_space=5):
        self.grid_size = grid_size
        self.max_step = max_step
        self.max_cell_carbon = max_cell_carbon
        self.tree_lifespan = tree_lifespan
        self.action_space = action_space

    def distance_feature(self, x, y) -> np.ndarray:
        """ 获取输入位置到地图任意位置需要的最小步数
        Calculate the minimum distance from current position to other positions on grid.
        :param x: position x
        :param y: position y
        :return distance_map: 2d-array, the value in the grid indicates the minimum distance form position (x, y) to
            current position.
        """
        distance_y = (np.ones((self.grid_size, self.grid_size)) * np.arange(self.grid_size)).astype(np.float32)
        distance_x = distance_y.T
        delta_distance_x = abs(distance_x - x)
        delta_distance_y = abs(distance_y - y)
        offset_distance_x = self.grid_size - delta_distance_x
        offset_distance_y = self.grid_size - delta_distance_y
        distance_x = np.where(delta_distance_x < offset_distance_x,
                            delta_distance_x, offset_distance_x)
        distance_y = np.where(delta_distance_y < offset_distance_y,
                            delta_distance_y, offset_distance_y)
        distance_map = distance_x + distance_y

        return distance_map
    

    def obs_transform(self, current_obs: Board):
        available_actions = {}
        my_player_id = current_obs.current_player_id
        
        
        # 1 碳的分布 15x15
        carbon_dis = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # 1x15x15 捕碳员位置
        collector_feature = np.zeros_like(carbon_dis, dtype=np.float32)  # me: +1; opponent: -1
        # 1x15x15 种树员位置
        planter_feature = np.zeros_like(carbon_dis, dtype=np.float32)  # me: +1; opponent: -1
        # 1x15x15 捕碳员携带的碳的数量
        worker_carbon_feature = np.zeros_like(carbon_dis, dtype=np.float32)
        # 1x15x15 树的分布，绝对值表示树的年龄
        tree_feature = np.zeros_like(carbon_dis, dtype=np.float32)  # trees, me: +; opponent: -.
        
        for point, cell in current_obs.cells.items():
            if cell.carbon > 0:
                carbon_dis[point.x, point.y] = cell.carbon
        # 2 当前轮数
        step_now = current_obs.step
        
        # 3 敌我基地位置
        my_base = None
        op_base = None
        for base_id, base in current_obs.recrtCenters.items():
            # 判断当前基地是否是我方基地
            is_myself = base.player_id == my_player_id
            if is_myself:
                my_base = (base.position.x, base.position.y)
            else:
                op_base = (base.position.x, base.position.y)
                
        # 4 敌我现金数量
        my_cash, op_cash = current_obs.current_player.cash, current_obs.opponents[0].cash
        
        # 5 敌我捕碳员信息
        my_collectors = []
        op_collectors = []
        my_planters = []
        op_planters = []
        
        for worker_id, worker in current_obs.workers.items():
            is_myself = worker.player_id == my_player_id

            worker_x, worker_y = worker.position.x, worker.position.y
            if is_myself:
                if worker.is_collector:
                    my_collectors.append((worker.id, worker_x, worker_y, worker.carbon))
                    collector_feature[worker_x, worker_y] = 1.0
                    worker_carbon_feature[worker_x, worker_y] = worker.carbon
                else:
                    my_planters.append(((worker.id, worker_x, worker_y, worker.carbon)))
                    planter_feature[worker_x, worker_y] = 1.0
            else:
                if worker.is_collector:
                    op_collectors.append((worker.id, worker_x, worker_y, worker.carbon))
                    collector_feature[worker_x, worker_y] = -1.0
                else:
                    op_planters.append(((worker.id, worker_x, worker_y, worker.carbon)))
                    planter_feature[worker_x, worker_y] = -1.0
        
        # 6 敌我树的分布
        my_trees = []
        op_trees = []
        for tree in current_obs.trees.values():
            if tree.player_id == my_player_id:
                my_trees.append(tree.id, tree.position.x, tree.position.y, tree.age)
                tree_feature[tree.position.x, tree.position.y] = tree.age
            else:
                op_trees.append(tree.id, tree.position.x, tree.position.y, tree.age)
                tree_feature[tree.position.x, tree.position.y] = - tree.age
        
        
        my_info = {
            "base": my_base,
            "cash": my_cash,
            "collectors": my_collectors,
            "planters": my_planters,
            "trees": my_trees
        }
        
        op_info = {
            "base": op_base,
            "cash": op_cash,
            "collectors": op_collectors,
            "planters": op_planters,
            "trees": op_trees
        }
        map_info = {
            "carbon": carbon_dis,
            "tree_feature": tree_feature,
            "collector_feature": collector_feature,
            "planter_feature": planter_feature,
            "worker_carbon_feature": worker_carbon_feature
        }
                
        return map_info, step_now, my_info, op_info
    
if __name__ == "__main__":
    obs_parser = ObservationParser()
    env = make("carbon", {"seed": 1234}, debug=True)
    # 查看环境的各项参数
    config = env.configuration

    # 确定地图选手数(只能是2)
    num_agent = 2

    # 获取自身初始观测状态并查看
    obs = env.reset(num_agent)[0].observation

    # 将obs转换为Board类从而更好获取信息
    board = Board(obs, config)
    map_info, step_now, my_info, op_info = obs_parser.obs_transform(board)
    print("map info:\n", map_info)
    print("Step:", step_now)
    print("My Info:\n", my_info)
    print("Op Info:\n", op_info)