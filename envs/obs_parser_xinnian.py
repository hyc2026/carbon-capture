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


# 核心代码：observation
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
    
    
    def obs_transform(self, current_obs: Board, previous_obs: Board = None) -> Tuple[Dict, Dict, Dict]:
        """
        通过前后两帧的原始观测状态值, 计算状态空间特征, agent, dones信息以及agent可用的动作空间.

        特征维度包含:
            1) vector_feature：(一维特征, dim: 8)
                Step feature: range [0, 1], dim 1, 游戏轮次
                my_cash: range [-1, 1], dim 1, 玩家金额
                opponent_cash: range [-1, 1], dim 1, 对手金额
                agent_type: range [0, 1], dim 3, agent类型(one-hot)
                x: range [0, 1), dim 1, agent x轴位置坐标
                y: range [0, 1), dim 1, agent y轴位置坐标
            2) cnn_feature: (CNN特征, dim: 11x15x15)
                carbon_feature: range [0, 1], dim: 1x15x15, 地图碳含量分布
                base_feature: range [-1, 1], dim: 1x15x15, 转化中心位置分布(我方:+1, 对手:-1)
                collector_feature: range [-1, 1], dim: 1x15x15, 捕碳员位置分布(我方:+1, 对手:-1)
                planter_feature: range [-1, 1], dim: 1x15x15, 种树员位置分布(我方:+1, 对手:-1)
                worker_carbon_feature: range [-1, 1], dim: 1x15x15, 捕碳员携带CO2量分布(我方:>=0, 对手:<=0)
                tree_feature: [-1, 1], dim: 1x15x15, 树分布,绝对值表示树龄(我方:>=0, 对手:<=0)
                action_feature:[0, 1], dim: 5x15x15, 上一轮次动作分布(one-hot)
                my_base_distance_feature: [0, 1], dim: 1x15x15, 我方转化中心在地图上与各点位的最短距离分布
                distance_features: [0, 1], dim: 1x15x15, 当前agent距离地图上各点位的最短距离分布

        :param current_obs: 当前轮次原始的状态
        :param previous_obs: 前一轮次原始的状态 (default: None)
        :return local_obs: (Dict[str, np.ndarray]) 当前选手每个agent的observation特征 (vector特征+CNN特征展成的一维特征)
        :return dones: (Dict[str, bool]) 当前选手每个agent的done标识, True-agent已死亡, False-agent尚存活
        :return available_actions: (Dict[str, np.ndarray]) 标识当前选手每个agent的动作维度是否可用, 1表示该动作可用,
            0表示动作不可用
        """
        # 加入agent上一轮次的动作
        agent_cmds = self._guess_previous_actions(previous_obs, current_obs)
        previous_action = {k: v.value if v is not None else 0 for k, v in agent_cmds.items()}

        available_actions = {}
        my_player_id = current_obs.current_player_id

        carbon_feature = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for point, cell in current_obs.cells.items():
            if cell.carbon > 0:
                carbon_feature[point.x, point.y] = cell.carbon / self.max_cell_carbon

        step_feature = current_obs.step / (self.max_step - 1)
        base_feature = np.zeros_like(carbon_feature, dtype=np.float32)  # me: +1; opponent: -1
        collector_feature = np.zeros_like(carbon_feature, dtype=np.float32)  # me: +1; opponent: -1
        planter_feature = np.zeros_like(carbon_feature, dtype=np.float32)  # me: +1; opponent: -1
        worker_carbon_feature = np.zeros_like(carbon_feature, dtype=np.float32)
        tree_feature = np.zeros_like(carbon_feature, dtype=np.float32)  # trees, me: +; opponent: -.
        action_feature = np.zeros((self.grid_size, self.grid_size, self.action_space), dtype=np.float32)

        my_base_distance_feature = None
        distance_features = {}

        my_cash, opponent_cash = current_obs.current_player.cash, current_obs.opponents[0].cash
        for base_id, base in current_obs.recrtCenters.items():
            is_myself = base.player_id == my_player_id

            base_x, base_y = base.position.x, base.position.y

            base_feature[base_x, base_y] = 1.0 if is_myself else -1.0
            base_distance_feature = self._distance_feature(base_x, base_y) / (self.grid_size - 1)
            distance_features[base_id] = base_distance_feature

            action_feature[base_x, base_y] = one_hot_np(previous_action.get(base_id, 0), self.action_space)
            if is_myself:
                available_actions[base_id] = np.array([1, 1, 1, 0, 0])

                my_base_distance_feature = distance_features[base_id]

        for worker_id, worker in current_obs.workers.items():
            is_myself = worker.player_id == my_player_id

            available_actions[worker_id] = np.array([1, 1, 1, 1, 1])

            worker_x, worker_y = worker.position.x, worker.position.y
            distance_features[worker_id] = self._distance_feature(worker_x, worker_y) / (self.grid_size - 1)

            action_feature[worker_x, worker_y] = one_hot_np(previous_action.get(worker_id, 0), self.action_space)

            if worker.is_collector:
                collector_feature[worker_x, worker_y] = 1.0 if is_myself else -1.0
            else:
                planter_feature[worker_x, worker_y] = 1.0 if is_myself else -1.0

            worker_carbon_feature[worker_x, worker_y] = worker.carbon
        worker_carbon_feature = np.clip(worker_carbon_feature / self.max_cell_carbon / 2, -1, 1)

        for tree in current_obs.trees.values():
            tree_feature[tree.position.x, tree.position.y] = tree.age if tree.player_id == my_player_id else -tree.age
        tree_feature /= self.tree_lifespan

        global_vector_feature = np.stack([step_feature,
                                          np.clip(my_cash / 2000., -1., 1.),
                                          np.clip(opponent_cash / 2000., -1., 1.),
                                          ]).astype(np.float32)
        global_cnn_feature = np.stack([carbon_feature,
                                       base_feature,
                                       collector_feature,
                                       planter_feature,
                                       worker_carbon_feature,
                                       tree_feature,
                                       *action_feature.transpose(2, 0, 1),  # dim: 5 x 15 x 15
                                       ])  # dim: 11 x 15 x 15

        dones = {}
        local_obs = {}
        previous_worker_ids = set() if previous_obs is None else set(previous_obs.current_player.worker_ids)
        worker_ids = set(current_obs.current_player.worker_ids)
        new_worker_ids, death_worker_ids = worker_ids - previous_worker_ids, previous_worker_ids - worker_ids
        obs = previous_obs if previous_obs is not None else current_obs
        total_agents = obs.current_player.recrtCenters + \
                       obs.current_player.workers + \
                       [current_obs.workers[id_] for id_ in new_worker_ids]  # 基地 + prev_workers + new_workers
        for my_agent in total_agents:
            if my_agent.id in death_worker_ids:  # 死亡的agent, 直接赋值为0
                local_obs[my_agent.id] = np.zeros(self.observation_dim, dtype=np.float32)
                available_actions[my_agent.id] = np.array([1, 1, 1, 1, 1])
                dones[my_agent.id] = True
            else:  # 未死亡的agent
                cnn_feature = np.stack([*global_cnn_feature,
                                        my_base_distance_feature,
                                        distance_features[my_agent.id],
                                        ])  # dim: 2925 (13 x 15 x 15)
                if not hasattr(my_agent, 'is_collector'):  # 转化中心
                    agent_type = [1, 0, 0]
                else:  # 工人
                    agent_type = [0, int(my_agent.is_collector), int(my_agent.is_planter)]
                vector_feature = np.stack([*global_vector_feature,
                                           *agent_type,
                                           my_agent.position.x / self.grid_size,
                                           my_agent.position.y / self.grid_size,
                                           ]).astype(np.float32)  # dim: 8
                local_obs[my_agent.id] = np.concatenate([vector_feature, cnn_feature.reshape(-1)])
                dones[my_agent.id] = False

        return local_obs, dones, available_actions
    
    # 简化规则，只根据当前情况判断
    def obs_transform_new(self, current_obs: Board):
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