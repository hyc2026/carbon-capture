import copy
import numpy as np

import random
from abc import abstractmethod

# from envs.obs_parser import ObservationParser
from zerosum_env.envs.carbon.helpers import (Board, Cell, Collector, Planter,
                                             Point, RecrtCenter,
                                             RecrtCenterAction, WorkerAction)

from typing import Tuple, Dict, List


# TODO: 大问题： 任务基地闪烁


BaseActions = [None,
               RecrtCenterAction.RECCOLLECTOR,
               RecrtCenterAction.RECPLANTER]

WorkerActions = [None,
                 WorkerAction.UP,
                 WorkerAction.RIGHT,
                 WorkerAction.DOWN,
                 WorkerAction.LEFT]

TOP_CARBON_CONTAIN = 5


class BasePolicy:
    """
    Base policy class that wraps actor and critic models to calculate actions and value for training and evaluating.
    """
    def __init__(self):
        pass

    def policy_reset(self, episode: int, n_episodes: int):
        """
        Policy Reset at the beginning of the new episode.
        :param episode: (int) current episode
        :param n_episodes: (int) number of total episodes
        """
        pass

    def can_sample_trajectory(self) -> bool:
        """
        Specifies whether the policy's actions output and values output can be collected for training or not
            (default False).
        :return: True means the policy's trajectory data can be collected for training, otherwise False.
        """
        return False

    def get_actions(self, observation, available_actions=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute actions predictions for the given inputs.
        :param observation:  (np.ndarray) local agent inputs to the actor.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)

        :return actions: (np.ndarray) actions to take.
        :return action_log_probs: (np.ndarray) log probabilities of chosen actions.
        """
        raise NotImplementedError("not implemented")

    def state_dict(self):
        """
        Returns a whole state of models and optimizers.
        :return:
            dict:
                a dictionary containing a whole state of the module
        """
        pass

    def restore(self, model_dict, strict=True):
        """
        Restore models and optimizers from model_dict.

        :param model_dict: (dict) State dict of models and optimizers.
        :param strict: (bool, optional) whether to strictly enforce that the keys
        """
        pass

    @staticmethod
    def to_env_commands(policy_actions: Dict[str, int]) -> Dict[str, str]:
        """
        Actions output from policy convert to actions environment can accept.
        :param policy_actions: (Dict[str, int]) Policy actions which specifies the agent name and action value.
        :return env_commands: (Dict[str, str]) Commands environment can accept,
            which specifies the agent name and the command string (None is the stay command, no need to send!).
        """

        def agent_action(agent_name, command_value) -> str:
            # hack here, 判断是否为转化中心,然后返回各自的命令
            actions = BaseActions if 'recrtCenter' in agent_name else WorkerActions
            return actions[command_value].name if 0 < command_value < len(actions) else None

        env_commands = {}
        for agent_id, cmd_value in policy_actions.items():
            command = agent_action(agent_id, cmd_value)
            if command is not None:
                env_commands[agent_id] = command
        return env_commands


class AgentBase:
    def __init__(self):
        pass

    def move(self, **kwargs):
        """移动行为，需要移动到什么位置"""
        pass

    @ abstractmethod
    def action(self, **kwargs):
        pass


def calculate_carbon_contain(map_carbon_cell: Dict) -> Dict:
    """遍历地图上每一个位置，附近碳最多的位置按从多到少进行排序"""
    carbon_contain_dict = dict()  # 用来存储地图上每一个位置周围4个位置当前的碳含量, {(0, 0): 32}
    for _loc, cell in map_carbon_cell.items():

        valid_loc = [(_loc[0], _loc[1] - 1),
                     (_loc[0] - 1, _loc[1]),
                     (_loc[0] + 1, _loc[1]),
                     (_loc[0], _loc[1] + 1)]  # 四个位置，按行遍历时从小到大

        forced_pos_valid_loc = str(valid_loc).replace('-1', '14')  # 因为棋盘大小是 15 * 15
        forced_pos_valid_loc = eval(forced_pos_valid_loc.replace('15', '0'))

        filter_cell = \
            [_c for _, _c in map_carbon_cell.items() if getattr(_c, "position", (-100, -100)) in forced_pos_valid_loc]

        assert len(filter_cell) == 4  # 因为选取周围四个值来吸收碳

        carbon_contain_dict[cell] = sum([_fc.carbon for _fc in filter_cell])

    map_carbon_sum_sorted = dict(sorted(carbon_contain_dict.items(), key=lambda x: x[1], reverse=True))

    return map_carbon_sum_sorted


class WorkerAct(AgentBase):
    def __init__(self):
        super().__init__()

    def action(self):
        pass


class PlanterAct(AgentBase):
    def __init__(self):
        super().__init__()
        self.workaction = WorkerAction
        self.planter_target = dict()

    @ staticmethod
    def _minimum_distance(point_1, point_2):
        abs_distance = abs(point_1 - point_2)
        cross_distance = min(point_1, point_2) + (14 - max(point_1, point_2)) + 1 # cell坐标范围是 [0, 14]
        return min(abs_distance, cross_distance)

    def _calculate_distance(self, planter_position, current_position):
        """计算真实距离，计算跨图距离，取两者最小值"""
        x_distance = self._minimum_distance(planter_position[0], current_position[0])
        y_distance = self._minimum_distance(planter_position[1], current_position[1])

        return x_distance + y_distance

    def _target_plan(self, planter: Planter, carbon_sort_dict: Dict):
        """结合某一位置的碳的含量和距离"""
        # TODO：钱够不够是否考虑？
        planter_position = planter.position
        # 取碳排量最高的前n

        carbon_sort_dict_top_n = \
            {_v: _k for _i, (_v, _k) in enumerate(carbon_sort_dict.items()) if _i < TOP_CARBON_CONTAIN}  # 只选取含碳量top_n的cell来进行计算，拿全部的cell可能会比较耗时？
        # 计算planter和他的相对距离，并且结合该位置四周碳的含量，得到一个总的得分
        planned_target = [Point(*_v.position) for _k, _v in self.planter_target.items()]
        max_score, max_score_cell = -1e9, None
        for _cell, _carbon_sum in carbon_sort_dict_top_n.items():
            if (_cell.tree is None) and (_cell.position not in planned_target):  # 这个位置没有树，且这个位置不在其他智能体正在进行的plan中
                planter_to_cell_distance = self._calculate_distance(planter_position, _cell.position)  # 我们希望这个距离越小越好
                target_preference_score = 0 * _carbon_sum + np.log(1 / (planter_to_cell_distance + 1e-9))  # 不考虑碳总量只考虑距离 TODO: 这会导致中了很多树，导致后期花费很高

                if target_preference_score > max_score:
                    max_score = target_preference_score
                    max_score_cell = _cell

        if max_score_cell is None:  # 没有找到符合条件的最大得分的cell，随机选一个cell
            max_score_cell = random.choice(list(carbon_sort_dict_top_n))

        return max_score_cell

    def _check_surround_validity(self, move: WorkerAction, planter: Planter) -> bool:
        move = move.name
        if move == 'UP':
            # 需要看前方三个位置有没有Agent
            check_cell_list = [planter.cell.up, planter.cell.up.left, planter.cell.up.right]
        elif move == 'DOWN':
            check_cell_list = [planter.cell.down, planter.cell.down.left, planter.cell.down.right]
        elif move == 'RIGHT':
            check_cell_list = [planter.cell.right, planter.cell.right.up, planter.cell.right.down]
        elif move == 'LEFT':
            check_cell_list = [planter.cell.left, planter.cell.left.up, planter.cell.left.down]
        else:
            raise NotImplementedError

        return all([True if (_c.collector is None) and (_c.planter is None) else False for _c in check_cell_list])

    def move(self, ours_info, oppo_info, **kwargs):

        move_action_dict = dict()

        """需要知道本方当前位置信息，敵方当前位置信息，地图上的碳的分布"""
        # 如果planter信息是空的，则无需执行任何操作
        if ours_info.planters == []:
            return None

        map_carbon_cell = kwargs["map_carbon_location"]
        carbon_sort_dict = calculate_carbon_contain(map_carbon_cell)  # 每一次move都先计算一次附近碳多少的分布

        for planter in ours_info.planters:
            # 先给他随机初始化一个行动
            if planter.id not in self.planter_target:   # 说明他还没有策略，要为其分配新的策略
                target_cell = self._target_plan(planter, carbon_sort_dict)  # 返回这个智能体要去哪里的一个字典
                self.planter_target[planter.id] = target_cell  # 给它新的行动
            else:  # 说明他有策略，看策略是否执行完毕，执行完了移出字典，没有执行完接着执行
                if planter.position == self.planter_target[planter.id].position:
                    # 执行一次种树行动, TODO: 如果钱够就种树，钱不够不执行任何操作
                    # move_action_dict[planter.id] = None
                    # TODO: 这里不执行任何行动就表示种树了？
                    # 移出字典
                    self.planter_target.pop(planter.id)
                else:  # 没有执行完接着执行

                    old_position = planter.position
                    target_position = self.planter_target[planter.id].position
                    old_distance = self._calculate_distance(old_position, target_position)

                    for move in WorkerAction.moves():
                        new_position = old_position + move.to_point()
                        new_distance = self._calculate_distance(new_position, target_position)

                        if new_distance < old_distance:
                            if self._check_surround_validity(move, planter):
                                move_action_dict[planter.id] = move.name
                            else:   # 随机移动，不要静止不动或反向移动，否则当我方多个智能体相遇会卡主
                                if move.name == 'UP':
                                    move_action_dict[planter.id] = random.choice(["DOWN", "RIGHT", "LEFT"])
                                elif move.name == 'DOWN':
                                    move_action_dict[planter.id] = random.choice(["UP", "RIGHT", "LEFT"])
                                elif move.name == 'RIGHT':
                                    move_action_dict[planter.id] = random.choice(["UP", "DOWN", "LEFT"])
                                elif move.name == 'LEFT':
                                    move_action_dict[planter.id] = random.choice(["UP", "DOWN", "RIGHT"])

        return move_action_dict


class RecruiterAct(AgentBase):
    def __init__(self):
        super().__init__()

    def action(self, ours_info, **kwargs):
        store_dict = dict()
        if len(ours_info.planters) == 0:  # 招募1个种树人员
            store_dict[ours_info.recrtCenters[0].id] = RecrtCenterAction.RECPLANTER.name
        if ours_info.recrtCenters[0].cell.worker is None:   # # 确定基地位置没有任何Agent才能进行招募
            if len(ours_info.planters) == 1 and len(ours_info.trees) > 3:
                store_dict[ours_info.recrtCenters[0].id] = RecrtCenterAction.RECPLANTER.name
            # elif len(ours_info.planters) == 2 and len(ours_info.trees) > 10:
            #     store_dict[ours_info.recrtCenters[0].id] = RecrtCenterAction.RECPLANTER.name

        return store_dict


class PlanningPolicy(BasePolicy):
    def __init__(self, ):
        super().__init__()
        # self.worker = WorkerAct()
        self.planter = PlanterAct()
        self.recruiter = RecruiterAct()

    def take_action(self, current_obs: Board, previous_obs: Board) -> Dict:
        overall_dict = dict()

        ours, oppo = current_obs.current_player, current_obs.opponents

        # 基地先做决策是否招募
        recruit_dict = self.recruiter.action(
            ours_info=ours,
            map_carbon_location=current_obs.cells,
            step=current_obs.step,
        )

        # 这里要进行一个判断，确保基地位置没有智能体才能招募下一个

        if recruit_dict is not None:
            overall_dict.update(recruit_dict)

        # 种树员做决策去哪里种树
        planter_dict = self.planter.move(
            ours_info=ours,
            oppo_info=oppo,
            map_carbon_location=current_obs.cells,
            step=current_obs.step,
        )

        if planter_dict is not None:
            overall_dict.update(planter_dict)

        return overall_dict

        # 对于我方每一个捕碳员，不采取任何行动


class MyPolicy:

    def __init__(self):
        # self.obs_parser = ObservationParser()
        self.policy = PlanningPolicy()

    def take_action(self, observation, configuration):

        current_obs = Board(observation, configuration)
        previous_obs = self.previous_obs if current_obs.step > 0 else None

        overall_action = self.policy.take_action(current_obs=current_obs, previous_obs=previous_obs)
        # overall_action = self.to_env_commands(overall_action)

        # agent_obs_dict, dones, available_actions_dict = self.obs_parser.obs_transform(current_obs, previous_obs)
        self.previous_obs = copy.deepcopy(current_obs)

        return overall_action





