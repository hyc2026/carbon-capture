import copy
import numpy as np
import random
from abc import abstractmethod

# from envs.obs_parser import ObservationParser
from zerosum_env.envs.carbon.helpers import (Board, Cell, Collector, Planter,
                                             Point, RecrtCenter,
                                             RecrtCenterAction, WorkerAction)

from typing import Tuple, Dict, List

BaseActions = [None,
               RecrtCenterAction.RECCOLLECTOR,
               RecrtCenterAction.RECPLANTER]

WorkerActions = [None,
                 WorkerAction.UP,
                 WorkerAction.RIGHT,
                 WorkerAction.DOWN,
                 WorkerAction.LEFT]

TOP_CARBON_CONTAIN_FOR_PLANTER = 5   # 這個要小一些
PREEMPT_BONUS = 5000
TREE_PLANTED_LIMIT = 15  # 在场树的数量大于该值，则停止种树, 這個最好大一些，因爲要往敵方基地中樞
CARBON_SATISFY_COLLECTOR = 2000  # COLLECTOR_CARBON_CARRY_LIMIT * 捕碳员数量
CARBON_SATISFY_PLANTER = 100   # 要小一些，在敌方基地附近种树
COLLECTOR_CARBON_CARRY_LIMIT = 150    # 捕碳员携带碳上限
CELL_CARBON_REMAIN = 50  # 单元当前含碳量小于该值，捕碳员会去其他位置补碳
GO_HOME_STEPS = 290  # 这个步数之后智能体开始返回基地


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
        self.worker_target = dict()
        self._move_action = None
        self.move_action = copy.deepcopy(WorkerAction.moves())

    @ staticmethod
    def _location_to_cell(position, cell_map: Dict[Point, Cell]) -> Cell:
        for _point, _cell in cell_map.items():
            if _point == position:
                return _cell
        assert False, "未能找到该位置对应的cell"

    def _find_optimal_submap(self, sub_map_gen, map_carbon_location: Dict[Point, Cell], limitation):

        """计算子图的碳总量, 返回cell和排序后的碳的字典"""

        max_carbon = 0
        max_carbon_map = None

        for _sub_map in sub_map_gen:
            _map_cell = [self._location_to_cell(_pos, map_carbon_location) for _pos in _sub_map]

            _map_dict = dict(sorted({_c: _c.carbon for _c in _map_cell}.items(), key=lambda x: x[1], reverse=True))
            _carbon_sum = sum(list(_map_dict.values()))

            if _carbon_sum > max_carbon:
                max_carbon = _carbon_sum
                max_carbon_map = _map_dict

            if _carbon_sum >= limitation:
                return _map_dict

        return max_carbon_map

    @ staticmethod
    def _minimum_distance(point_1, point_2):
        abs_distance = abs(point_1 - point_2)
        cross_distance = min(point_1, point_2) + (15 - max(point_1, point_2))
        return min(abs_distance, cross_distance)

    def _calculate_distance(self, current_position, target_position):
        """计算真实距离，计算跨图距离，取两者最小值"""
        x_distance = self._minimum_distance(current_position[0], target_position[0])
        y_distance = self._minimum_distance(current_position[1], target_position[1])

        return x_distance + y_distance

    # def _check_surround_validity(self, random_move, worker) -> bool:
    #     raise NotImplementedError

    @ abstractmethod
    def action(self, **kwargs):
        pass


def old_to_new_position(old_position, move_point: Point):
    new_position = old_position + move_point
    new_position = str(new_position).replace("15", "0")
    new_position = Point(*eval(new_position.replace("-1", "14")))
    return new_position

# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------


def calculate_carbon_contain(map_carbon_cell: Dict) -> Dict:
    """遍历地图上每一个位置，附近碳最多的位置按从多到少进行排序"""
    carbon_contain_dict = dict()  # 用来存储地图上每一个位置周围4个位置当前的碳含量, {(0, 0): 32}
    for _loc, cell in map_carbon_cell.items():

        valid_loc = [(_loc[0], _loc[1] - 1),
                     (_loc[0] - 1, _loc[1]),
                     (_loc[0] + 1, _loc[1]),
                     (_loc[0], _loc[1] + 1)]  # 四个位置，按行遍历时从小到大

        forced_pos_valid_loc = str(valid_loc).replace('-1', '14')  # 因为棋盘大小是 15 * 15
        forced_pos_valid_loc = [Point(*_i) for _i in eval(forced_pos_valid_loc.replace('15', '0'))]

        filter_cell = \
            [_c for _, _c in map_carbon_cell.items() if getattr(_c, "position", Point(-100, -100)) in forced_pos_valid_loc]

        # assert len(filter_cell) == 4  # 当时子图时，可能不是四

        carbon_contain_dict[cell] = sum([_fc.carbon for _fc in filter_cell])

    map_carbon_sum_sorted = dict(sorted(carbon_contain_dict.items(), key=lambda x: x[1], reverse=True))

    return map_carbon_sum_sorted


class SubMap:
    def __init__(self, recrt_position):

        """建立一个可以不断变大的子图，以便于寻找最优子图方便捕碳员转运碳"""

        self.init_sub_map = [Point(*[recrt_position[0], recrt_position[1] - 1]),
                             Point(*[recrt_position[0] - 1, recrt_position[1]]),
                             Point(*[recrt_position[0] + 1, recrt_position[1]]),
                             Point(*[recrt_position[0], recrt_position[1] + 1])]

    def get_larger_sub_map(self):
        step = 0
        while True:
            new_sub_map = list()
            for _p in self.init_sub_map:
                for _m in WorkerAction.moves():
                    new_action = old_to_new_position(_p, _m.to_point())
                    new_sub_map.append(new_action)

            new_sub_map = list(set(new_sub_map))
            self.init_sub_map.extend(new_sub_map)
            yield self.init_sub_map

            step += 1
            if step > 14:   # 子图考虑步数的上限
                break
        return self.init_sub_map


class CollectorAct(AgentBase):
    def __init__(self):
        super().__init__()
        self.store_new_position_cell = set()

    def _dynamic_map_allocated(self, map_carbon_location, recrtCenter: Point):

        self.sub_map = SubMap(recrtCenter)

        return self._find_optimal_submap(sub_map_gen=self.sub_map.get_larger_sub_map(),
                                         map_carbon_location=map_carbon_location,
                                         limitation=CARBON_SATISFY_COLLECTOR)

    def _target_plan(self, collector: Collector, carbon_sort_dict: Dict, ours_info, oppo_info) -> List:

        """Planter 我们的种树员，carbon_sort_dict Cell碳含量从多到少排序"""

        collector_position = collector.position
        # 取碳排量最高的前n

        """在这个范围之内，有树先抢树，没树了再种树，种树也要有个上限，种的树到达一定数量之后，开始保护树"""

        # 计算planter和他的相对距离，并且结合该位置四周碳的含量，得到一个总的得分
        planned_target = [Point(*_v.position) for _k, _v in self.worker_target.items()]   # 计划的位置
        target_preference_score = -1e9
        max_score_cell_dict = dict()

        for _cell, _carbon_sum in carbon_sort_dict.items():

            planter_to_cell_distance = self._calculate_distance(collector_position,
                                                                _cell.position)  # 我们希望这个距离越小越好

            if _cell.position not in planned_target:  # TODO：这样会减少回家的次数，造成回家的碳车减少

                target_preference_score = np.log(1 / (planter_to_cell_distance + 1e-9)) + _carbon_sum * 10000  # 以碳量为主，碳量一样再看距离

            max_score_cell_dict[_cell] = target_preference_score

        max_score_cell_dict = \
            sorted(max_score_cell_dict.items(), key=lambda x: x[1], reverse=True)

        return max_score_cell_dict[0][0]

    @ staticmethod
    def _check_surround_enemy(collector: Collector, oppo_id: int):

        check_cell_list = [collector.cell.up, collector.cell.up.up, collector.cell.up.left, collector.cell.up.right,
                           collector.cell.down, collector.cell.down.down, collector.cell.down.left, collector.cell.down.right,
                           collector.cell.right, collector.cell.right.right, collector.cell.right.up, collector.cell.right.down,
                           collector.cell.left, collector.cell.left.left, collector.cell.left.up, collector.cell.left.down]

        for _cell in check_cell_list:
            if (_cell.planter is not None) and (int(_cell.planter.id.split('-')[1]) == oppo_id):
                return _cell.planter  # 只撞planter
            elif (_cell.collector is not None) and (int(_cell.collector.id.split('-')[1]) == oppo_id) and \
                    (_cell.collector.carbon > collector.carbon):
                return _cell.collector

    def _check_surround_validity(self, move: WorkerAction, worker, move_action_dict) -> bool:
        move_name = move.name
        if move_name == 'UP':
            # 需要看前方四个位置有没有Agent
            check_cell_list = [worker.cell.up, worker.cell.up.up, worker.cell.up.left, worker.cell.up.right]
        elif move_name == 'DOWN':
            check_cell_list = [worker.cell.down, worker.cell.down.down, worker.cell.down.left, worker.cell.down.right]
        elif move_name == 'RIGHT':
            check_cell_list = [worker.cell.right, worker.cell.right.right, worker.cell.right.up, worker.cell.right.down]
        elif move_name == 'LEFT':
            check_cell_list = [worker.cell.left, worker.cell.left.left, worker.cell.left.up, worker.cell.left.down]
        else:
            raise NotImplementedError

        # 如果有agent，是地方agent就躲，是我方agent，看他下一步行动会不会冲突

        check_list = list()
        for _c in check_cell_list:
            if _c.worker is None:
                check_list.append(True)
            else:
                _c_worker_id = int(_c.worker_id.split('-')[1])
                if _c_worker_id == worker.player_id:  # 我方，看他下一步的位置是否和我方冲突
                    get_move_from_dict = move_action_dict.get(_c.worker_id, None)
                    move_to_point = WorkerAction[get_move_from_dict].to_point() if get_move_from_dict is not None else Point(0, 0)
                    cell_position = old_to_new_position(_c.position, move_to_point)
                    worker_position = old_to_new_position(worker.position, move.to_point())

                    if cell_position == worker_position:
                        check_list.append(False)
                    else:
                        check_list.append(True)
                else:  # 躲避地方单位
                    # check_list.append(False)  # 躲

                    if (_c.collector is not None) and (_c.collector.carbon < worker.carbon):  # 是敌方捕碳员，且携带碳量小于我方捕碳员
                        check_list.append(False)  # 躲
                    else:
                        check_list.append(True)

        return all(check_list)

    def _random_move_with_check(self, current_move_candidate: list, worker, current_move, move_action_dict):

        """检查随机移动是否合法，不合法的话重新挑选"""

        if len(current_move_candidate) == 0:
            return None

        random_move = random.choice(current_move_candidate)

        # 随机选择一个动作
        if self._check_surround_validity(random_move, worker, move_action_dict):
            return [current_move, random_move][np.random.binomial(n=1, p=1)]   # p是1发生的概率

        self._move_action.remove(random_move)
        return self._random_move_with_check(current_move_candidate, worker, current_move, move_action_dict)

    def action(self, ours_info, oppo_info, map_carbon_location, step):

        move_action_dict = dict()
        # map_all_carbon = sum([_v.carbon for _k, _v in map_carbon_location.items()])

        # optimal_carbon_map = {_cell: _cell.carbon for _pos, _cell in map_carbon_location.items()}   # 全图

        optimal_carbon_map = self._dynamic_map_allocated(map_carbon_location=map_carbon_location,
                                                         recrtCenter=ours_info.recrtCenters[0].position)  # 子图

        ours_collector_id_list = [_cell.id for _cell in ours_info.collectors]

        collector_last_two_id = ours_collector_id_list[:-2]

        for collector in ours_info.collectors:

            if collector.id not in self.worker_target:   # 说明他还没有策略，要为其分配新的策略

                target_cell = self._target_plan(collector=collector,
                                                carbon_sort_dict=optimal_carbon_map,
                                                ours_info=ours_info,
                                                oppo_info=oppo_info[0])  # 返回这个智能体要去哪里的一个字典

                self.worker_target[collector.id] = target_cell

            # # 撞人策略, 仅刚招募的两个人去追敌方种树员
            if collector.id in collector_last_two_id:
                is_enemy = self._check_surround_enemy(collector, oppo_id=oppo_info[0].id)
                if is_enemy:
                    self.worker_target[collector.id] = is_enemy

            # target_position_list = [_v.position for _k, _v in self.worker_target.items()]

            if collector.position == self.worker_target[collector.id].position:

                if (collector.carbon >= COLLECTOR_CARBON_CARRY_LIMIT):    # 收集够了,并且没有人要回基地，那么就回基地
                    # and (collector.position not in target_position_list)

                    self.worker_target[collector.id] = ours_info.recrtCenters[0]

                else:   # 换目标接着收集

                    if collector.cell.carbon >= CELL_CARBON_REMAIN:  # 如果当前位置的cell的含碳量大于规定的阈值，那就接着采碳
                        pass
                    else:
                        target_cell = self._target_plan(collector=collector,
                                                        carbon_sort_dict=optimal_carbon_map,
                                                        ours_info=ours_info,
                                                        oppo_info=oppo_info[0])  # 返回这个智能体要去哪里的一个字典

                        self.worker_target[collector.id] = target_cell

            else:  # 没有执行完接着执行， TODO: 家附近有人，先不回家，换路线

                old_position = collector.position
                target_position = self.worker_target[collector.id].position
                old_distance = self._calculate_distance(old_position, target_position)
                old_position_cell = self._location_to_cell(old_position, map_carbon_location)

                for move in WorkerAction.moves():
                    new_position = old_to_new_position(old_position, move.to_point())
                    new_distance = self._calculate_distance(new_position, target_position)

                    if new_distance < old_distance:

                        if old_position_cell.carbon < CELL_CARBON_REMAIN:
                            if self._check_surround_validity(move, collector, move_action_dict):
                                move_action_dict[collector.id] = move.name
                            else:   # 随机移动，不要静止不动或反向移动，否则当我方多个智能体相遇会卡主
                                self._move_action = copy.deepcopy(self.move_action)
                                self._move_action.remove(move)
                                move_check = self._random_move_with_check(current_move_candidate=self._move_action,
                                                                          worker=collector,
                                                                          current_move=move,
                                                                          move_action_dict=move_action_dict)

                                if move_check is not None:  # 并且当前位置的含碳量大于给定阈值
                                    move_action_dict[collector.id] = move_check.name
                        else:
                            pass  # 停下收集碳

                        if step >= GO_HOME_STEPS:
                            move_action_dict[collector.id] = move.name

            if step >= GO_HOME_STEPS:
                self.worker_target[collector.id] = ours_info.recrtCenters[0]

        return move_action_dict


class PlanterAct(AgentBase):
    def __init__(self):
        super().__init__()
        self.workaction = WorkerAction

    def _dynamic_map_allocated(self, map_carbon_location, recrtCenter: Point):

        self.sub_map = SubMap(recrtCenter)

        return self._find_optimal_submap(sub_map_gen=self.sub_map.get_larger_sub_map(),
                                         map_carbon_location=map_carbon_location,
                                         limitation=CARBON_SATISFY_PLANTER)

    def _check_surround_validity(self, move: WorkerAction, worker, move_action_dict) -> bool:
        move_name = move.name
        if move_name == 'UP':
            # 需要看前方四个位置有没有Agent
            check_cell_list = [worker.cell.up, worker.cell.up.up, worker.cell.up.left, worker.cell.up.right]
        elif move_name == 'DOWN':
            check_cell_list = [worker.cell.down, worker.cell.down.down, worker.cell.down.left, worker.cell.down.right]
        elif move_name == 'RIGHT':
            check_cell_list = [worker.cell.right, worker.cell.right.right, worker.cell.right.up, worker.cell.right.down]
        elif move_name == 'LEFT':
            check_cell_list = [worker.cell.left, worker.cell.left.left, worker.cell.left.up, worker.cell.left.down]
        else:
            raise NotImplementedError

        # 如果有agent，是地方agent就躲，是我方agent，看他下一步行动会不会冲突

        check_list = list()
        for _c in check_cell_list:
            if _c.worker is None:
                check_list.append(True)
            else:
                _c_worker_id = int(_c.worker_id.split('-')[1])
                if _c_worker_id == worker.player_id:  # 我方，看他下一步的位置是否和我方冲突
                    get_move_from_dict = move_action_dict.get(_c.worker_id, None)
                    move_to_point = WorkerAction[get_move_from_dict].to_point() if get_move_from_dict is not None else Point(0, 0)
                    cell_position = old_to_new_position(_c.position, move_to_point)
                    worker_position = old_to_new_position(worker.position, move.to_point())

                    if cell_position == worker_position:
                        check_list.append(False)
                    else:
                        check_list.append(True)
                else:
                    # check_list.append(False)  # 躲

                    if _c.planter is not None:  # 是敌方种树员
                        check_list.append(True)  # BU DUO
                    else:
                        check_list.append(False)

        return all(check_list)

    def _random_move_with_check(self, current_move_candidate: list, worker, current_move, move_action_dict):

        """检查随机移动是否合法，不合法的话重新挑选"""

        if len(current_move_candidate) == 0:
            return None

        random_move = random.choice(current_move_candidate)

        # 随机选择一个动作
        if self._check_surround_validity(random_move, worker, move_action_dict):
            return [current_move, random_move][np.random.binomial(n=1, p=1)]   # p是1发生的概率

        self._move_action.remove(random_move)
        return self._random_move_with_check(current_move_candidate, worker, current_move, move_action_dict)

    def _target_plan(self, planter: Planter, carbon_sort_dict: Dict, carbon_sort_dict_top_n,
                     ours_info, oppo_info) -> List:
        """Planter 我们的种树员，carbon_sort_dict Cell碳含量从多到少排序"""

        planter_position = planter.position
        # 取碳排量最高的前n

        """在这个范围之内，有树先抢树，没树了再种树，种树也要有个上限，种的树到达一定数量之后，开始保护树"""

        # 计算planter和他的相对距离，并且结合该位置四周碳的含量，得到一个总的得分
        planned_target = [Point(*_v.position) for _k, _v in self.worker_target.items()]   # 计划的位置
        tree_num_sum = len(ours_info.trees) + len(oppo_info.trees)  # 在场所有树的数量
        target_preference_score = -1e9
        max_score_cell_dict = dict()

        for _cell, _carbon_sum in carbon_sort_dict.items():

            planter_to_cell_distance = self._calculate_distance(planter_position,
                                                                _cell.position)  # 我们希望这个距离越小越好
            if _cell.position not in planned_target:

                if _cell.tree is None:

                    if (_cell in carbon_sort_dict_top_n) and \
                            (_cell.recrtCenter is None) and (tree_num_sum <= TREE_PLANTED_LIMIT):

                        target_preference_score = np.log(1 / (planter_to_cell_distance + 1e-9))  #log的 max: 20左右， _carbon_sum的max最大400, 3乘表示更看重carbon_sum

                else:

                    tree_player_id = _cell.tree.player_id

                    if tree_player_id == oppo_info.id:   # 是对方的树

                        target_preference_score = PREEMPT_BONUS + np.log(1 / (planter_to_cell_distance + 1e-9))   #   加一个大数，表示抢敌方树优先，抢敌方距离最近的树优先

                    if (tree_player_id == ours_info.id) and (tree_num_sum > TREE_PLANTED_LIMIT):  # 是我方的树，那就开始保护我方的树

                        target_preference_score = np.log(1 / (planter_to_cell_distance + 1e-9))

            max_score_cell_dict[_cell] = target_preference_score

        max_score_cell_dict = \
            sorted(max_score_cell_dict.items(), key=lambda x: x[1], reverse=True)

        return max_score_cell_dict[0][0]

    def action(self, ours_info, oppo_info, map_carbon_location, steps):

        move_action_dict = dict()

        """需要知道本方当前位置信息，敵方当前位置信息，地图上的碳的分布"""
        # 如果planter信息是空的，则无需执行任何操作
        # if ours_info.planters == []:
        #     return None, None

        # map_carbon_cell = map_carbon_location

        ours_planter_id_list = [_cell.id for _cell in ours_info.planters]

        planter_last_two_id = ours_planter_id_list[:-2]

        for planter in ours_info.planters:

            if (len(ours_info.planters) > 2) and (planter.id in planter_last_two_id):
                optimal_carbon_map = self._dynamic_map_allocated(
                    map_carbon_location=map_carbon_location,
                    recrtCenter=oppo_info[0].recrtCenters[0].position
                )  # 子图
                map_carbon_cell = {_cell.position: _cell for _cell, _ in optimal_carbon_map.items()}

            else:

                # optimal_carbon_map = self._dynamic_map_allocated(map_carbon_location=map_carbon_location,
                #                                                  recrtCenter=ours_info.recrtCenters[0].position)  # 子图
                #
                # map_carbon_cell = {_cell.position: _cell for _cell, _ in optimal_carbon_map.items()}
                map_carbon_cell = map_carbon_location  # 全图

            carbon_sort_dict = calculate_carbon_contain(map_carbon_cell)  # 每一次move都先计算一次附近碳多少的分布
            carbon_sort_dict_top_n = \
                [_k for _i, (_k, _) in enumerate(carbon_sort_dict.items()) if
                 _i < TOP_CARBON_CONTAIN_FOR_PLANTER]  # 只选取含碳量top_n的cell来进行计算。

            if planter.id not in self.worker_target:   # 说明他还没有策略，要为其分配新的策略

                target_cell = self._target_plan(planter=planter,
                                                carbon_sort_dict=carbon_sort_dict, carbon_sort_dict_top_n=carbon_sort_dict_top_n,
                                                ours_info=ours_info, oppo_info=oppo_info[0])  # 返回这个智能体要去哪里的一个字典

                self.worker_target[planter.id] = target_cell

            if planter.position == self.worker_target[planter.id].position:   # 当前的planter目标执行完了,

                self.worker_target.pop(planter.id, None)

            else:  # 没有执行完接着执行

                old_position = planter.position
                target_position = self.worker_target[planter.id].position
                old_distance = self._calculate_distance(old_position, target_position)

                for move in WorkerAction.moves():
                    new_position = old_to_new_position(old_position, move.to_point())
                    new_distance = self._calculate_distance(new_position, target_position)

                    if new_distance < old_distance:
                        if self._check_surround_validity(move, planter, move_action_dict):
                            move_action_dict[planter.id] = move.name
                        else:   # 随机移动，不要静止不动或反向移动，否则当我方多个智能体相遇会卡主
                            self._move_action = copy.deepcopy(self.move_action)
                            self._move_action.remove(move)
                            move_check = self._random_move_with_check(current_move_candidate=self._move_action,
                                                                      worker=planter,
                                                                      current_move=move,
                                                                      move_action_dict=move_action_dict)
                            if move_check is not None:
                                move_action_dict[planter.id] = move_check.name
        return move_action_dict


class RecruiterAct(AgentBase):
    def __init__(self):
        super().__init__()

    def _check_surround_validity(self, random_move, worker):
        pass

    def action(self, ours_info, oppo_info, steps):
        store_dict = dict()
        if (ours_info.recrtCenters[0].cell.worker is None) and (steps < GO_HOME_STEPS):   # # 确定基地位置没有任何Agent才能进行招募
            # if len(ours_info.collectors) < 10:
            #     store_dict[ours_info.recrtCenters[0].id] = RecrtCenterAction.RECCOLLECTOR.name

            if ours_info.cash < 100:
                if len(ours_info.planters) < 1:
                    store_dict[ours_info.recrtCenters[0].id] = RecrtCenterAction.RECPLANTER.name
                elif len(ours_info.collectors) < 1:
                    store_dict[ours_info.recrtCenters[0].id] = RecrtCenterAction.RECCOLLECTOR.name
            else:

                if 0 <= steps < 150:
                    if len(ours_info.collectors) < 3:
                        store_dict[ours_info.recrtCenters[0].id] = RecrtCenterAction.RECCOLLECTOR.name
                    elif len(ours_info.planters) < 7:
                        store_dict[ours_info.recrtCenters[0].id] = RecrtCenterAction.RECPLANTER.name
                else:
                    if len(ours_info.collectors) < 7:
                        store_dict[ours_info.recrtCenters[0].id] = RecrtCenterAction.RECCOLLECTOR.name
                    elif len(ours_info.planters) < 3:
                        store_dict[ours_info.recrtCenters[0].id] = RecrtCenterAction.RECPLANTER.name




        return store_dict


class PlanningPolicy(BasePolicy):
    def __init__(self, ):
        super().__init__()
        self.collector = CollectorAct()
        self.planter = PlanterAct()
        self.recruiter = RecruiterAct()

    def take_action(self, current_obs: Board, previous_obs: Board) -> Dict:
        overall_dict = dict()

        ours, oppo = current_obs.current_player, current_obs.opponents

        # 基地先做决策是否招募
        recruit_dict = self.recruiter.action(
            ours_info=ours,
            oppo_info=oppo,
            steps=current_obs.step,
        )

        if recruit_dict is not None:
            overall_dict.update(recruit_dict)

        # 种树员做决策去哪里种树
        planter_dict = self.planter.action(
            ours_info=ours,
            oppo_info=oppo,
            map_carbon_location=current_obs.cells,
            steps=current_obs.step
        )

        overall_dict.update(planter_dict)

        collection_dict = self.collector.action(
            ours_info=ours,
            oppo_info=oppo,
            map_carbon_location=current_obs.cells,
            step=current_obs.step,
        )

        if collection_dict is not None:
            overall_dict.update(collection_dict)

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





