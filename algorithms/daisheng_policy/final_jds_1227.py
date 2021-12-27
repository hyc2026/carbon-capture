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

def cal_carbon_one_cell(cell, map_carbon_cell: Dict) -> float:
    _loc = cell.position
    valid_loc = [(_loc[0], _loc[1] - 1),
                 (_loc[0] - 1, _loc[1]),
                 (_loc[0] + 1, _loc[1]),
                 (_loc[0], _loc[1] + 1)]  # 四个位置，按行遍历时从小到大

    forced_pos_valid_loc = str(valid_loc).replace('-1', '14')  # 因为棋盘大小是 15 * 15
    forced_pos_valid_loc = eval(forced_pos_valid_loc.replace('15', '0'))

    filter_cell = \
        [_c for _, _c in map_carbon_cell.items() if getattr(_c, "position", (-100, -100)) in forced_pos_valid_loc]

    assert len(filter_cell) == 4  # 因为选取周围四个值来吸收碳

    return sum([_fc.carbon for _fc in filter_cell])

def calculate_carbon_contain_single(map_carbon_cell: Dict) -> Dict:
    """遍历地图上每一个位置，当前碳最多的位置按从多到少进行排序"""
    carbon_contain_dict = dict()  # 用来存储地图上每一个位置周围4个位置当前的碳含量, {(0, 0): 32}
    for _loc, cell in map_carbon_cell.items():

        valid_loc = [(_loc[0], _loc[1])]  # 四个位置，按行遍历时从小到大

        forced_pos_valid_loc = str(valid_loc).replace('-1', '14')  # 因为棋盘大小是 15 * 15
        forced_pos_valid_loc = eval(forced_pos_valid_loc.replace('15', '0'))

        filter_cell = \
            [_c for _, _c in map_carbon_cell.items() if getattr(_c, "position", (-100, -100)) in forced_pos_valid_loc]

        assert len(filter_cell) == 1  # 因为选取周围四个值来吸收碳

        carbon_contain_dict[cell] = sum([_fc.carbon for _fc in filter_cell])

    map_carbon_sum_sorted = dict(sorted(carbon_contain_dict.items(), key=lambda x: x[1], reverse=True))

    return map_carbon_sum_sorted



class CollectorAct(AgentBase):
    def __init__(self):
        super().__init__()
        self.action_dict = {}
        for move in WorkerAction.moves():
            self.action_dict[move.name] = move
        self.collector_target = dict()

    @staticmethod
    def _minimum_distance(point_1, point_2):
        abs_distance = abs(point_1 - point_2)
        cross_distance = min(point_1, point_2) + (15 - max(point_1, point_2))  # TODO: 这里对吗，是14减?
        return min(abs_distance, cross_distance)

    def _calculate_distance(self, collector_position, current_position):
        """计算真实距离，计算跨图距离，取两者最小值"""
        x_distance = self._minimum_distance(collector_position[0], current_position[0])
        y_distance = self._minimum_distance(collector_position[1], current_position[1])

        return x_distance + y_distance

    def _target_plan(self, collector: Collector, carbon_sort_dict: Dict, ours_info, oppo_info):
        """结合某一位置的碳的含量和距离"""
        # TODO：钱够不够是否考虑？
        global overall_plan
        collector_position = collector.position
        # 取碳排量最高的前十
        carbon_sort_dict_top_n = \
            {_v: _k for _i, (_v, _k) in enumerate(carbon_sort_dict.items()) if
             _i < 225}  # 只选取含碳量top_n的cell来进行计算，拿全部的cell可能会比较耗时？
        # 计算planter和他的相对距离，并且结合该位置四周碳的含量，得到一个总的得分
        planned_target = [Point(*_v.position) for _k, _v in self.collector_target.items()]
        max_score, max_score_cell = -1e9, None
        for _cell, _carbon_sum in carbon_sort_dict_top_n.items():
            #if (_cell.position not in planned_target):  # 这个位置不在其他智能体正在进行的plan中
            if (_cell.position not in planned_target):  # 这个位置不在其他智能体正在进行的plan中
                collector_to_cell_distance = self._calculate_distance(collector_position, _cell.position)  # 我们希望这个距离越小越好
                cell_to_center_distance = self._calculate_distance(_cell.position, ours_info.recrtCenters[0].position)
                collector_to_center_distance = self._calculate_distance(collector_position, ours_info.recrtCenters[0].position)

                #if collector_to_center_distance!=collector_to_cell_distance+cell_to_center_distance:
                #collector_to_center_distance += (collector_to_cell_distance+cell_to_center_distance-collector_to_center_distance) /3
                target_preference_score = min(_carbon_sum*(1.03**collector_to_cell_distance), 100)/ (0.5*collector_to_cell_distance+1)
                #target_preference_score = min(_carbon_sum*(1.03**collector_to_cell_distance), 100) - (collector_to_cell_distance) * 5
                #if collector_to_cell_distance == 0:
                #    target_preference_score += _carbon_sum * 2
                #if collector_to_cell_distance == 1:
                #    target_preference_score += _carbon_sum

                #_carbon_sum + collector_to_cell_distance * (-7)  # 不考虑碳总量只考虑距离 TODO: 这会导致中了很多树，导致后期花费很高
                num0 = cal_region_num(ours_info, oppo_info, _cell.position, 3)
                target_preference_score -= num0 * 2

                if target_preference_score > max_score:
                    max_score = target_preference_score
                    max_score_cell = _cell

        #print(max_score)
        if max_score_cell is None:  # 没有找到符合条件的最大得分的cell，随机选一个cell
            max_score_cell = random.choice(list(carbon_sort_dict_top_n))

        if self.rct_attacker_id == collector.id and \
            self._calculate_distance(collector_position, oppo_info[0].recrtCenters[0].position)>0:
                max_score_cell = oppo_info[0].recrtCenters[0].cell

        return max_score_cell

    def _target_plan_2_home(self, collector: Collector, carbon_sort_dict: Dict, ours_info, oppo_info):
        """结合某一位置的碳的含量和距离"""
        # TODO：钱够不够是否考虑？
        global overall_plan
        collector_position = collector.position
        # 取碳排量最高的前十
        carbon_sort_dict_top_n = \
            {_v: _k for _i, (_v, _k) in enumerate(carbon_sort_dict.items()) if
             _i < 225}  # 只选取含碳量top_n的cell来进行计算，拿全部的cell可能会比较耗时？
        # 计算planter和他的相对距离，并且结合该位置四周碳的含量，得到一个总的得分
        planned_target = [Point(*_v.position) for _k, _v in self.collector_target.items()]
        max_score, max_score_cell = -1e9, None

        dis0 = self._calculate_distance(ours_info.recrtCenters[0].cell.position, collector.position)
        dis1 = self._calculate_distance(ours_info.recrtCenters[1].cell.position, collector.position)
        if dis0<dis1:
            r_id = 0
        else:
            r_id = 1

        num0 = cal_region_num(ours_info, oppo_info, ours_info.recrtCenters[0].position, 3)
        num1 = cal_region_num(ours_info, oppo_info, ours_info.recrtCenters[1].position, 3)
        if abs(num0-num1)>2:
            if num0<num1:
                r_id = 0
            else:
                r_id = 1


        max_score_cell = ours_info.recrtCenters[r_id].cell
        for _cell, _carbon_sum in carbon_sort_dict_top_n.items():
            #if (_cell.position not in planned_target):  # 这个位置不在其他智能体正在进行的plan中
            if (_cell.position not in planned_target):  # 这个位置不在其他智能体正在进行的plan中
                collector_to_cell_distance = self._calculate_distance(collector_position, _cell.position)  # 我们希望这个距离越小越好
                cell_to_center_distance = self._calculate_distance(_cell.position, ours_info.recrtCenters[r_id].position)
                collector_to_center_distance = self._calculate_distance(collector_position, ours_info.recrtCenters[r_id].position)

                if collector_to_center_distance!=collector_to_cell_distance+cell_to_center_distance:
                    continue
                if _carbon_sum*(1.03**collector_to_cell_distance)<15:
                    continue
                collector_to_center_distance += (collector_to_cell_distance+cell_to_center_distance-collector_to_center_distance) /3
                target_preference_score = min(_carbon_sum*(1.03**collector_to_cell_distance)/(0.3*collector_to_cell_distance+1), 200)

                if target_preference_score > max_score:
                    max_score = target_preference_score
                    max_score_cell = _cell

        if max_score_cell is None:  # 没有找到符合条件的最大得分的cell，随机选一个cell
            max_score_cell = random.choice(list(carbon_sort_dict_top_n))

        return max_score_cell

    def _target_plan_attacker(self, collector: Collector, ours_info, oppo_info):
        global overall_plan

        max_score, max_score_cell = -1e9, None

        max_score_cell = oppo_info[0].recrtCenters[0].cell
        collector_position = collector.position
        for oppo_collector in oppo_info[0].collectors:
            if oppo_collector.carbon <= collector.carbon:
                continue
            _cell = oppo_collector.cell
            collector_to_cell_distance = self._calculate_distance(collector_position, _cell.position)

            target_preference_score = -collector_to_cell_distance + oppo_collector.carbon/30

            if target_preference_score > max_score:
                max_score = target_preference_score
                max_score_cell = _cell

        for oppo_planter in oppo_info[0].planters:
            _cell = oppo_planter.cell
            collector_to_cell_distance = self._calculate_distance(collector_position, _cell.position)

            target_preference_score = -collector_to_cell_distance

            if target_preference_score > max_score:
                max_score = target_preference_score
                max_score_cell = _cell

        if self.rct_attacker_id == collector.id:
            dis0 = self._calculate_distance(collector_position, oppo_info[0].recrtCenters[0].position)
            dis1 = self._calculate_distance(collector_position, oppo_info[0].recrtCenters[1].position)
            if dis0 < dis1:
                r_id = 0
            else:
                r_id = 1
            if min(dis0, dis1)>1:
                max_score_cell = oppo_info[0].recrtCenters[r_id].cell

        return max_score_cell

    def cal_cell_tree_num(self, cell, ours_info):
        check_cell_list = [cell.left, cell.right, cell.down, cell.up, cell.up.right, cell.up.left,
                           cell.down.left, cell.down.right]
        return sum([False if (_c.tree is None or _c.tree.player_id==ours_info.id) else True for _c in check_cell_list])

    def _check_surround_validity(self, move: WorkerAction, collector: Collector, steps) -> bool:
        move = move.name

        if move == 'UP':
            # 需要看前方三个位置有没有Agent
            check_cell_list = [collector.cell.up]
        elif move == 'DOWN':
            check_cell_list = [collector.cell.down]
        elif move == 'RIGHT':
            check_cell_list = [collector.cell.right]
        elif move == 'LEFT':
            check_cell_list = [collector.cell.left]
        else:
            raise NotImplementedError


        if move == 'UP':
            # 需要看前方三个位置有没有Agent
            check_cell_list_2 = [collector.cell.up, collector.cell.up.left, collector.cell.up.right, collector.cell.up.up]
        elif move == 'DOWN':
            check_cell_list_2 = [collector.cell.down, collector.cell.down.left, collector.cell.down.right, collector.cell.down.down]
        elif move == 'RIGHT':
            check_cell_list_2 = [collector.cell.right, collector.cell.right.up, collector.cell.right.down, collector.cell.right.right]
        elif move == 'LEFT':
            check_cell_list_2 = [collector.cell.left, collector.cell.left.up, collector.cell.left.down, collector.cell.left.left]
        else:
            raise NotImplementedError


        global overall_plan
        term2 = all([False if (_c.collector is not None and (_c.collector.carbon<collector.carbon and
                                _c.collector.player_id!=collector.player_id)) else True for _c in check_cell_list_2])

        term3 = all([False if (_c.recrtCenter is not None and _c.recrtCenter.player_id!=collector.player_id) else True
                     for _c in check_cell_list])
        #term2 = True
        return all([True if ((_c.collector is None or (_c.collector.carbon<collector.carbon and
                                _c.collector.player_id!=collector.player_id))
                             and (not _c.position in overall_plan)
                             ) or ((_c.recrtCenter is not None)) or steps>290
                    else False for _c in check_cell_list]) and term2 and term3

    def _check_surround_validity_cell(self, collector: Collector, steps) -> bool:
        check_cell_list = [collector.cell]
        check_cell_list_2 = [collector.cell.up, collector.cell.left, collector.cell.right, collector.cell.down]

        global overall_plan
        term2 = all([False if (_c.collector is not None and (_c.collector.carbon<collector.carbon and
                                _c.collector.player_id!=collector.player_id)) else True for _c in check_cell_list_2])
        #term2 = True
        return all([True if (not _c.position in overall_plan)
                              or ((_c.recrtCenter is not None) and steps>270) or steps>290
                    else False for _c in check_cell_list]) and term2

    def get_min_oppo_dis(self, collector, oppo_info):

        min_dis = 1000000
        collector_position = collector.position
        for oppo_collector in oppo_info[0].collectors:
            if oppo_collector.carbon < collector.carbon:
                continue
            _cell = oppo_collector.cell
            collector_to_cell_distance = self._calculate_distance(collector_position, _cell.position)

            if  collector_to_cell_distance< min_dis:
                min_dis = collector_to_cell_distance
                max_score_cell = _cell

        for oppo_planter in oppo_info[0].planters:
            _cell = oppo_planter.cell
            collector_to_cell_distance = self._calculate_distance(collector_position, _cell.position)

            if  collector_to_cell_distance< min_dis:
                min_dis = collector_to_cell_distance
                max_score_cell = _cell

        return min_dis

    def move(self, ours_info, oppo_info, **kwargs):
        global overall_plan, attacker_sum
        move_action_dict = dict()

        """需要知道本方当前位置信息，敵方当前位置信息，地图上的碳的分布"""
        # 如果planter信息是空的，则无需执行任何操作
        if ours_info.collectors == []:
            return None

        map_carbon_cell = kwargs["map_carbon_location"]
        carbon_sort_dict = calculate_carbon_contain_single(map_carbon_cell)  # 每一次move都先计算一次附近碳多少的分布
        '''
        ha = 0
        for _cell in carbon_sort_dict:
            if _cell.carbon>0:
                ha+=1
        '''

        self.rct_attacker_id = -1
        dis_list = []
        min_dis = 100000
        for collector in ours_info.collectors:
            # 先给他随机初始化一个行动
            if collector.carbon>0:
                continue
            #attacker = False
            #tmp_dis = self._calculate_distance(collector.position, oppo_info[0].recrtCenters[0].position)
            dis0 = self._calculate_distance(collector.position, oppo_info[0].recrtCenters[0].position)
            dis1 = self._calculate_distance(collector.position, oppo_info[0].recrtCenters[1].position)
            tmp_dis = min(dis0, dis1)
            if dis0 < dis1:
                r_id = 0
            else:
                r_id = 1
            min_dis = min(tmp_dis, min_dis)
            if tmp_dis == min_dis:
                self.rct_attacker_id = collector.id
            dis_list.append(self.get_min_oppo_dis(collector, oppo_info))

        if ours_info.cash < 400 or min_dis>3:
            self.rct_attacker_id = -1
        dis_list.sort()

        attacker_sum = min(len(ours_info.collectors)//2, len(dis_list)-1)
        if attacker_sum <= 0:
            thresh = 0
        else:
            thresh = dis_list[attacker_sum-1]

        attacker_sum = 0
        go_home_sum = 0
        for collector in ours_info.collectors:
            # 先给他随机初始化一个行动
            attacker = False
            #if self.get_min_oppo_dis(collector, oppo_info)<=thresh:
            #    attacker = True

            attacker = False
            if self.get_min_oppo_dis(collector, oppo_info)<=2 and collector.carbon<10 and ours_info.cash>100:
                if random.random()<0.7:
                    attacker = True
                    attacker_sum += 1

            if self.get_min_oppo_dis(collector, oppo_info)<=1 and collector.carbon<40 and ours_info.cash>100:
                attacker = True
                attacker_sum += 1

            if attacker_sum<=0:
                attacker = False

            attacker_sum -=attacker
            #if collector.carbon==0 and collector.position==ours_info.recrtCenters[0].position:#self._calculate_distance(collector.position, oppo_info[0].recrtCenters[0].position)<5:
            #    if random.random() < 0.:
            #        self.attacker = True
            #    else:
            #        self.attacker = False
            dis0 = self._calculate_distance(ours_info.recrtCenters[0].position, collector.position)
            dis1 = self._calculate_distance(ours_info.recrtCenters[1].position, collector.position)
            dis_to_home = min(dis0, dis1)
            if dis0<dis1:
                r_id = 0
            else:
                r_id = 1
            carbon_thresh = 100
            if ours_info.cash<30:
                carbon_thresh = 20
            if ours_info.cash > 500:
                carbon_thresh = 120 + dis_to_home*20
            if ours_info.cash>2000:
                carbon_thresh = 120 + dis_to_home*40
            if (collector.id in self.collector_target) and collector.position == self.collector_target[collector.id].position:
                self.collector_target.pop(collector.id)
            if collector.id not in self.collector_target:  # 说明他还没有策略，要为其分配新的策略
                if attacker:
                    target_cell = self._target_plan_attacker(collector, ours_info, oppo_info)
                elif collector.carbon<carbon_thresh and (300-kwargs['step'] >= dis_to_home+9):
                    target_cell = self._target_plan(collector, carbon_sort_dict, ours_info, oppo_info)  # 返回这个智能体要去哪里的一个字典
                elif (300-kwargs['step']<dis_to_home+9):
                    target_cell = ours_info.recrtCenters[r_id].cell
                else:
                    #if go_home_sum >= len(ours_info.collectors)/4*3:
                    #    target_cell = self._target_plan(collector, carbon_sort_dict, ours_info, oppo_info)
                    #else:
                    go_home_sum += 1
                    target_cell = self._target_plan_2_home(collector, carbon_sort_dict, ours_info, oppo_info)
                self.collector_target[collector.id] = target_cell  # 给它新的行动
            #else:  # 说明他有策略，看策略是否执行完毕，执行完了移出字典，没有执行完接着执行
            if collector.position == self.collector_target[collector.id].position:
                if self._check_surround_validity_cell(collector, kwargs['step']):
                    overall_plan[collector.position] = 1
                else:
                    filtered_list_act = WorkerAction.moves()
                    for move in WorkerAction.moves():
                        if not self._check_surround_validity(move, collector, kwargs['step']):
                            filtered_list_act.remove(move)
                    if len(filtered_list_act) == 0:
                        filtered_list_act.append(move)
                    if not collector.id in move_action_dict:
                        tmp = random.choice(filtered_list_act)
                        move_action_dict[collector.id] = tmp.name
                        new_position = cal_new_pos(collector.position, tmp)
                        overall_plan[new_position] = 1
                self.collector_target.pop(collector.id)
            else:  # 没有执行完接着执行
                old_position = collector.position
                target_position = self.collector_target[collector.id].position
                old_distance = self._calculate_distance(old_position, target_position)

                filtered_list_act = WorkerAction.moves()
                for move in WorkerAction.moves():
                    if not self._check_surround_validity(move, collector, kwargs['step']):
                        filtered_list_act.remove(move)

                best_move = WorkerAction.UP
                best_choice = []
                for move in WorkerAction.moves():
                    new_position = cal_new_pos(old_position, move)
                    new_distance = self._calculate_distance(new_position, target_position)

                    if new_distance < old_distance:
                        best_move = move
                        if self.cal_cell_tree_num(
                                cal_new_pos_cell(collector.cell, move), ours_info) >= 2 and collector.carbon > 50:
                            continue
                        #if self.cal_cell_tree_num(cal_new_pos_cell(collector.cell, move), ours_info)>=1 and collector.carbon>80:
                        #    continue
                        if self._check_surround_validity(move, collector, kwargs['step']):
                            best_choice.append(move)
                            continue
                            #move_action_dict[collector.id] = move.name
                            #overall_plan[new_position] = 1
                            #break
                if len(best_choice)>0:
                    move = random.choice(best_choice)
                    new_position = cal_new_pos(old_position, move)
                    move_action_dict[collector.id] = move.name
                    overall_plan[new_position] = 1

                #if len(filtered_list_act) == 0:
                #    filtered_list_act.append(move)
                if not attacker and self._check_surround_validity_cell(collector, kwargs['step']):
                    filtered_list_act.append('')
                if len(filtered_list_act) == 0:
                    filtered_list_act.append(best_move)
                if not collector.id in move_action_dict:
                    tmp = random.choice(filtered_list_act)
                    if tmp!='':
                        move_action_dict[collector.id] = tmp.name
                        new_position = cal_new_pos(old_position, tmp)
                        overall_plan[new_position] = 1
                    else:
                        #move_action_dict.pop(collector.id)
                        overall_plan[old_position] = 1

                self.collector_target.pop(collector.id) #每步决策

        return move_action_dict

def cal_new_pos(pos, move):
    new_position = pos + move.to_point()
    new_position = Point(*eval(str(new_position).replace("15", "0")))
    new_position = Point(*eval(str(new_position).replace("-1", "14")))
    return new_position

def cal_new_pos_cell(cell, move):
    new_cell = cell
    if move.name=='UP':
        new_cell = cell.up
    if move.name == 'DOWN':
        new_cell = cell.down
    if move.name == 'LEFT':
        new_cell = cell.left
    if move.name == 'RIGHT':
        new_cell = cell.right
    return new_cell

class PlanterAct(AgentBase):
    def __init__(self):
        super().__init__()
        self.workaction = WorkerAction
        self.planter_target = dict()

    @ staticmethod
    def _minimum_distance(point_1, point_2):
        abs_distance = abs(point_1 - point_2)
        cross_distance = min(point_1, point_2) + (15 - max(point_1, point_2))  # TODO: 这里对吗，是14减?
        return min(abs_distance, cross_distance)

    def _calculate_distance(self, planter_position, current_position):
        """计算真实距离，计算跨图距离，取两者最小值"""
        x_distance = self._minimum_distance(planter_position[0], current_position[0])
        y_distance = self._minimum_distance(planter_position[1], current_position[1])

        return x_distance + y_distance

    def _target_plan(self, planter: Planter, carbon_sort_dict: Dict, ours_info, oppo_info):
        """结合某一位置的碳的含量和距离"""
        # TODO：钱够不够是否考虑？
        planter_position = planter.position
        # 取碳排量最高的前十
        carbon_sort_dict_top_n = \
            {_v: _k for _i, (_v, _k) in enumerate(carbon_sort_dict.items()) if _i < 100}  # 只选取含碳量top_n的cell来进行计算，拿全部的cell可能会比较耗时？
        # 计算planter和他的相对距离，并且结合该位置四周碳的含量，得到一个总的得分
        planned_target = [Point(*_v.position) for _k, _v in self.planter_target.items()]
        max_score, max_score_cell = -1e9, None
        for _cell, _carbon_sum in carbon_sort_dict_top_n.items():
            if (_cell.tree is None) and (_cell.position not in planned_target) and (_cell.recrtCenter is None):  # 这个位置没有树，且这个位置不在其他智能体正在进行的plan中
                planter_to_cell_distance = self._calculate_distance(planter_position, _cell.position)  # 我们希望这个距离越小越好
                target_preference_score = 0 * _carbon_sum + np.log(1 / (planter_to_cell_distance + 1e-9)) # 不考虑碳总量只考虑距离 TODO: 这会导致中了很多树，导致后期花费很高
                target_preference_score = _carbon_sum * 1.5 + planter_to_cell_distance * (-25)

                if _carbon_sum < 50:
                    target_preference_score -= (
                                20 + self.cal_tree_money(len(ours_info.trees) + len(oppo_info[0].trees))) * 8
                if planter_to_cell_distance == 0:
                    target_preference_score += _carbon_sum * 1.2
                    if _carbon_sum<self.cal_tree_money(len(ours_info.trees)+len(oppo_info[0].trees))*2 or _carbon_sum<50:
                        target_preference_score -= _carbon_sum * 1.2 + (20 + self.cal_tree_money(len(ours_info.trees)+len(oppo_info[0].trees)))*10

                target_preference_score -= cal_region_num(ours_info, oppo_info, _cell.position, 4) * 9
                if planter_to_cell_distance==1:
                    target_preference_score += _carbon_sum * 0.7

                if self.cal_cell_tree_num(_cell) > 0:
                    target_preference_score -= 400

                if self.cal_cell_tree_num(_cell)>1:
                    target_preference_score -= 400

                #target_preference_score = min(_carbon_sum*(1.05**planter_to_cell_distance)/(planter_to_cell_distance+1), 200)

                if target_preference_score > max_score:
                    max_score = target_preference_score
                    max_score_cell = _cell

        '''
        if planter.player_id==0:
            center = (10, 4)
        else:
            center = (4, 10)
        '''

        for _cell, _carbon_sum in carbon_sort_dict.items():

            if self.expendable_id==planter.id:
                dis0 = self._calculate_distance(planter.position, oppo_info[0].recrtCenters[0].position)
                dis1 = self._calculate_distance(planter.position, oppo_info[0].recrtCenters[1].position)
                if dis0 < dis1:
                    r_id = 0
                else:
                    r_id = 1
                if _cell.position!=oppo_info[0].recrtCenters[r_id].position \
                        and (_cell.tree is None or _cell.tree.player_id != planter.player_id)\
                        and abs(_cell.position[0]-oppo_info[0].recrtCenters[r_id].position[0])==1\
                        and abs(_cell.position[1]-oppo_info[0].recrtCenters[r_id].position[1])==1:

                    planter_to_cell_distance = self._calculate_distance(planter_position, _cell.position)
                    target_preference_score = 10000 + planter_to_cell_distance * (-20) #+_carbon_sum + planter_to_cell_distance * (-20) #+ 10*len(oppo_info[0].collectors)
                    if target_preference_score > max_score:
                        max_score = target_preference_score
                        max_score_cell = _cell

            if not _cell.tree is None:
                if _cell.tree.player_id != planter.player_id:
                    cell_to_center_dis = self._calculate_distance(_cell.position, ours_info.recrtCenters[0].position)
                    cell_to_center_dis = min(cell_to_center_dis, self._calculate_distance(_cell.position, ours_info.recrtCenters[1].position))
                    planter_to_cell_distance = self._calculate_distance(planter_position, _cell.position)
                    if 50 - (_cell.tree.age + planter_to_cell_distance) < 8 and cell_to_center_dis>2:
                        continue
                    target_preference_score = 0 * _carbon_sum + np.log(1 / (planter_to_cell_distance + 1e-9)) + 200
                    if len(ours_info.trees)+len(oppo_info[0].trees)>8:
                        pri = 500
                    else:
                        pri = 200

                    _carbon_sum = (50 - (_cell.tree.age + planter_to_cell_distance))*8
                    target_preference_score = _carbon_sum + planter_to_cell_distance * (-30) + pri + \
                                              (self.cal_tree_money(len(ours_info.trees)+len(oppo_info[0].trees)) - 20)*1.5
                    if cell_to_center_dis<=2:
                        target_preference_score += 200
                    else:
                        target_preference_score += 1/cell_to_center_dis * 500
                    #if planter_to_cell_distance>4:
                    #    target_preference_score -= 2**planter_to_cell_distance
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
            check_cell_list = [planter.cell.up, planter.cell.up.left, planter.cell.up.right, planter.cell.up.up]
        elif move == 'DOWN':
            check_cell_list = [planter.cell.down, planter.cell.down.left, planter.cell.down.right, planter.cell.down.down]
        elif move == 'RIGHT':
            check_cell_list = [planter.cell.right, planter.cell.right.up, planter.cell.right.down, planter.cell.right.right]
        elif move == 'LEFT':
            check_cell_list = [planter.cell.left, planter.cell.left.up, planter.cell.left.down, planter.cell.left.left]
        else:
            raise NotImplementedError

        global overall_plan
        return all([True if ((_c.collector is None or (_c.collector.player_id == planter.player_id)) and
                             (not _c.position in overall_plan)) else False for _c in check_cell_list])

    def _check_surround_validity_cell(self, planter: Planter) -> bool:
        check_cell_list = [planter.cell]
        check_cell_list_2 = [planter.cell.up, planter.cell.left, planter.cell.right, planter.cell.down]

        global overall_plan
        term2 = all([False if (_c.collector is not None and (_c.collector.player_id!=planter.player_id)) else True for _c in check_cell_list_2])
        #term2 = True
        term1 = all([False if _c.position in overall_plan else True for _c in check_cell_list])
        return term2 and term1

    def cal_tree_money(self, tree_num):
        return 5 * 1.235 ** tree_num

    def cal_cell_tree_num(self, cell):
        check_cell_list = [cell.left, cell.right, cell.down, cell.up, cell.up.right, cell.up.left,
                           cell.down.left, cell.down.right,
                           cell.left.left, cell.right.right, cell.down.down, cell.up.up]
        return sum([False if (_c.tree is None) else True for _c in check_cell_list])

    def move(self, ours_info, oppo_info, **kwargs):
        global overall_plan
        move_action_dict = dict()

        """需要知道本方当前位置信息，敵方当前位置信息，地图上的碳的分布"""
        # 如果planter信息是空的，则无需执行任何操作
        if ours_info.planters == []:
            return None

        map_carbon_cell = kwargs["map_carbon_location"]
        carbon_sort_dict = calculate_carbon_contain(map_carbon_cell)  # 每一次move都先计算一次附近碳多少的分布

        min_dis = 100000
        self.expendable_id = -1

        for planter in ours_info.planters:
            dis0 = self._calculate_distance(planter.position, oppo_info[0].recrtCenters[0].position)
            dis1 = self._calculate_distance(planter.position, oppo_info[0].recrtCenters[1].position)
            if dis0 < dis1:
                r_id = 0
            else:
                r_id = 1

            planter_to_oppo_dis = self._calculate_distance(planter.position, oppo_info[0].recrtCenters[r_id].position)
            min_dis = min(min_dis, planter_to_oppo_dis)
            if min_dis==planter_to_oppo_dis and len(oppo_info[0].collectors)>2 and ours_info.cash>800 and min_dis<=4:
                self.expendable_id = planter.id

        for planter in ours_info.planters:
            # 先给他随机初始化一个行动
            if planter.id not in self.planter_target:   # 说明他还没有策略，要为其分配新的策略
                target_cell = self._target_plan(planter, carbon_sort_dict, ours_info, oppo_info)  # 返回这个智能体要去哪里的一个字典
                self.planter_target[planter.id] = target_cell  # 给它新的行动
            #else:  # 说明他有策略，看策略是否执行完毕，执行完了移出字典，没有执行完接着执行
            if planter.position == self.planter_target[planter.id].position:
                # 执行一次种树行动, TODO: 如果钱够就种树，钱不够不执行任何操作
                # move_action_dict[planter.id] = None
                # TODO: 这里不执行任何行动就表示种树了？
                # 移出字典
                money = 20 + self.cal_tree_money(len(ours_info.trees)+len(oppo_info[0].trees))
                if (planter.cell.tree is not None) and planter.cell.tree.player_id!=planter.player_id:
                    money = 20
                carbon = cal_carbon_one_cell(planter.cell, map_carbon_cell)
                if (((money * 1.4 < carbon or money<50) and (kwargs['step']<295)) or self.expendable_id==planter.id)\
                        and self._check_surround_validity_cell(planter):
                    overall_plan[planter.position] = 1
                else:
                    filtered_list_act = WorkerAction.moves()
                    for move in WorkerAction.moves():
                        if not self._check_surround_validity(move, planter):
                            filtered_list_act.remove(move)

                    if len(filtered_list_act) == 0:
                        filtered_list_act.append(move)
                    if not planter.id in move_action_dict:
                        tmp = random.choice(filtered_list_act)
                        move_action_dict[planter.id] = tmp.name
                        new_position = cal_new_pos(planter.position, tmp)
                        overall_plan[new_position] = 1

                self.planter_target.pop(planter.id)

            else:  # 没有执行完接着执行
                old_position = planter.position
                target_position = self.planter_target[planter.id].position
                old_distance = self._calculate_distance(old_position, target_position)

                filtered_list_act = WorkerAction.moves()
                for move in WorkerAction.moves():
                    if not self._check_surround_validity(move, planter):
                        filtered_list_act.remove(move)

                for move in WorkerAction.moves():
                    new_position = cal_new_pos(old_position, move)
                    new_distance = self._calculate_distance(new_position, target_position)

                    if new_distance < old_distance:
                        if self._check_surround_validity(move, planter):
                            move_action_dict[planter.id] = move.name
                            overall_plan[new_position] = 1
                            break

                if len(filtered_list_act)==0:
                    filtered_list_act.append(move)
                if not planter.id in move_action_dict:
                    tmp = random.choice(filtered_list_act)
                    move_action_dict[planter.id] = tmp.name
                    new_position = cal_new_pos(old_position, tmp)
                    overall_plan[new_position] = 1

                self.planter_target.pop(planter.id)

        return move_action_dict

def _minimum_distance(point_1, point_2):
    abs_distance = abs(point_1 - point_2)
    cross_distance = min(point_1, point_2) + (15 - max(point_1, point_2))  # TODO: 这里对吗，是14减?
    return min(abs_distance, cross_distance)

def _calculate_distance(planter_position, current_position):
    """计算真实距离，计算跨图距离，取两者最小值"""
    x_distance = _minimum_distance(planter_position[0], current_position[0])
    y_distance = _minimum_distance(planter_position[1], current_position[1])

    return x_distance + y_distance

def cal_region_num(ours_info, oppo_info, position, radius):
    sum = 0
    for collector in ours_info.collectors:
        dis = _calculate_distance(collector.position, position)
        if dis <= radius:
            sum += 1
    for planter in ours_info.planters:
        dis = _calculate_distance(planter.position, position)
        if dis<=radius:
            sum+=1

    return sum

class RecruiterAct(AgentBase):
    def __init__(self):
        super().__init__()

    def action(self, ours_info, oppo_info, **kwargs):
        store_dict = dict()

        global overall_plan

        sum0 = cal_region_num(ours_info, oppo_info, ours_info.recrtCenters[0].position, 5)
        sum1 = cal_region_num(ours_info, oppo_info, ours_info.recrtCenters[1].position, 5)
        if sum0<sum1:
            r_list = [0, 1]
        else:
            r_list = [1, 0]
        for r_id in r_list:
            if not ours_info.recrtCenters[r_id].cell.position in overall_plan:
                if len(ours_info.planters) < 3 \
                        and len(ours_info.collectors) >= 7:
                    store_dict[ours_info.recrtCenters[r_id].id] = RecrtCenterAction.RECPLANTER.name
                else:
                    store_dict[ours_info.recrtCenters[r_id].id] = RecrtCenterAction.RECCOLLECTOR.name

                if len(ours_info.planters) < 1 and len(ours_info.collectors) > 0:
                    store_dict[ours_info.recrtCenters[r_id].id] = RecrtCenterAction.RECPLANTER.name

                if len(ours_info.planters) < 2 and len(ours_info.collectors) > 2:
                    store_dict[ours_info.recrtCenters[r_id].id] = RecrtCenterAction.RECPLANTER.name

                #if len(ours_info.planters) > 0 and ours_info.cash < 50:
                #    store_dict.pop(ours_info.recrtCenters[r_id].id)

                if (ours_info.recrtCenters[r_id].cell.collector is not None and
                        ours_info.recrtCenters[r_id].cell.collector.player_id != ours_info.recrtCenters[0].player_id):
                    store_dict[ours_info.recrtCenters[r_id].id] = RecrtCenterAction.RECCOLLECTOR.name

                if (ours_info.cash<60 or (len(ours_info.planters) + len(ours_info.collectors) ==9)) and r_id==r_list[1]:
                    store_dict.pop(ours_info.recrtCenters[r_id].id)

                #if len(ours_info.planters)==0 and r_id==r_list[1]:
                #    store_dict[ours_info.recrtCenters[r_id].id] = RecrtCenterAction.RECPLANTER.name

            else:
                pass

        return store_dict


class PlanningPolicy(BasePolicy):
    def __init__(self, ):
        super().__init__()
        # self.worker = WorkerAct()
        self.collector = CollectorAct()
        self.planter = PlanterAct()
        self.recruiter = RecruiterAct()

    def take_action(self, current_obs: Board, previous_obs: Board) -> Dict:
        global overall_plan
        overall_plan = dict()
        overall_dict = dict()

        ours, oppo = current_obs.current_player, current_obs.opponents

        # 种树员做决策去哪里种树
        planter_dict = self.planter.move(
            ours_info=ours,
            oppo_info=oppo,
            map_carbon_location=current_obs.cells,
            step=current_obs.step,
        )

        if planter_dict is not None:
            overall_dict.update(planter_dict)

        collector_dict = self.collector.move(
            ours_info=ours,
            oppo_info=oppo,
            map_carbon_location=current_obs.cells,
            step=current_obs.step,
        )

        if collector_dict is not None:
            overall_dict.update(collector_dict)

        # 基地先做决策是否招募
        recruit_dict = self.recruiter.action(
            ours_info=ours,
            oppo_info=oppo[0],
            map_carbon_location=current_obs.cells,
            step=current_obs.step,
        )

        # 这里要进行一个判断，确保基地位置没有智能体才能招募下一个

        if recruit_dict is not None:
            overall_dict.update(recruit_dict)

        return overall_dict

        # 对于我方每一个捕碳员，不采取任何行动


class MyPolicy:

    def __init__(self):
        # self.obs_parser = ObservationParser()
        self.policy = PlanningPolicy()

    def take_action(self, observation, configuration):
        global attacker_sum
        attacker_sum = 0
        current_obs = Board(observation, configuration)
        previous_obs = self.previous_obs if current_obs.step > 0 else None

        overall_action = self.policy.take_action(current_obs=current_obs, previous_obs=previous_obs)
        # overall_action = self.to_env_commands(overall_action)

        # agent_obs_dict, dones, available_actions_dict = self.obs_parser.obs_transform(current_obs, previous_obs)
        self.previous_obs = copy.deepcopy(current_obs)

        return overall_action
    
    def reset_record(self):
        self.record_list = []

my_policy = MyPolicy()

def agent(obs, configuration):
    global my_policy
    commands = my_policy.take_action(obs, configuration)
    return commands





