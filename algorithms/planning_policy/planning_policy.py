from os import cpu_count
import sys
import numpy as np
from numpy import positive

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import copy
from abc import ABC, abstractmethod
from typing import Dict, Tuple
from math import log, exp

from zerosum_env.envs.carbon.helpers import (Board, Cell, Collector, Planter,
                                             Point, RecrtCenter, Worker,
                                             RecrtCenterAction, WorkerAction)
from random import randint, shuffle


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


# Plan是一个Agent行动的目标，它可以由一个Action完成(比如招募捕碳者），也可以由多
# 个Action完成（比如种树者走到一个地方去种树）
#
# 我们的方法是对每个Agent用我们设计的优先级函数选出最好的Plan，然后对每个Agent把这个Plan翻译成(当前最好的)Action
class BasePlan(ABC):
    #这里的source_agent,target都是对象，而不是字符串
    #source: 实施这个Plan的Agent: collector,planter,recrtCenter
    #target: 被实施这个Plan的对象: collector,planter,recrtCenter,cell
    def __init__(self, source_agent, target, planning_policy):
        self.source_agent = source_agent
        self.target = target
        self.planning_policy = planning_policy
        self.preference_index = None  #这个Plan的优先级因子

    #根据Plan生成Action
    @abstractmethod
    def translate_to_action(self):
        pass


# 这个类是由转化中心实施的Plans
class RecrtCenterPlan(BasePlan):
    def __init__(self, source_agent, target, planning_policy):
        super().__init__(source_agent, target, planning_policy)


# 这个Plan是指转化中心招募种树者
class SpawnPlanterPlan(RecrtCenterPlan):
    def __init__(self, source_agent, target, planning_policy):
        super().__init__(source_agent, target, planning_policy)
        self.calculate_score()

    #CUSTOM:根据策略随意修改
    #计算转化中心生产种树者的优先级因子
    #当前策略是返回PlanningPolicy中设定的固定值或者一个Mask(代表关闭，值为负无穷)
    def calculate_score(self):
        #is valid
        if self.check_validity() == False:
            self.preference_index = self.planning_policy.config[
                'mask_preference_index']
        else:
            self_planters_count=self.planning_policy.game_state['our_player'].planters.__len__() 
            self_collectors_count =  self.planning_policy.game_state['our_player'].collectors.__len__() 
            self.preference_index =  (self.planning_policy.config['enabled_plans']['SpawnPlanterPlan']['planter_count_weight'] * self_planters_count \
                + self.planning_policy.config['enabled_plans']['SpawnPlanterPlan']['collector_count_weight'] * self_collectors_count \
                + 1) / 1000

    def check_validity(self):
        #没有开启
        if self.planning_policy.config['enabled_plans'][
                'SpawnPlanterPlan']['enabled'] == False:
            return False
        #类型不对
        if not isinstance(self.source_agent, RecrtCenter):
            return False
        if not isinstance(self.target, Cell):
            return False

        #位置不对
        if self.source_agent.cell != self.target:
            return False

        #人口已满
        if self.planning_policy.game_state['our_player'].planters.__len__() + self.planning_policy.game_state['our_player'].collectors.__len__() >= 10:
            return False

        #钱不够
        if self.planning_policy.game_state[
                'our_player'].cash < self.planning_policy.game_state[
                    'configuration']['recPlanterCost']:
            return False
        return True

    def translate_to_action(self):
        if self.planning_policy.global_position_mask.get(self.source_agent.position, 0) == 0:
            self.planning_policy.global_position_mask[self.source_agent.position] = 1
            return RecrtCenterAction.RECPLANTER
        else:
            return None

# 这个Plan是指转化中心招募捕碳者
class SpawnCollectorPlan(RecrtCenterPlan):
    def __init__(self, source_agent, target, planning_policy):
        super().__init__(source_agent, target, planning_policy)
        self.calculate_score()

    #CUSTOM:根据策略随意修改
    #计算转化中心生产种树者的优先级因子
    #当前策略是返回PlanningPolicy中设定的固定值或者一个Mask(代表关闭，值为负无穷)
    def calculate_score(self):
        #is valid
        if self.check_validity() == False:
            self.preference_index = self.planning_policy.config[
                'mask_preference_index']
        else:
            self_planters_count=self.planning_policy.game_state['our_player'].planters.__len__() 
            self_collectors_count =  self.planning_policy.game_state['our_player'].collectors.__len__() 
            self.preference_index =  (self.planning_policy.config['enabled_plans']['SpawnCollectorPlan']['planter_count_weight'] * self_planters_count \
                + self.planning_policy.config['enabled_plans']['SpawnCollectorPlan']['collector_count_weight'] * self_collectors_count \
                    + 1) / 1000 + 0.0001

    def check_validity(self):
        #没有开启
        if self.planning_policy.config['enabled_plans'][
                'SpawnCollectorPlan']['enabled'] == False:
            return False
        #类型不对
        if not isinstance(self.source_agent, RecrtCenter):
            return False
        if not isinstance(self.target, Cell):
            return False
        #人口已满
        if self.planning_policy.game_state['our_player'].planters.__len__() + self.planning_policy.game_state['our_player'].collectors.__len__() >= 10:
            return False
        #位置不对
        if self.source_agent.cell != self.target:
            return False
        #钱不够
        if self.planning_policy.game_state[
                'our_player'].cash < self.planning_policy.game_state[
                    'configuration']['recCollectorCost']:
            return False
        return True

    def translate_to_action(self):
        if self.planning_policy.global_position_mask.get(self.source_agent.position, 0) == 0:
            self.planning_policy.global_position_mask[self.source_agent.position] = 1
            return RecrtCenterAction.RECCOLLECTOR
        else:
            return None

class PlanterPlan(BasePlan):
    def __init__(self, source_agent, target, planning_policy):
        super().__init__(source_agent, target, planning_policy)
        # self.num_of_trees = len(self.planning_policy.game_state['board'].trees())

    def check_valid(self):
        yes_it_is = isinstance(self.source_agent, Planter)
        return yes_it_is
        
    def get_distance2target(self):
        source_posotion = self.source_agent.position
        target_position = self.target.position
        distance = self.planning_policy.get_distance(
            source_posotion[0], source_posotion[1], target_position[0],
            target_position[1])
        return distance

    def get_total_carbon(self):
        return self.target.up.carbon + self.target.down.carbon + self.target.left.carbon + self.target.right.carbon

    
    # 根据未来走过去的步数n，计算n步之后目标位置的碳含量
    def get_total_carbon_predicted(self, step_number, carbon_growth_rate):
        target_sorrounding_cells = [self.target.up, self.target.down, self.target.left, self.target.right]
        total_carbon = 0
        for cell in target_sorrounding_cells:
            cur_list = [cell.up, cell.down, cell.left, cell.right]
            flag = 0
            for cur_pos in cur_list:
                if cur_pos.tree:
                    flag = 1
                    break
            if flag:
                total_carbon += cell.carbon * (1 + carbon_growth_rate) ** step_number
            else:
                total_carbon += cell.carbon
        return total_carbon


    def can_action(self, action_position):
        if self.planning_policy.global_position_mask.get(action_position, 0) == 0:
            action_cell = self.planning_policy.game_state['board']._cells[action_position]
            flag = True

            collectors = [action_cell.collector,
                          action_cell.up.collector, 
                          action_cell.down.collector,
                          action_cell.left.collector,
                          action_cell.right.collector]

            for worker in collectors:
                if worker is None:
                    continue
                # if worker.player_id == self.source_agent.player_id:
                #     continue
                return False
            
            return True
        else:
            return False

    def get_actual_plant_cost(self):
        configuration = self.planning_policy.game_state['configuration']

        if len(self.planning_policy.game_state['our_player'].tree_ids) == 0:
            return configuration.recPlanterCost
        else:
            return configuration.recPlanterCost + configuration.plantCostInflationRatio * configuration.plantCostInflationBase**self.planning_policy.game_state[
                'board'].trees.__len__()

    def translate_to_action_first(self):
        if self.source_agent.cell == self.target and self.can_action(self.target.position):
            # self.planning_policy.global_position_mask[self.target.position] = 1
            return None, self.target.position
        else:
            old_position = self.source_agent.cell.position
            old_distance = self.planning_policy.get_distance(
                old_position[0], old_position[1], self.target.position[0],
                self.target.position[1])

            move_list = []

            for i, action in enumerate(WorkerActions):
                if action == None:
                    continue
                new_position = (
                    (WorkerDirections[i][0] + old_position[0]+ self.planning_policy.config['row_count']) % self.planning_policy.config['row_count'],
                    (WorkerDirections[i][1] + old_position[1]+ self.planning_policy.config['column_count']) % self.planning_policy.config['column_count'],
                )
                new_distance = self.planning_policy.get_distance(
                    new_position[0], new_position[1], self.target.position[0],
                    self.target.position[1])
                rand_factor = randint(0, 100)
                move_list.append((action, new_position, new_distance, rand_factor))

            move_list = sorted(move_list, key=lambda x: x[2: 4])

            for move, new_position, new_d, _ in move_list:
                if self.can_action(new_position):
                    # self.planning_policy.global_position_mask[new_position] = 1
                    return move, new_position
            return None, old_position

    def translate_to_action_second(self, cash):
        cur, position = self.translate_to_action_first()
        old_position = self.source_agent.cell.position
        if cur is None:
            # 钱不够
            if self.planning_policy.game_state[
               'our_player'].cash < cash:
                waiting_list = WorkerActions[0:]
                # 两倍的概率继续None
                waiting_list.append(None)
                shuffle(waiting_list)
                for i, action in enumerate(waiting_list):
                    if action is None:
                        self.planning_policy.global_position_mask[self.target.position] = 1
                        return None
                    new_position = (
                        (WorkerDirections[i][0] + old_position[0]+ self.planning_policy.config['row_count']) % self.planning_policy.config['row_count'],
                        (WorkerDirections[i][1] + old_position[1]+ self.planning_policy.config['column_count']) % self.planning_policy.config['column_count'],
                    )
                    if self.can_action(new_position):
                        self.planning_policy.global_position_mask[new_position] = 1
                        return action
        else:
            self.planning_policy.global_position_mask[position] = 1
            return cur


    def get_tree_absorb_carbon_speed_at_cell(self, cell: Cell):
        pass


# 种树员 抢树计划
class PlanterRobTreePlan(PlanterPlan):
    def __init__(self, source_agent, target, planning_policy):
        super().__init__(source_agent, target, planning_policy)
        self.calculate_score()

    def calculate_score(self):
        if self.check_validity() == False:
            self.preference_index = self.planning_policy.config[
                'mask_preference_index']
        else:
            if self.target.tree is None:
                self.preference_index = 0.0001
                return
            if self.target.tree.player_id == self.source_agent.player_id:
                self.preference_index = 0.0001
                return 

            # source_posotion = self.source_agent.position
            # target_position = self.target.position
            # distance = self.planning_policy.get_distance(
            #     source_posotion[0], source_posotion[1], target_position[0],
            #     target_position[1])
            distance = self.get_distance2target()

            # self.preference_index = (50 - self.target.tree.age) * self.planning_policy.config[
            #     'enabled_plans']['PlanterRobTreePlan'][
            #         'cell_carbon_weight'] + distance * self.planning_policy.config[
            #             'enabled_plans']['PlanterRobTreePlan'][
            #                 'cell_distance_weight']
            total_carbon = self.get_total_carbon()
            self.preference_index = total_carbon * 0.9625 ** distance
            # print(self.preference_index)

    def check_validity(self):
        #没有开启
        if self.planning_policy.config['enabled_plans'][
                'PlanterRobTreePlan']['enabled'] == False:
            return False
        #类型不对
        if not isinstance(self.source_agent, Planter):
            return False
        if not isinstance(self.target, Cell):
            return False
        
        if self.target.tree is None:
           return False
        if self.target.tree.player_id == self.source_agent.player_id:
           return False
        
        return True

    def translate_to_action(self):
        return self.translate_to_action_second(20)


class PlanterPlantTreePlan(PlanterPlan):
    def __init__(self, source_agent, target, planning_policy):
        super().__init__(source_agent, target, planning_policy)
        self.calculate_score()
    
    def calculate_score(self):
        if self.check_validity() == False:
            self.preference_index = self.planning_policy.config[
                'mask_preference_index']
        else:
            distance2target = self.get_distance2target()
            my_id = self.source_agent.player_id  # 我方玩家id
            worker_dict = self.planning_policy.game_state['board'].workers  # 地图所有的 Planter & Collector
            min_distance = 100000
            source_position = self.source_agent.position
            for work_id, cur_worker in worker_dict.items():
                if cur_worker.player_id != my_id:  # 敌方worker
                    cur_pos = cur_worker.position
                    cur_dis = self.planning_policy.get_distance(source_position[0], source_position[1],
                                                                cur_pos[0], cur_pos[1])
                    if cur_dis < min_distance:
                        min_distance = cur_dis
            # 到这里为止，算出来的 min_distance 是距离敌方worker的最近距离
            cur_json = self.planning_policy.config['enabled_plans']['PlanterPlantTreePlan']
            # w0, w1, w2 = cur_json['cell_carbon_weight'], cur_json['cell_distance_weight'], cur_json['enemy_min_distance_weight']
            # 'tree_damp_rate': 0.08,
            # 'distance_damp_rate': 0.999
            # self.preference_index = exp(total_carbon * w0 + distance2target * w1 + min_distance * w2)

            # 'PlanterPlantTreePlan': {
            #     'enabled': True,
            #     'cell_carbon_weight': 50,
            #     'cell_distance_weight': -40,
            #     'enemy_min_distance_weight': 50,
            #     'tree_damp_rate': 0.08,
            #     'distance_damp_rate': 0.999,
            #     'fuzzy_value': 2,
            #     'carbon_growth_rate': 0.05
            # },

            tree_damp_rate = cur_json['tree_damp_rate']
            distance_damp_rate = cur_json['distance_damp_rate']
            fuzzy_value = cur_json['fuzzy_value']
            carbon_growth_rate =cur_json['carbon_growth_rate']
            total_predict_carbon = self.get_total_carbon_predicted(distance2target, carbon_growth_rate)
            cur_index = total_predict_carbon * (distance_damp_rate ** distance2target) * (min_distance - distance2target) * fuzzy_value
            surroundings = [self.target.up, self.target.down, self.target.left, self.target.right]
            damp_count = 0
            for su in surroundings:
                cur_list = [su.up, su.down, su.left, su.down]
                for eve in cur_list:
                    if eve.tree:
                        damp_count += 1
            self.preference_index = cur_index * (1 - tree_damp_rate * damp_count)

                    
    def translate_to_action(self):
        return self.translate_to_action_second(self.get_actual_plant_cost())

            
    def check_validity(self):
        if self.planning_policy.config['enabled_plans'][
                'PlanterPlantTreePlan']['enabled'] == False:
            return False
        if self.target.tree:
            return False
        
        # if self.planning_policy.game_state[
        #        'our_player'].cash < self.get_actual_plant_cost():
        #        return False
        return True
            




class CollectorPlan(BasePlan):
    def __init__(self, source_agent, target, planning_policy):
        super().__init__(source_agent, target, planning_policy)

    def check_validity(self):
        yes_it_is = isinstance(self.source_agent, Collector)
        return yes_it_is

    def can_action(self, action_position):
        if self.planning_policy.global_position_mask.get(action_position, 0) == 0:
            action_cell = self.planning_policy.game_state['board']._cells[action_position]
            flag = True
            collectors = [action_cell.collector,
                          action_cell.up.collector, 
                          action_cell.down.collector,
                          action_cell.left.collector,
                          action_cell.right.collector]

            for collector in collectors:
                if collector is None:
                    continue
                if collector.player_id == self.source_agent.player_id:
                    continue
                if collector.carbon <= self.source_agent.carbon:
                    return False
            return True
        else:
            return False

    def translate_to_action(self):
        potential_action = None
        potential_action_position = self.source_agent.position
        potential_carbon = -1
        source_position = self.source_agent.position
        target_position = self.target.position
        source_target_distance = self.planning_policy.get_distance(
                source_position[0], source_position[1], target_position[0],
                target_position[1])

        potential_action_list = []

        for i, action in enumerate(WorkerActions):
            action_position = (
                (WorkerDirections[i][0] + source_position[0]+ self.planning_policy.config['row_count']) % self.planning_policy.config['row_count'],
                (WorkerDirections[i][1] + source_position[1]+ self.planning_policy.config['column_count']) % self.planning_policy.config['column_count'],
            )
            if not self.can_action(action_position):
                continue
                        
            target_action_distance = self.planning_policy.get_distance(
                target_position[0], target_position[1], action_position[0],
                action_position[1])
            
            source_action_distance = self.planning_policy.get_distance(
                source_position[0], source_position[1], action_position[0],
                action_position[1])
            
            potential_action_list.append((action, 
                                         action_position,
                                         target_action_distance + source_action_distance - source_target_distance,
                                         self.planning_policy.game_state['board']._cells[action_position].carbon))

        potential_action_list = sorted(potential_action_list, key=lambda x: (-x[2], x[3]), reverse=True)
        if len(potential_action_list) > 0:
            potential_action = potential_action_list[0][0]
            potential_action_position = potential_action_list[0][1]
            if potential_action == None and target_position == action_position:
                pass
            elif potential_action == None and len(potential_action_list) > 1 and potential_action_list[1][2] == 0:
                potential_action = potential_action_list[1][0]
                potential_action_position = potential_action_list[1][1]                

        self.planning_policy.global_position_mask[potential_action_position] = 1
        return  potential_action


class CollectorGoToAndCollectCarbonPlan(CollectorPlan):
    def __init__(self, source_agent, target, planning_policy):
        super().__init__(source_agent, target, planning_policy)
        self.calculate_score()
    
    def check_validity(self):
        if self.planning_policy.config['enabled_plans'][
                'CollectorGoToAndCollectCarbonPlan']['enabled'] == False:
            return False
        else:
        #类型不对
            if not isinstance(self.source_agent, Collector):
                return False
            if not isinstance(self.target, Cell):
                return False
            if self.target.tree is not None:
                return False
            if self.source_agent.carbon > self.planning_policy.config['collector_config']['gohomethreshold']:
                return False
            center_position = self.planning_policy.game_state['our_player'].recrtCenters[0].position
            source_posotion = self.source_agent.position
            source_center_distance = self.planning_policy.get_distance(
                source_posotion[0], source_posotion[1], center_position[0],
                center_position[1])
            if source_center_distance >= 300 - self.planning_policy.game_state['board'].step - 4:
                return False           
        return True

    def calculate_score(self):
        if self.check_validity() == False:
            self.preference_index = self.planning_policy.config[
                'mask_preference_index']
        else:
            source_posotion = self.source_agent.position
            target_position = self.target.position
            distance = self.planning_policy.get_distance(
                source_posotion[0], source_posotion[1], target_position[0],
                target_position[1])

            self.preference_index = get_cell_carbon_after_n_step(self.planning_policy.game_state['board'], 
                                                                self.target.position,
                                                                distance) / (distance + 1)
            
    
    def translate_to_action(self):
        return super().translate_to_action()

class CollectorGoToAndGoHomeWithCollectCarbonPlan(CollectorPlan):
    def __init__(self, source_agent, target, planning_policy):
        super().__init__(source_agent, target, planning_policy)
        self.calculate_score()
    
    def check_validity(self):
        if self.planning_policy.config['enabled_plans'][
                'CollectorGoToAndGoHomeWithCollectCarbonPlan']['enabled'] == False:
            return False
        else:
        #类型不对
            if not isinstance(self.source_agent, Collector):
                return False
            if not isinstance(self.target, Cell):
                return False
            if self.target.tree is not None:
                return False
            if self.source_agent.carbon <= self.planning_policy.config['collector_config']['gohomethreshold']:
                return False
            center_position = self.planning_policy.game_state['our_player'].recrtCenters[0].position
            source_posotion = self.source_agent.position
            source_center_distance = self.planning_policy.get_distance(
                source_posotion[0], source_posotion[1], center_position[0],
                center_position[1])
            if source_center_distance >= 300 - self.planning_policy.game_state['board'].step - 4:
                return False
        return True

    def calculate_score(self):
        if self.check_validity() == False:
            self.preference_index = self.planning_policy.config[
                'mask_preference_index']
        else:
            source_posotion = self.source_agent.position
            target_position = self.target.position

            center_position = self.planning_policy.game_state['our_player'].recrtCenters[0].position
            target_center_distance = self.planning_policy.get_distance(
                center_position[0], center_position[1], target_position[0],
                target_position[1])
            
            source_target_distance = self.planning_policy.get_distance(
                source_posotion[0], source_posotion[1], target_position[0],
                target_position[1])
            
            source_center_distance = self.planning_policy.get_distance(
                source_posotion[0], source_posotion[1], target_position[0],
                target_position[1])

            if target_center_distance + source_target_distance == source_center_distance:
                self.preference_index = get_cell_carbon_after_n_step(self.planning_policy.game_state['board'], 
                                                                    self.target.position,
                                                                    source_target_distance) / (source_target_distance + 1) + 100
            else:
                self.preference_index = get_cell_carbon_after_n_step(self.planning_policy.game_state['board'], 
                                                                    self.target.position,
                                                                    source_target_distance) / (source_target_distance + 1) - 100

    def translate_to_action(self):
        return super().translate_to_action() 

class CollectorGoToAndGoHomePlan(CollectorPlan):
    def __init__(self, source_agent, target, planning_policy):
        super().__init__(source_agent, target, planning_policy)
        self.calculate_score()
    
    def check_validity(self):
        if self.planning_policy.config['enabled_plans'][
                'CollectorGoToAndGoHomePlan']['enabled'] == False:
            return False
        else:
        #类型不对
            if not isinstance(self.source_agent, Collector):
                return False
            if not isinstance(self.target, Cell):
                return False
            if self.target.tree is not None:
                return False
            if self.source_agent.carbon <= self.planning_policy.config['collector_config']['gohomethreshold']:
                return False

            # 与转化中心距离大于1
            center_position = self.planning_policy.game_state['our_player'].recrtCenters[0].position
            source_position = self.source_agent.position
            if self.planning_policy.get_distance(
                source_position[0], source_position[1], center_position[0],
                center_position[1]) > 1:
                return False
            # target 不是转化中心
            target_position = self.target.position
            if target_position[0] != center_position[0] or target_position[1] != center_position[1]:
                return False

            
            
        return True

    def calculate_score(self):
        if self.check_validity() == False:
            self.preference_index = self.planning_policy.config[
                'mask_preference_index']
        else:
            self.preference_index = 10000

    def translate_to_action(self):
        if not self.can_action(self.target.position):
            self.planning_policy.global_position_mask[self.source_agent.position] = 1
            return None
        else:
            self.planning_policy.global_position_mask[self.target.position] = 1
        for move in WorkerAction.moves():
            new_position = self.source_agent.cell.position + move.to_point()
            if new_position[0] == self.target.position[0] and new_position[1] == self.target.position[1]:
                return move 


class CollectorRushHomePlan(CollectorPlan):
    def __init__(self, source_agent, target, planning_policy):
        super().__init__(source_agent, target, planning_policy)
        self.calculate_score()
    
    def check_validity(self):
        if self.planning_policy.config['enabled_plans'][
                'CollectorRushHomePlan']['enabled'] == False:
            return False
        else:
        #类型不对
            if not isinstance(self.source_agent, Collector):
                return False
            if not isinstance(self.target, Cell):
                return False
            if self.target.tree is not None:
                return False

            center_position = self.planning_policy.game_state['our_player'].recrtCenters[0].position
            source_posotion = self.source_agent.position
            source_center_distance = self.planning_policy.get_distance(
                source_posotion[0], source_posotion[1], center_position[0],
                center_position[1])

            if self.target.position[0] != center_position[0] or \
                self.target.position[1] != center_position[1]:
                return False
            if self.source_agent.carbon <= 10:
                return False

            if source_center_distance < 300 - self.planning_policy.game_state['board'].step - 5:
                return False
            
        return True

    def calculate_score(self):
        if self.check_validity() == False:
            self.preference_index = self.planning_policy.config[
                'mask_preference_index']
        else:
            self.preference_index = 5000

    def translate_to_action(self):
        return super().translate_to_action() 

class PlanningPolicy(BasePolicy):
    '''
    这个版本的机器人只能够发出两种指令:
    1. 基地招募种树者
       什么时候招募:  根据场上种树者数量、树的数量和现金三个维度进行加权判断。

    2. 种树者走到一个地方种树
       什么时候种: 一直种
       去哪种: 整张地图上碳最多的位置
    '''
    def __init__(self):
        super().__init__()
        #这里是策略的晁灿
        self.config = {
            # 表示我们的策略库中有多少可使用的策略
            'enabled_plans': {
                # 基地 招募种树员计划
                # enabled 为 true 表示运行时会考虑该策略
                # 以下plan同理
                'SpawnPlanterPlan': {
                    'enabled': True,
                    'planter_count_weight':-8,
                    'collector_count_weight':2,
                    # 'cash_weight':2,
                    # 'constant_weight':,
                    # 'denominator_weight':
                },
                # 基地 招募捕碳员计划
                'SpawnCollectorPlan': {
                    'enabled': True,
                    'planter_count_weight':8,
                    'collector_count_weight':-2,
                    # 'cash_weight':2,
                    # 'constant_weight':,
                    # 'denominator_weight':
                },
                # 种树员 抢树计划
                'PlanterRobTreePlan': {
                    'enabled': True,
                    'cell_carbon_weight': 1,
                    'cell_distance_weight': -7
                },
                # 种树员 种树计划
                'PlanterPlantTreePlan': {
                    'enabled': False,
                    'cell_carbon_weight': 50,  # cell碳含量所占权重
                    'cell_distance_weight': -40,  # 与目标cell距离所占权重
                    'enemy_min_distance_weight': 50,  # 与敌方worker最近距离所占权重
                    'tree_damp_rate': 0.08,  # TODO: 这个系数是代表什么？
                    'distance_damp_rate': 0.999,
                    'fuzzy_value': 2,
                    'carbon_growth_rate': 0.05
                },
                #Collector plans
                # 捕碳者去全地图score最高的地方采集碳的策略
                'CollectorGoToAndCollectCarbonPlan': {
                    'enabled': True
                },
                # 捕碳者碳携带的数量超过阈值后，打算回家，并且顺路去score高的地方采集碳
                'CollectorGoToAndGoHomeWithCollectCarbonPlan': {
                    'enabled': True
                },
                # 捕碳者碳携带的数量超过阈值并且与家距离为1，那么就直接回家
                'CollectorGoToAndGoHomePlan': {
                    'enabled': True
                },
                # 捕碳者根据与家的距离和剩余回合数，判断是否应该立刻冲回家送碳
                'CollectorRushHomePlan': {
                    'enabled': True
                }
            },
            # 捕碳者相关超参
            'collector_config': {
                # 回家阈值
                'gohomethreshold': 100,
            },
            # 地图大小
            'row_count': 15,
            'column_count': 15,
            # 把执行不了的策略的score设成该值（-inf）
            'mask_preference_index': -1e9
        }
        #存储游戏中的状态，配置
        self.game_state = {
            'board': None,
            'observation': None,
            'configuration': None,
            'our_player': None,  #carbon.helpers.Player class from board field
            'opponent_player':
            None  #carbon.helpers.Player class from board field
        }

    #get Chebyshev distance of two positions, x mod self.config['row_count] ,y
    #mod self.config['column_count]
    def get_distance(self, x1, y1, x2, y2):
        x_1_to_2= (x1 - x2 +
                self.config['row_count']) % self.config['row_count'] 
        y_1_to_2= (
                    y1 - y2 +
                    self.config['column_count']) % self.config['column_count']
        dis_x = min(self.config['row_count'] - x_1_to_2 , x_1_to_2)
        dis_y = min(self.config['column_count'] - y_1_to_2 , y_1_to_2)
        return dis_x + dis_y

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
            return actions[command_value].name if 0 < command_value < len(
                actions) else None

        env_commands = {}
        for agent_id, cmd_value in policy_actions.items():
            command = agent_action(agent_id, cmd_value)
            if command is not None:
                env_commands[agent_id] = command
        return env_commands

    #计算出所有合法的Plan
    def make_possible_plans(self):
        plans = []
        board = self.game_state['board']
        for cell_id, cell in board.cells.items():
            # iterate over all collectors planters and recrtCenter of currnet
            # player
            for collector in self.game_state['our_player'].collectors:
                plan = (CollectorGoToAndCollectCarbonPlan(
                    collector, cell, self))
                plans.append(plan)
                plan = (CollectorGoToAndGoHomeWithCollectCarbonPlan(
                    collector, cell, self))
                plans.append(plan)
                plan = (CollectorGoToAndGoHomePlan(
                    collector, cell, self))
                plans.append(plan)

                plan = (CollectorRushHomePlan(
                    collector, cell, self))
                plans.append(plan)

            for planter in self.game_state['our_player'].planters:
                plan = (PlanterRobTreePlan(
                    planter, cell, self))

                plans.append(plan)
                plan = (PlanterPlantTreePlan(
                    planter, cell, self))
                plans.append(plan)

            for recrtCenter in self.game_state['our_player'].recrtCenters:
                #TODO:动态地load所有的recrtCenterPlan类
                plan = SpawnPlanterPlan(recrtCenter, cell, self)
                plans.append(plan)
                plan = SpawnCollectorPlan(recrtCenter, cell, self)
                plans.append(plan)
            pass
        pass
        plans = [
            plan for plan in plans
            if plan.preference_index != self.config['mask_preference_index'] and plan.preference_index > 0
        ]
        return plans

    #把Board,Observation,Configuration变量的信息存到PlanningPolicy中
    def parse_observation(self, observation, configuration):
        self.game_state['observation'] = observation
        self.game_state['configuration'] = configuration
        self.game_state['board'] = Board(observation, configuration)
        self.game_state['our_player'] = self.game_state['board'].players[
            self.game_state['board'].current_player_id]
        self.game_state['opponent_player'] = self.game_state['board'].players[
            1 - self.game_state['board'].current_player_id]

    #从合法的Plan中为每一个Agent选择一个最优的Plan
    def possible_plans_to_plans(self, possible_plans: BasePlan):
        #TODO:解决plan之间的冲突,比如2个种树者要去同一个地方种树，现在的plan选择
        #方式是不解决冲突
        source_agent_id_plan_dict = {}
        possible_plans = sorted(possible_plans, key=lambda x: x.preference_index, reverse=True)
        
        collector_cell_plan = dict()
        planter_cell_plan = dict()
        
        # 去转化中心都不冲突x
        center_position = self.game_state['our_player'].recrtCenters[0].position
        collector_cell_plan[center_position] = -100

        for possible_plan in possible_plans:
            if possible_plan.source_agent.id in source_agent_id_plan_dict:
                continue
            if isinstance(possible_plan.source_agent, Collector):
                if collector_cell_plan.get(possible_plan.target.position, 0) > 0:
                    continue
                collector_cell_plan[possible_plan.target.position] = collector_cell_plan.get(possible_plan.target.position, 1)
                source_agent_id_plan_dict[
                    possible_plan.source_agent.id] = possible_plan    
            elif isinstance(possible_plan.source_agent, Planter):
                if planter_cell_plan.get(possible_plan.target.position, 0) > 0:
                    continue
                planter_cell_plan[possible_plan.target.position] = planter_cell_plan.get(possible_plan.target.position, 1)
                source_agent_id_plan_dict[
                    possible_plan.source_agent.id] = possible_plan             
            else:
                source_agent_id_plan_dict[
                    possible_plan.source_agent.id] = possible_plan
        #print(source_agent_id_plan_dict)
        #for s, t in source_agent_id_plan_dict.items():
        #    print(s, t.target.position)
        return source_agent_id_plan_dict.values()

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

    #被上层调用的函数
    #所有规则为这个函数所调用
    def take_action(self, observation, configuration):
        self.global_position_mask = dict()
                            
        self.parse_observation(observation, configuration)
        possible_plans = self.make_possible_plans()
        plans = self.possible_plans_to_plans(possible_plans)

        # print(command)
        # 这个地方返回一个cmd字典
        # 类似这样
        """
        {'player-0-recrtCenter-0': 'RECPLANTER', 'player-0-worker-0': 'RIGHT', 'player-0-worker-5': 'DOWN', 'player-0-worker-6': 'DOWN', 'player-0-worker-7': 'RIGHT', 'player-0-worker-8': 'UP', 'player-0-worker-12': 'UP', 'player-0-worker-13': 'UP'}
        """
        def remove_none_action_actions(plan_action_dict):
            return {
                k: v['action'].value
                for k, v in plan_action_dict.items() if v['action'] is not None
            }

        plan_dict = {
            plan.source_agent.id: {
                'action': plan.translate_to_action(),
                'plan': plan
            }
            for plan in plans
        }
        clean_plan_id_action_value_dict = remove_none_action_actions(plan_dict)
        command_list = self.to_env_commands(clean_plan_id_action_value_dict)
        #print(command_list)
        return command_list
