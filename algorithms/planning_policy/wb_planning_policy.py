from os import cpu_count
import sys
import numpy as np

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import copy
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from zerosum_env.envs.carbon.helpers import (Board, Cell, Collector, Planter,
                                             Point, RecrtCenter, Worker,
                                             RecrtCenterAction, WorkerAction)
from random import randint, shuffle, choice

TOP_CARBON_CONTAIN = 5

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

Action2Direction = {None: np.array((0, 0)),
                    WorkerAction.UP: np.array((0, 1)),
                    WorkerAction.RIGHT: np.array((1, 0)),
                    WorkerAction.DOWN: np.array((0, -1)),
                    WorkerAction.LEFT: np.array((-1, 0))
                    }


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


class PlanterAct(AgentBase):
    def __init__(self):
        super().__init__()
        self.workaction = WorkerAction
        self.planter_target = dict()

    @ staticmethod
    def _minimum_distance(point_1, point_2):
        abs_distance = abs(point_1 - point_2)
        cross_distance = min(point_1, point_2) + (15 - max(point_1, point_2))
        return min(abs_distance, cross_distance)

    def _calculate_distance(self, planter_position, current_position):
        """计算真实距离，计算跨图距离，取两者最小值"""
        x_distance = self._minimum_distance(planter_position[0], current_position[0])
        y_distance = self._minimum_distance(planter_position[1], current_position[1])

        return x_distance + y_distance

    def _target_plan(self, planter: Planter, carbon_sort_dict: Dict):
        # 什么时候种树，敌方树多少，我方树多少，地图上的碳
        # 我方树在N棵以内的时候，我方种树
        # 敌方有树，优先抢敌方的树
        # 我方树大于N棵，敌方没有树，我方种树员停留在树龄最小的树上保护树

        planter_position = planter.position
        # 取碳排量最高的前n

        # if True:  # 我方树在N棵以内的时候，我方优先种树
        #     pass
        # elif False:   # 我方树大于N棵的时候，优先抢敌方的树
        #     pass
        # elif False:   # 我方树大于M棵的时候，停止种树，开始守护树
        #     pass

        carbon_sort_dict_top_n = \
            {_v: _k for _i, (_v, _k) in enumerate(carbon_sort_dict.items()) if _i < TOP_CARBON_CONTAIN}  # 只选取含碳量top_n的cell来进行计算，拿全部的cell可能会比较耗时？
        # 计算planter和他的相对距离，并且结合该位置四周碳的含量，得到一个总的得分
        planned_target = [Point(*_v.position) for _k, _v in self.planter_target.items()]
        max_score, max_score_cell = -1e9, None
        for _cell, _carbon_sum in carbon_sort_dict_top_n.items():
            if (_cell.tree is None) and (_cell.position not in planned_target) and (_cell.recrtCenter is None):  # 这个位置没有树，且这个位置不在其他智能体正在进行的plan中, 且这个位置不能是基地
                planter_to_cell_distance = self._calculate_distance(planter_position, _cell.position)  # 我们希望这个距离越小越好
                target_preference_score = 0 * _carbon_sum + np.log(1 / (planter_to_cell_distance + 1e-9))  # 不考虑碳总量只考虑距离 TODO: 这会导致中了很多树，导致后期花费很高

                if target_preference_score > max_score:
                    max_score = target_preference_score
                    max_score_cell = _cell

        if max_score_cell is None:  # 没有找到符合条件的最大得分的cell，随机选一个cell
            max_score_cell = choice(list(carbon_sort_dict_top_n))

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
        # 
        """
        TODO: 需要改进的地方：
        1 避免碰撞：平常走路，从基地出来
        2 抢树
        """

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
                    self.planter_target.pop(planter.id)
                    # TODO: 这个地方还需要加入一个判断：当前是否有现金种树，如果没有种树员得离开这个地方，
                    # 不然待着不动，容易被人干掉
                else:  # 没有执行完接着执行

                    old_position = planter.position
                    target_position = self.planter_target[planter.id].position
                    old_distance = self._calculate_distance(old_position, target_position)

                    for move in WorkerAction.moves():
                        new_position = old_position + move.to_point()   # TODO: 考虑地图跨界情况
                        new_position = str(new_position).replace("15", "0")
                        new_position = Point(*eval(new_position.replace("-1", "14")))
                        new_distance = self._calculate_distance(new_position, target_position)

                        if new_distance < old_distance:
                            if self._check_surround_validity(move, planter):
                                move_action_dict[planter.id] = move.name
                            else:   # 随机移动，不要静止不动或反向移动，否则当我方多个智能体相遇会卡主
                                if move.name == 'UP':
                                    move_action_dict[planter.id] = choice(["DOWN", "RIGHT", "LEFT"])
                                elif move.name == 'DOWN':
                                    move_action_dict[planter.id] = choice(["UP", "RIGHT", "LEFT"])
                                elif move.name == 'RIGHT':
                                    move_action_dict[planter.id] = choice(["UP", "DOWN", "LEFT"])
                                elif move.name == 'LEFT':
                                    move_action_dict[planter.id] = choice(["UP", "DOWN", "RIGHT"])

        return move_action_dict

def get_cell_carbon_after_n_step(board: Board, position: Point, n: int) -> float:
    # 计算position这个位置的碳含量在n步之后的预估数值（考虑上下左右四个位置的树的影响）
    danger_zone = []
    x_left = position.x - 1 if position.x > 0 else 14
    x_right = position.x + 1 if position.x < 14 else 0
    y_up = position.y - 1 if position.y > 0 else 14
    y_down = position.y + 1 if position.y < 14 else 0
    # 上下左右4个格子
    danger_zone.append(Point(position.x, y_up))
    danger_zone.append(Point(x_left, position.y))
    danger_zone.append(Point(x_right, position.y))
    danger_zone.append(Point(position.x, y_down))
    
    start = 0
    target_cell = board.cells[position]
    c = target_cell.carbon
    if n == 0:
        return c
    
    # position的位置有树，直接从树暴毙之后开始算
    if target_cell.tree is not None:
        start = 50 - target_cell.tree.age + 1
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
        source_position = self.source_agent.position
        target_position = self.target.position
        distance = self.planning_policy.get_distance(
            source_position[0], source_position[1], target_position[0],
            target_position[1])
        return distance

    def get_total_carbon(self, distance=0):
        target_carbon_expect = 0
        for c in [self.target.up, self.target.left, self.target.right, self.target.down]:
            target_carbon_expect += get_cell_carbon_after_n_step(self.planning_policy.game_state['board'],
                                                        c.position,
                                                        distance + 1)        
        return target_carbon_expect

    
    # 根据未来走过去的步数n，计算n步之后目标位置的碳含量
    # TODO: 这个函数有问题，已经有新的函数了，这个可以删除
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
                if worker:
                    if worker.player_id == self.source_agent.player_id:
                        continue
                    return False
            
            return True
        else:
            return False

    def get_actual_plant_cost(self):
        configuration = self.planning_policy.game_state['configuration']

        if len(self.planning_policy.game_state['our_player'].tree_ids) == 0:
            return configuration.recPlanterCost
        else:
            # 当该玩家有树的时候，种树的成本会越来越高
            # i = configuration.plantCostInflationBase  # 1.235
            # j = configuration.plantCostInflationRatio  # 5
            return configuration.recPlanterCost + configuration.plantCostInflationRatio * configuration.plantCostInflationBase**self.planning_policy.game_state[
                'board'].trees.__len__()

    def get_random_direction(self, waiting_list):
        old_position = self.source_agent.cell.position
        shuffle(waiting_list)
        for action in waiting_list:
            new_position = (
                (Action2Direction[action][0] + old_position[0]+ self.planning_policy.config['row_count']) % self.planning_policy.config['row_count'],
                (Action2Direction[action][1] + old_position[1]+ self.planning_policy.config['column_count']) % self.planning_policy.config['column_count'],
            )
            if self.can_action(new_position):
                return action, new_position
        return None, old_position



    def translate_to_action_first(self):
        # 如果当前已经到达目标单元格，并且满足移动条件
        if self.source_agent.cell == self.target:
            # self.planning_policy.global_position_mask[self.target.position] = 1
            # 站着不动（行为就是种树 / 抢树）
            if self.can_action(self.target.position):
                return None, self.target.position
            else:
                waiting_list = WorkerActions[1:] # 没有None
                return self.get_random_direction(waiting_list)
                
        else:
            old_position = self.source_agent.cell.position
            old_distance = self.planning_policy.get_distance(
                old_position[0], old_position[1], self.target.position[0],
                self.target.position[1])

            move_list = []

            for i, action in enumerate(WorkerActions):
                rand_factor = randint(0, 100)
                if action == None:
                    move_list.append((None, self.source_agent.cell.position, old_distance, rand_factor))
                else:
                    new_position = (
                        (WorkerDirections[i][0] + old_position[0]+ self.planning_policy.config['row_count']) % self.planning_policy.config['row_count'],
                        (WorkerDirections[i][1] + old_position[1]+ self.planning_policy.config['column_count']) % self.planning_policy.config['column_count'],
                    )
                    new_distance = self.planning_policy.get_distance(
                        new_position[0], new_position[1], self.target.position[0],
                        self.target.position[1])
                    
                    move_list.append((action, new_position, new_distance, rand_factor))

            move_list = sorted(move_list, key=lambda x: x[2: 4])

            for move, new_position, new_d, _ in move_list:
                if self.can_action(new_position):
                    # self.planning_policy.global_position_mask[new_position] = 1
                    return move, new_position
            return None, old_position

    def translate_to_action_second(self, cash):
        action, position = self.translate_to_action_first()
        if action is None:
            # 钱不够，要动起来，策略是随机动
            if self.planning_policy.game_state[
               'our_player'].cash < cash:
                waiting_list = copy.deepcopy(WorkerActions[1:])  # 重新copy一份 WorkerActions
                # waiting_list.append(None)  # 两倍的概率继续None
                action, position = self.get_random_direction(waiting_list)
                # for i, action in enumerate(waiting_list):
                #     if action is None:
                #         self.planning_policy.global_position_mask[position] = 1
                #         return None
                #     new_position = (
                #         (WorkerDirections[i][0] + old_position[0]+ self.planning_policy.config['row_count']) % self.planning_policy.config['row_count'],
                #         (WorkerDirections[i][1] + old_position[1]+ self.planning_policy.config['column_count']) % self.planning_policy.config['column_count'],
                #     )
                #     if self.can_action(new_position):
                #         self.planning_policy.global_position_mask[new_position] = 1
                #         return action
                # self.planning_policy.global_position_mask[position] = 1
        self.planning_policy.global_position_mask[position] = 1
        return action


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
            # 到这一步说明那个地方的树可以抢了
            distance = self.get_distance2target()
            total_carbon = self.get_total_carbon(distance)
            nearest_oppo_planter_distance = 10000
            age_can_use = min(50 - self.target.tree.age - distance - 1, nearest_oppo_planter_distance)
            self.preference_index = 2 * sum([total_carbon * (0.0375 ** i) for i in range(1, age_can_use + 1)])
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
        return self.translate_to_action_second(20 + 30)


class PlanterPlantTreePlan(PlanterPlan):
    def __init__(self, source_agent, target, planning_policy):
        super().__init__(source_agent, target, planning_policy)
        self.calculate_score()
        self.planning_policy = planning_policy
    
    def calculate_score(self):
        if self.check_validity() == False:
            self.preference_index = self.planning_policy.config[
                'mask_preference_index']
        else:
            distance2target = self.get_distance2target()
            my_id = self.source_agent.player_id  # 我方玩家id
            worker_dict = self.planning_policy.game_state['board'].workers  # 地图所有的 Planter & Collector
            cur_json = self.planning_policy.config['enabled_plans']['PlanterPlantTreePlan']

            tree_damp_rate = cur_json['tree_damp_rate']
            distance_damp_rate = cur_json['distance_damp_rate']
            fuzzy_value = cur_json['fuzzy_value']
            carbon_growth_rate =cur_json['carbon_growth_rate']

            board = self.planning_policy.game_state['board']
            total_predict_carbon = self.get_total_carbon(distance2target)
            # carbon_expectation = total_predict_carbon * (distance_damp_rate ** distance2target)
            carbon_expectation = total_predict_carbon
            self.preference_index = carbon_expectation

                    
    def translate_to_action(self):
        return self.translate_to_action_second(self.get_actual_plant_cost() + 30)

            
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
                    'enabled': True,
                    'cell_carbon_weight': 50,  # cell碳含量所占权重
                    'cell_distance_weight': -40,  # 与目标cell距离所占权重
                    'enemy_min_distance_weight': 50,  # 与敌方worker最近距离所占权重
                    'tree_damp_rate': 0.08,  # TODO: 这个系数是代表什么？
                    'distance_damp_rate': 0.999,  # 距离越远，其实越不划算，性价比衰减率
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
        self.planter = PlanterAct()

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

            # for planter in self.game_state['our_player'].planters:
            #     plan = (PlanterRobTreePlan(
            #         planter, cell, self))

            #     plans.append(plan)
            #     plan = (PlanterPlantTreePlan(
            #         planter, cell, self))
            #     plans.append(plan)

            for recrtCenter in self.game_state['our_player'].recrtCenters:
                #TODO:动态地load所有的recrtCenterPlan类
                plan = SpawnPlanterPlan(recrtCenter, cell, self)
                plans.append(plan)
                plan = SpawnCollectorPlan(recrtCenter, cell, self)
                plans.append(plan)
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
            # Planter 的计划不在这里实现
            elif isinstance(possible_plan.source_agent, Planter):
                pass
            else:
                source_agent_id_plan_dict[
                    possible_plan.source_agent.id] = possible_plan
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

    def plan2dict(self, plans):
        agent_id_2_action_number = {}
        for plan in plans:
            agent_id = plan.source_agent.id
            action = plan.translate_to_action()
            if action:
                agent_id_2_action_number[agent_id] = action.value
            else:
                agent_id_2_action_number[agent_id] = 0
        return agent_id_2_action_number
    
    #被上层调用的函数
    #所有规则为这个函数所调用
    def take_action(self, observation, configuration):
        self.global_position_mask = dict()
                            
        self.parse_observation(observation, configuration)
        
        possible_plans = self.make_possible_plans()
        plans = self.possible_plans_to_plans(possible_plans)

        ## 种树员的策略从这里开始吧，独立出来
        # 种树员做决策去哪里种树
        cur_board = self.game_state['board']
        ours, oppo = cur_board.current_player, cur_board.opponents
        planter_dict = self.planter.move(
            ours_info=ours,
            oppo_info=oppo,
            map_carbon_location=cur_board.cells,
            step=cur_board.step,
        )
        
        agent_id_2_action_number = self.plan2dict(plans)
        

        """
        agent_id_2_action_number:
        {'player-0-worker-0': 1, 'player-0-worker-3': 2, 'player-0-worker-5': 4, 'player-0-worker-7': 4, 'player-0-worker-9': 3, 'player-0-recrtCenter-0': 2}
        """
        command_list = self.to_env_commands(agent_id_2_action_number)
        #print(command_list)
        # print(command)
        # 这个地方返回一个cmd字典
        # 类似这样
        """
        {'player-0-recrtCenter-0': 'RECPLANTER', 'player-0-worker-0': 'RIGHT', 'player-0-worker-5': 'DOWN', 'player-0-worker-6': 'DOWN', 'player-0-worker-7': 'RIGHT', 'player-0-worker-8': 'UP', 'player-0-worker-12': 'UP', 'player-0-worker-13': 'UP'}
        """

        if planter_dict:
            command_list.update(planter_dict)


        return command_list