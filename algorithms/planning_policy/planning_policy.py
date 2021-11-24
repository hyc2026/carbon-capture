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
from zerosum_env.envs.carbon.helpers import (Board, Cell, Collector, Planter,
                                             Point, RecrtCenter,
                                             RecrtCenterAction, WorkerAction)


class BasePlan(ABC):
    #这里的source_agent,target都是对象，而不是字符串
    #source: collector,planter,recrtCenter
    #target: collector,planter,recrtCenter,cell
    def __init__(self, source_agent, target, planning_policy):
        self.source_agent = source_agent
        self.target = target
        self.planning_policy = planning_policy
        self.preference_index = None

    @abstractmethod
    def translate_to_action(self):
        pass


class RecrtCenterPlan(BasePlan):
    def __init__(self, source_agent, target, planning_policy):
        super().__init__(source_agent, target, planning_policy)


#CUSTOM:根据策略随意修改
class RecrtCenterSpawnPlanterPlan(RecrtCenterPlan):
    def __init__(self, source_agent, target, planning_policy):
        super().__init__(source_agent, target, planning_policy)
        self.calculate_score()

    #CUSTOM:根据策略随意修改
    #计算转化中心生产种树者的倾向分数
    #当前策略是返回PlanningPolicy中设定的固定值或者一个Mask(代表关闭，值为负无穷)
    def calculate_score(self):
        #is valid
        if self.check_validity() == False:
            self.preference_index = self.planning_policy.config[
                'mask_preference_index']
        else:
            self.preference_index = self.planning_policy.config[
                'enabled_plans']['RecrtCenterSpawnPlanterPlan'][
                    'preference_factor']

    def check_validity(self):
        #没有开启
        if self.planning_policy.config['enabled_plans'][
                'RecrtCenterSpawnPlanterPlan']['enabled'] == False:
            return False
        #类型不对
        if not isinstance(self.source_agent, RecrtCenter):
            return False
        if not isinstance(self.target, Cell):
            return False

        #位置不对
        if self.source_agent.cell != self.target:
            return False
        #钱不够
        if self.planning_policy.game_state[
                'our_player'].cash < self.planning_policy.game_state[
                    'configuration']['recPlanterCost']:
            return False
        #数量达到上限
        if self.planning_policy.game_state['our_player'].planters.__len__(
        ) >= self.planning_policy.game_state['configuration']['planterLimit']:
            return False
        return True

    #暂时还没发现这个action有什么用，感觉直接用command就行了
    def translate_to_action(self):
        return RecrtCenterAction.RECPLANTER


class PlanterPlan(BasePlan):
    def __init__(self, source_agent, target, planning_policy):
        super().__init__(source_agent, target, planning_policy)

    def check_valid(self):
        yes_it_is = isinstance(self.source_agent, Planter)
        return yes_it_is

    def get_actual_plant_cost(self):
        configuration = self.planning_policy.game_state['configuration']

        if len(self.planning_policy.game_state['our_player'].tree_ids) == 0:
            return configuration.recPlanterCost
        else:
            return configuration.recPlanterCost + configuration.plantCostInflationRatio * configuration.plantCostInflationBase**self.planning_policy.game_state[
                'board'].trees.__len__()

    def get_tree_absorb_carbon_speed_at_cell(self,cell:Cell):
        pass
        

#CUSTOM:根据策略随意修改
class PlanterGoToAndPlantTreeAtTreeAtPlan(PlanterPlan):
    def __init__(self, source_agent, target, planning_policy):
        super().__init__(source_agent, target, planning_policy)
        self.calculate_score()

    #CUSTOM:根据策略随意修改
    #计算转化中心生产种树者的倾向分数
    #当前策略是返回PlanningPolicy中设定的固定值或者一个Mask(代表关闭，值为负无穷)
    def calculate_score(self):
        #is valid
        if self.check_validity() == False:
            self.preference_index = self.planning_policy.config[
                'mask_preference_index']
        else:
            source_posotion = self.source_agent.position
            target_position = self.target.position
            distance = self.planning_policy.get_distance(
                source_posotion[0], source_posotion[1], target_position[0],
                target_position[1])

            self.preference_index = self.target.carbon * self.planning_policy.config[
                'enabled_plans']['PlanterGoToAndPlantTreeAtTreeAtPlan'][
                    'cell_carbon_weight'] + distance * self.planning_policy.config[
                        'enabled_plans']['PlanterGoToAndPlantTreeAtTreeAtPlan'][
                            'cell_distance_weight']


    def check_validity(self):
        #没有开启
        if self.planning_policy.config['enabled_plans'][
                'PlanterGoToAndPlantTreeAtTreeAtPlan']['enabled'] == False:
            return False
        #类型不对
        if not isinstance(self.source_agent, Planter):
            return False
        if not isinstance(self.target, Cell):
            return False
        if self.target.tree is not None:
            return False

        #钱不够
        if self.planning_policy.game_state[
                'our_player'].cash < self.get_actual_plant_cost():
            return False
        return True

    #暂时还没发现这个action有什么用，感觉直接用command就行了
    def translate_to_action(self):
        if self.source_agent.cell == self.target:
            return None
        else:
            old_position = self.source_agent.cell.position
            old_distance = self.planning_policy.get_distance(
                old_position[0], old_position[1], self.target.position[0],
                self.target.position[1])

            for move in WorkerAction.moves():
                new_position = self.source_agent.cell.position + move.to_point(
                )
                new_distance = self.planning_policy.get_distance(
                    new_position[0], new_position[1], self.target.position[0],
                    self.target.position[1])

                if new_distance < old_distance:
                    return move


class PlanningPolicy(BasePolicy):
    '''
    这个版本的机器人只能够发出两种指令:
    1. 基地招募种树者
       什么时候种: 钱多树多种树者少(这种情况下资金不能得到有效利用)。 cash > 5
       *( 3 *
       tree_count - 2 * planter_count)
       planter_count * 

    2. 种树者走到一个地方种树
       什么时候种: 一直种
       去哪种: 整张地图上碳最多的位置
    '''

    #输入:
    def __init__(self):
        super().__init__()
        self.config = {
            'enabled_plans': {
                #recrtCenter plans
                'RecrtCenterSpawnPlanterPlan': {
                    'enabled': True,
                    'preference_factor': 100
                },
                #Planter plans
                'PlanterGoToAndPlantTreeAtTreeAtPlan': {
                    'enabled': True,
                    'cell_carbon_weight': 1,
                    'cell_distance_weight': -7
                }
            },
            'row_count': 15,
            'column_count': 15,
            'mask_preference_index': -1e9
        }
        self.game_state = object()
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
        return (x1 - x2 +
                self.config['row_count']) % self.config['row_count'] + (
                    y1 - y2 +
                    self.config['column_count']) % self.config['column_count']

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

    def make_possible_plans(self):
        plans = []
        board = self.game_state['board']
        for cell_id, cell in board.cells.items():
            # iterate over all collectors planters and recrtCenter of currnet
            # player
            for collector in self.game_state['our_player'].collectors:
                pass

            for planter in self.game_state['our_player'].planters:
                plan = (PlanterGoToAndPlantTreeAtTreeAtPlan(
                    planter, cell, self))
                plans.append(plan)
            for recrtCenter in self.game_state['our_player'].recrtCenters:
                #TODO:动态地load所有的recrtCenterPlan类
                plan = RecrtCenterSpawnPlanterPlan(recrtCenter, cell, self)
                plans.append(plan)
            pass
        pass
        plans = [
            plan for plan in plans
            if plan.preference_index != self.config['mask_preference_index']
        ]
        return plans

    def parse_observation(self, observation, configuration):
        self.game_state['observation'] = observation
        self.game_state['configuration'] = configuration
        self.game_state['board'] = Board(observation, configuration)
        self.game_state['our_player'] = self.game_state['board'].players[
            self.game_state['board'].current_player_id]
        self.game_state['opponent_player'] = self.game_state['board'].players[
            1 - self.game_state['board'].current_player_id]

    def possible_plans_to_plans(self, possible_plans: BasePlan):
        #TODO:解决plan之间的冲突,比如2个种树者要去同一个地方种树，现在的plan选择
        #方式是不解决冲突
        source_agent_id_plan_dict={}
        for possible_plan in possible_plans:
            if possible_plan.source_agent.id not in source_agent_id_plan_dict: 
                source_agent_id_plan_dict[possible_plan.source_agent.id] = possible_plan
            else:
                if source_agent_id_plan_dict[possible_plan.source_agent.id].preference_index < possible_plan.preference_index:
                    source_agent_id_plan_dict[possible_plan.source_agent.id] = possible_plan
        return source_agent_id_plan_dict.values()

    def take_action(self, observation, configuration):
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
                k: enum_v.value
                for k, enum_v in plan_action_dict.items() if enum_v is not None
            }

        plan_action_dict = {
            plan.source_agent.id: plan.translate_to_action()
            for plan in plans
        }
        clean_plan_action_value_dict = remove_none_action_actions(
            plan_action_dict)
        plan_action_dict = remove_none_action_actions(plan_action_dict)

        command_list = self.to_env_commands(plan_action_dict)
        return command_list
