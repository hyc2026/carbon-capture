import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
import copy
from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, OrderedDict, Tuple

import torch
from algorithms.base_policy import BasePolicy
from algorithms.model import Model
from envs.obs_parser_xinnian import (BaseActions, ObservationParser,
                                     WorkerActions)
from icecream import ic
from torch.distributions import Categorical
from utils.utils import to_tensor
from zerosum_env.envs.carbon.helpers import (Board, Cell, Collector, Planter,
                                             Point, RecrtCenter,
                                             RecrtCenterAction)


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

    def check_valid(self):
        yes_it_is = isinstance(self.source_agent, RecrtCenter)
        return yes_it_is


#CUSTOM:根据策略随意修改；但要注意最好
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
        if not super().check_valid():
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


class PlanningPolicy(BasePolicy):

    #输入:
    def __init__(self):
        super().__init__()
        self.config = {
            'enabled_plans': {
                #recrtCenter plans
                'RecrtCenterSpawnPlanterPlan': {
                    'enabled': True,
                    'preference_factor': 100
                }
            },
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
            for worker_id, worker in board.collectors.items():
                pass
            for recrtCenter_id, recrtCenter in board.recrtCenters.items():
                #TODO:动态地load所有的recrtCenterPlan类
                plan = RecrtCenterSpawnPlanterPlan(recrtCenter, cell, self)
                if plan.preference_index != self.config[
                        'mask_preference_index']:
                    plans.append(plan)
                pass
            pass
        pass
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
        plans = []
        plan_source_agents = set()
        for possible_plan in possible_plans:
            if possible_plan.source_agent not in plan_source_agents:
                plans.append(possible_plan)
                plan_source_agents.add(possible_plan.source_agent)
        return plans

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
        command_list = self.to_env_commands({
            plan.source_agent.id: plan.translate_to_action().value
            for plan in plans
        })
        return command_list
