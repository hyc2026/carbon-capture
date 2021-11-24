import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from icecream import ic
from typing import Tuple, Dict, OrderedDict
import copy

import torch
from torch.distributions import Categorical

from zerosum_env.envs.carbon.helpers import Board, Point

from algorithms.model import Model
from algorithms.base_policy import BasePolicy
from envs.obs_parser_xinnian import ObservationParser
from utils.utils import to_tensor

from zerosum_env.envs.carbon.helpers import Planter, RecrtCenter, Collector


class BasePlan():
    #这里的source_agent,target都是对象，而不是字符串
    #source: collector,planter,recrtCenter

    #target: collector,planter,recrtCenter,cell
    def __init__(self, source_agent, target,
                 planning_policy):
        self.source_agent = source_agent
        self.target = target
        self.planning_policy = planning_policy
        self.preference_index = None
    
    def translate_to_actions(self):
        pass

    @classmethod
    def resolve_plan_conflict(plans):
        pass


class RecrtCenterPlan(BasePlan):
    def __init__(self, source_agent,  target,planning_policy):
        super().__init__(source_agent,  target,planning_policy)

    def check_valid(self):
        yes_it_is = isinstance(self.source_agent, RecrtCenter)
        return yes_it_is


class PlanterPlan(BasePlan):
    def __init__(self, source_agent,  target,planning_policy):
        super().__init__(source_agent,  target,planning_policy)

    def check_source_agent_is_planter(self):
        yes_it_is = issubclass(self.source_agent, Planter)
        return yes_it_is


#CUSTOM:根据策略随意修改；但要注意最好
class RecrtCenterSpawnPlanterPlan(RecrtCenterPlan):
    def __init__(self, source_agent,  target,planning_policy):
        super().__init__(source_agent, target,planning_policy)
        self.calculate_score()



    #CUSTOM:根据策略随意修改
    #计算转化中心生产种树者的倾向分数
    #当前策略是返回PlanningPolicy中设定的固定值或者一个Mask(代表关闭，值为负无穷)
    def calculate_score(self):
        #is valid
        if self.check_validity() == False:
            return self.planning_policy.config['mask_preference_index']
        else:
            return self.planning_policy.config['enabled_plans'][
                'RecrtCenterSpawnPlanterPlan']['preference_factor']

    def check_validity(self):
        if self.planning_policy.config['enabled_plans'][
                'RecrtCenterSpawnPlanterPlan']['enabled'] == False:
            return False
        if not super().check_valid():
            return False
        if self.planning_policy.game_state[
                'our_player'].cash < self.planning_policy.game_state[
                    'configuration']['recPlanterCost']:
            return False
        if self.planning_policy.game_state['our_player'].planters.__len__() >= self.planning_policy.game_state[
                    'configuration']['planterLimit']:
            return False
        return True

    def translate_to_actions(self, planning_policy):

        pass


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

    def make_plans(self):
        plans=[]
        board = self.game_state['board']
        for cell_id, cell in board.cells.items():
            # iterate over all collectors planters and recrtCenter of currnet
            # player
            for worker_id,worker in board.collectors.items():
                pass
            for recrtCenter_id,recrtCenter in board.recrtCenters.items():
                #TODO:动态地load所有的recrtCenterPlan类
                plans.append(RecrtCenterSpawnPlanterPlan(recrtCenter,cell,self))
                pass
        return None

    def parse_observation(self, observation, configuration):
        self.game_state['observation'] = observation
        self.game_state['configuration'] = configuration
        self.game_state['board'] = Board(observation, configuration)
        self.game_state['our_player'] = self.game_state['board'].players[
            self.game_state['board'].current_player_id]
        self.game_state['opponent_player'] = self.game_state['board'].players[
            1 - self.game_state['board'].current_player_id]

    def take_action(self, observation, configuration):
        self.parse_observation(observation, configuration)
        plans = self.make_plans()
        plans = BasePlan.resolve_plan_conflict(plans)

        # print(command)
        # 这个地方返回一个cmd字典
        # 类似这样
        """
        {'player-0-recrtCenter-0': 'RECPLANTER', 'player-0-worker-0': 'RIGHT', 'player-0-worker-5': 'DOWN', 'player-0-worker-6': 'DOWN', 'player-0-worker-7': 'RIGHT', 'player-0-worker-8': 'UP', 'player-0-worker-12': 'UP', 'player-0-worker-13': 'UP'}
        """
        return command
