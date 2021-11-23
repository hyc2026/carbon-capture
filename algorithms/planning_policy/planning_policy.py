import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from typing import Tuple, Dict, OrderedDict
import copy

import torch
from torch.distributions import Categorical

from zerosum_env.envs.carbon.helpers import Board, Point

from algorithms.model import Model
from algorithms.base_policy import BasePolicy
from envs.obs_parser_xinnian import ObservationParser
from utils.utils import to_tensor





class PlanningPolicy(BasePolicy):
    
    #输入:
    def __init__(self):
        super().__init__()
        self.config = {

        }
        self.game_state = {
            'board':None,
            'observation':None,
            'configurations':None,
        }

        

    clas BasePlan():
        def __init__(self,source_agent,task,target):
            self.source_agent=source_agent
            self.task=task
            self.target=target
        def translate_to_actions(self,planning_polisy:BasePolicy):
            pass

        
    def make_plans():
        for 
    

    
    def parse_observation(self,observation,configuration):
        self.observation=observation
        self.configuration=configuration
        # 将obs转换为Board类从而更好获取信息
        self.board = Board(observation, configuration)


    def take_action(self, observation, configuration):
        self.parse_observation(observation,configuration)
        plans=self.make_plans()
        

        # print(command)
        # 这个地方返回一个cmd字典
        # 类似这样
        """
        {'player-0-recrtCenter-0': 'RECPLANTER', 'player-0-worker-0': 'RIGHT', 'player-0-worker-5': 'DOWN', 'player-0-worker-6': 'DOWN', 'player-0-worker-7': 'RIGHT', 'player-0-worker-8': 'UP', 'player-0-worker-12': 'UP', 'player-0-worker-13': 'UP'}
        """
        return command
    
    

