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
from envs.obs_parser import ObservationParser
from utils.utils import to_tensor

# 核心代码：policy
class EvalPolicy(BasePolicy):
    """
    展示策略训练结果使用
    """
    def __init__(self):
        super().__init__()
        self.obs_parser = ObservationParser()
        self.previous_obs = None

        self.tensor_kwargs = {"dtype": torch.float32, "device": torch.device("cpu")}
        self.actor_model = Model(is_actor=True)

    def restore(self, model_dict: Dict[str, OrderedDict[str, torch.Tensor]], strict=True):
        """
        Restore models and optimizers from model_dict.

        :param model_dict: (dict) State dict of models and optimizers.
        :param strict: (bool, optional) whether to strictly enforce the keys of torch models
        """
        self.actor_model.load_state_dict(model_dict['actor'], strict=strict)
        self.actor_model.eval()

    def get_actions(self, observation, available_actions=None) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = to_tensor(observation).to(**self.tensor_kwargs)

        # 根据观察的状态，actor model 进行决策，生成每个 action 的logits
        # actor model 就是我们训好的 model
        action_logits = self.actor_model(obs)
        if available_actions is not None:
            available_actions = to_tensor(available_actions).to(**self.tensor_kwargs)
            # 如果在 available_actions 中已经屏蔽了一些 action, 那么将其 logits 设置为最小值
            action_logits[available_actions == 0] = torch.finfo(torch.float32).min

        action = action_logits.sort(dim=1, descending=True)[1][:, 0]  # 按照概率值倒排,选择最大概率位置的索引
        dist = Categorical(logits=action_logits)
        log_prob = dist.log_prob(action)

        action = action.detach().cpu().numpy().flatten()
        log_prob = log_prob.detach().cpu().numpy()

        return action, log_prob

    
    def get_all_agent_ids(self, obs: Board):
        player = obs.current_player
        ids = [player.recrtCenter_ids + player.worker_ids]
        return ids
    
    def update_agent_actions(self, obs: Board, agent_actions: Dict[str, int], map_info):
        player = obs.current_player
        for planter in player.planters:
            pid = planter.id
            # todo: 当前为种树员，且当前位置无树，无转化中心
            # if pid in agent_actions:
                # agent_actions[pid] = 0
    

    def take_action(self, observation, configuration):
        # todo: 这个地方只使用了当前step，上一步状态，可以探索多观察前面几步
        current_obs = Board(observation, configuration)
        previous_obs = self.previous_obs if current_obs.step > 0 else None

        # 通过观察当前和历史状态，返回下一步候选状态集合
        # 这个是旧的 observation
        agent_obs_dict, dones, available_actions_dict = self.obs_parser.obs_transform(current_obs, previous_obs)

        # 这个是新的 observation，后面很多信息都要从这里拿
        map_info, step_now, my_info, op_info = self.obs_parser.obs_transform_new(current_obs)

        # 从这里开始写规则
        # 针对每种 agent 写一套规则
        # 我现在尝试写一下种树的规则

        all_agent_ids = self.get_all_agent_ids(current_obs)
        print(f'all_agent_ids: {all_agent_ids}')

        self.previous_obs = copy.deepcopy(current_obs)

        agent_ids, agent_obs, avail_actions = zip(*[(agent_id, torch.from_numpy(obs_), available_actions_dict[agent_id])
                                                    for agent_id, obs_ in agent_obs_dict.items()])

        # 返回计算出的概率最大的 action
        actions, _ = self.get_actions(agent_obs, avail_actions)
        # agent_id, action_value
        agent_actions = {agent_id: action.item() for agent_id, action in zip(agent_ids, actions)}
        
        ## 这个地方修改一下，把种树员的action全部换成种树
        agent_actions = self.update_agent_actions(current_obs, agent_actions, map_info)
        
        command = self.to_env_commands(agent_actions)
        # print(command)
        # 这个地方返回一个cmd字典
        # 类似这样
        """
        {'player-0-recrtCenter-0': 'RECPLANTER', 'player-0-worker-0': 'RIGHT', 'player-0-worker-5': 'DOWN', 'player-0-worker-6': 'DOWN', 'player-0-worker-7': 'RIGHT', 'player-0-worker-8': 'UP', 'player-0-worker-12': 'UP', 'player-0-worker-13': 'UP'}
        """

        return command

