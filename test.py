import sys

sys.path.append("game")

import torch

from zerosum_env import make, evaluate
from zerosum_env.envs.carbon.helpers import *

from algorithms.planning_policy.planning_policy import PlanningPolicy

policy = PlanningPolicy()

logs = []
env = make("carbon", configuration={"randomSeed": 1}, logs=logs)
env.reset()

info = env.run([policy.take_action, "random"])

env.render(mode="ipython", width=1000, height=700)

# print(logs)
player_a, player_b = info[-1]
if player_a['reward'] == player_b['reward']:
    print("平局!")
elif player_a['reward'] is None:
    print(f"队伍B获胜 ({player_b['reward']})!")
elif player_b['reward'] is None:
    print(f"队伍A获胜 ({player_a['reward']})!")
elif player_a['reward'] > player_b['reward']:
    print(f"队伍A获胜 ({player_a['reward']})!")
else:
    print(f"队伍B获胜 ({player_b['reward']})!")