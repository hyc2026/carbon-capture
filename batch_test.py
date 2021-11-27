import sys
sys.path.append("game")
import torch
import random
from zerosum_env import make, evaluate
from zerosum_env.envs.carbon.helpers import *

from algorithms.planning_policy.planning_policy import PlanningPolicy
from algorithms.jiaqi_policy.my_policy import MyPolicy

policy_class_list = [PlanningPolicy, MyPolicy]
expriment_repeat = 1

def main():
    random.seed(0)
    for policy_class_A in policy_class_list:
        for policy_class_B in policy_class_list:
            if policy_class_A is policy_class_B: break
            policy_A = policy_class_A()
            policy_B = policy_class_B()
            A_win_B_error = A_normal_win = dual = B_normal_win = B_win_A_error = 0
            for i in range(expriment_repeat):
                
                logs=[]
                env = make(
                        "carbon",
                        configuration={"randomSeed": random.randint(1, 100)},
                        logs=logs)
                env.reset()
                info = env.run([policy_A.take_action, policy_B.take_action])
                env.render(mode="ipython", width=1000, height=700)


                player_a, player_b = info[-1]
                if player_a['reward'] == player_b['reward']:
                    print("平局!")
                    dual += 1
                elif player_a['reward'] is None:
                    print(f"队伍B获胜 ({player_b['reward']})!")
                    B_win_A_error += 1
                elif player_b['reward'] is None:
                    print(f"队伍A获胜 ({player_a['reward']})!")
                    A_win_B_error += 1
                elif player_a['reward'] > player_b['reward']:
                    print(f"队伍A获胜 ({player_a['reward']})!")
                    A_normal_win += 1
                else:
                    print(f"队伍B获胜 ({player_b['reward']})!")
                    B_normal_win += 1

            print(f"{policy_A.__class__} VS {policy_B.__class__}")
            print(
                f"{policy_A.__class__}:    normal_win:{A_normal_win} dual:{dual} normal_lose:{B_normal_win} error:{B_win_A_error} opponent_error:{A_win_B_error}"
            )

if __name__ == "__main__":
    main()