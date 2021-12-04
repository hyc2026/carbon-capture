import sys
import easydict
sys.path.append("game")
import torch
import random
from easydict import EasyDict 
from zerosum_env import make, evaluate
from zerosum_env.envs.carbon.helpers import *




from algorithms.planning_policy.planning_policy import PlanningPolicy
from algorithms.planning_policy.wb_planning_policy import PlanningPolicy as WbPolicy
from algorithms.jiaqi_policy.my_policy_2 import MyPolicy as JiaqiPolicy
from algorithms.planning_policy.planning_policy_jds import PlanningPolicyJDS as JdsPolicy


expriment_repeat = 5
policy_class_list = [PlanningPolicy, JiaqiPolicy,WbPolicy,JdsPolicy]

def run_experiment(policy_class_A,policy_class_B):
    policy_A = policy_class_A()
    policy_B = policy_class_B()

    logs=[]
    env = make(
            "carbon",
            configuration={"randomSeed": random.randint(1, 100)},
            logs=logs)
    env.reset()
    info = env.run([policy_A.take_action, policy_B.take_action])
    env.render(mode="ipython", width=1000, height=700)

    result=EasyDict()

    result.dual=result.B_normal_win=result.A_normal_win=result.A_win_B_error=result.B_win_A_error=0

    player_a, player_b = info[-1]
    if player_a['reward'] == player_b['reward']:
        print("平局!")
        result.dual = 1
    elif player_a['reward'] is None:
        print(f"队伍B获胜 ({player_b['reward']})!")
        result.B_win_A_error = 1
    elif player_b['reward'] is None:
        print(f"队伍A获胜 ({player_a['reward']})!")
        result.A_win_B_error = 1
    elif player_a['reward'] > player_b['reward']:
        print(f"队伍A获胜 ({player_a['reward']})!")
        result.A_normal_win = 1
    else:
        print(f"队伍B获胜 ({player_b['reward']})!")
        result.B_normal_win = 1
    
    result.A_relative_win=result.A_win_B_error+result.A_normal_win -result.B_win_A_error-result.B_normal_win
    result.experiment_count=1
    return result
    

def run_experiments(policy_class_A,policy_class_B,experiment_count:int=expriment_repeat):
    total_result=None
    for i in range(experiment_count):
        single_experiment_result=run_experiment(policy_class_A,policy_class_B)
        if total_result is None:
            total_result=single_experiment_result
        else:
            for key in total_result.keys():
                total_result[key]+=single_experiment_result[key]
    return total_result



def main():
    random.seed(0)
    all_policy_class_total_results=[]
    for policy_class_A in policy_class_list:
        for policy_class_B in policy_class_list:
            if policy_class_A is policy_class_B: continue
            for i in range(expriment_repeat):
                total_result=run_experiments(policy_class_A,policy_class_B)
    all_policy_class_total_results.append((policy_class_A,policy_class_B,total_result))
    print(all_policy_class_total_results)

            

if __name__ == "__main__":
    main()