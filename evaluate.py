import time
import torch
from collections import Counter

from zerosum_env import evaluate
from zerosum_env.envs.carbon.helpers import *
from algorithms.planning_policy.planning_policy import PlanningPolicy
from algorithms.eval_policy import EvalPolicy
from submission_model import agent
# from zyp_model import agent
import random

try:
    NUM_EPISODES = int(sys.argv[1])
except IndexError as e:
    print("没有指定测评轮数，使用默认值50")
    NUM_EPISODES = 3

# 计算从start到现在花费的时间
def time_cost(start):
    cost = int(time.time() - start)
    h = cost // 3600
    m = (cost % 3600) // 60
    print('')
    print('cost %s hours %s mins' % (h, m))


if __name__ == '__main__':

    model_policy = EvalPolicy()
    # model_path = './runs/run3/models/model_best.pth'
    # model_policy.restore(torch.load(model_path))
    #model_path = '/sdb/v-bingwang/workspace/carbon_baseline_cuda/runs/run2/models/model_950.pth'
    my_ploicy = PlanningPolicy()

    t0 = time.time()

    # function for testing agent
    def take_action(observation, configuration):
        action = model_policy.take_action(observation, configuration)
        return action

    # function for testing
    def evaluate_agent():
        rew, _, _, _ = evaluate(
            "carbon",
            agents=[agent, "random"],
            configuration={"randomSeed": random.randint(1,2147483646)},
            debug=True,
            num_episodes=NUM_EPISODES)  # default == 1
        # measure the mean of rewards of two agents
        # counter = Counter("win vs. loss")
        # TODO: More rounds to evaluate?
        win_round = list(filter(lambda x: x[0] > x[1], rew))
        print(f"[Agent vs. Random]: {win_round}\n")
        win_round_num = len(win_round)
        return win_round_num, NUM_EPISODES - win_round_num

    r1, r2 = evaluate_agent()
    print("agent : {0}, random : {1}\n".format(r1, r2))
    print(f"Win Rate: {round(r1 / NUM_EPISODES, 4)}")
    time_cost(t0)
