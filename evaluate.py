import torch
from collections import Counter

from zerosum_env import evaluate
from zerosum_env.envs.carbon.helpers import *

from algorithms.eval_policy import EvalPolicy

try:
    NUM_EPISODES = sys.argv[1]
except IndexError as e:
    print("没有指定测评轮数，使用默认值50")
    NUM_EPISODES = 50

if __name__ == '__main__':

    player = EvalPolicy()
    player.restore(torch.load('./runs/run2/models/model_best.pth'))

    # function for testing agent
    def take_action(observation, configuration):
        action = player.take_action(observation, configuration)
        return action

    # function for testing
    def evaluate_agent():
        rew, _, _, _ = evaluate(
            "carbon",
            agents=[take_action, "random"],
            configuration={"randomSeed": 1},
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
