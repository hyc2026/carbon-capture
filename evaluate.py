import torch

from zerosum_env import evaluate
from zerosum_env.envs.carbon.helpers import *

from algorithms.eval_policy import EvalPolicy


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
            num_episodes=100)  # default == 1
        # measure the mean of rewards of two agents
        rew_1 = sum([i[0] for i in rew]) / len(rew)
        rew_2 = sum([i[1] for i in rew]) / len(rew)
        return rew_1, rew_2
        # return rew[0][0], rew[0][1]

    r1, r2 = evaluate_agent()
    print("agent : {0}, random : {1}".format(r1, r2))
