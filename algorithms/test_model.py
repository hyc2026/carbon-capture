import torch
import torch.nn.functional as F
import torch.optim as optim

from algorithms.model import Model


class TestModel:

    def test_critic_net(self):
        critic_net = Model(is_actor=False)
        print(critic_net)

        num_agents = 5
        target = torch.randn(num_agents, 1)
        agent_obs = []
        for i in range(num_agents):
            obs_vector = torch.randn(8)
            obs_cnn = torch.randn(13, 15, 15)
            obs = torch.cat([obs_vector, obs_cnn.reshape(-1)])
            agent_obs.append(obs)
        agent_obs = torch.stack(agent_obs)

        optimizer = optim.AdamW(critic_net.parameters(), lr=0.001)
        for i in range(100):
            value = critic_net(agent_obs)
            loss = F.mse_loss(value, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())

    def test_actor_net(self):
        actor_net = Model()
        print(actor_net)

        num_agents = 10
        action_dim = 5
        target_action = torch.randint(action_dim, (num_agents, ))

        agent_obs = []
        for i in range(num_agents):
            obs_vector = torch.randn(8)
            obs_cnn = torch.randn(13, 15, 15)
            obs = torch.cat([obs_vector, obs_cnn.reshape(-1)])
            agent_obs.append(obs)
        agent_obs = torch.stack(agent_obs)

        optimizer = optim.AdamW(actor_net.parameters(), lr=0.001)
        for i in range(1000):
            actor_output = actor_net(agent_obs)
            loss = F.cross_entropy(actor_output, target_action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())


