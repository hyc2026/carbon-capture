from copy import deepcopy
from posixpath import commonpath
import torch
from torch import tensor
import torch.nn.functional as func
from torch.optim import optimizer
from algorithms.model import Model
from numpy import array
import copy
from zerosum_env.envs.carbon.helpers import \
    (Board, Player, Cell, Collector, Planter, Point, \
        RecrtCenter, Tree, Worker, RecrtCenterAction, WorkerAction)
from tqdm import tqdm
from submission import ObservationParser
BaseActions = [None,
               RecrtCenterAction.RECCOLLECTOR,
               RecrtCenterAction.RECPLANTER]

WorkerActions = [None,
                 WorkerAction.UP,
                 WorkerAction.RIGHT,
                 WorkerAction.DOWN,
                 WorkerAction.LEFT]


def transfer_ob_feature_to_model_feature(ob_result, label_agent2action=None):

    map_feature = {}
    agent_info = []
    for agent_id, ob_features in ob_result.items():
        step_features = ob_features[0]
        my_cash = ob_features[1]
        opponent_cash = ob_features[2]
        agent_type = ob_features[3:6].tolist()
        x = ob_features[6]
        y = ob_features[7]
        index_begin = 8
        carbon_feature = ob_features[index_begin: index_begin + 15*15].reshape(15, 15).tolist()
        index_begin += 15 * 15
        base_feature = ob_features[index_begin: index_begin + 15 * 15].reshape(15, 15).tolist()
        index_begin += 15 * 15
        collector_feature = ob_features[index_begin: index_begin + 15 * 15].reshape(15, 15).tolist()
        index_begin += 15 * 15
        planter_feature = ob_features[index_begin: index_begin + 15 * 15].reshape(15, 15).tolist() 
        index_begin += 15 * 15
        worker_carbon_feature = ob_features[index_begin: index_begin + 15 * 15].reshape(15, 15).tolist()  
        index_begin += 15 * 15
        tree_feature = ob_features[index_begin: index_begin + 15 * 15].reshape(15, 15).tolist()
        index_begin += 15 * 15
        action_feature = ob_features[index_begin: index_begin + 5 * 15 * 15].reshape(5, 15, 15).tolist() 
        index_begin += 5 * 15 * 15
        my_base_distance_feature = ob_features[index_begin: index_begin + 15 * 15].reshape(15, 15).tolist() 
        index_begin += 15 * 15
        distance_features = ob_features[index_begin: index_begin + 15 * 15].reshape(15, 15).tolist()

        map_feature["step_features"] = step_features
        map_feature["my_cash"] = my_cash
        map_feature["opponent_cash"] = opponent_cash
        map_feature["carbon_feature"] = carbon_feature
        map_feature["base_feature"] = base_feature
        map_feature["collector_feature"] = collector_feature
        map_feature["planter_feature"] = planter_feature
        map_feature["worker_carbon_feature"] = worker_carbon_feature
        map_feature["tree_feature"] = tree_feature
        map_feature["action_feature"] = action_feature
        map_feature["my_base_distance_feature"] = my_base_distance_feature

        agent_info.append((
            agent_id,
            x,
            y,
            agent_type,
            distance_features,
            label_agent2action.get(agent_id, 0) if label_agent2action is not None else -1
        ))

    item = {
        "map_features" : map_feature,
        "agent_info" : agent_info
    }
    return item


class DataLoader:
    def __init__(self):
        pass

    def process_data(self, data: list, batch_size=256):
        final_batches = []
        middle_data = []
        labels = []
        agent_ids = []
        for eve in data:
            cur_data = []
            shared_feature = eve['map_features']
            vector_feature = [shared_feature['step_features'], 
                                   shared_feature['my_cash'],
                                   shared_feature['opponent_cash']
                                       
            ]
            cnn_feature = [shared_feature['carbon_feature'],
                           shared_feature['base_feature'],
                           shared_feature['collector_feature'],
                           shared_feature['planter_feature'],
                           shared_feature['worker_carbon_feature'],
                           shared_feature['tree_feature']

            ]
            cnn_feature.extend(shared_feature['action_feature'])
            cnn_feature.append(shared_feature['my_base_distance_feature'])
            agent_infos = eve['agent_info']
            for agent in agent_infos:
                agent_id, x, y, agent_type, distance, label = agent
                # 处理vector
                cur_vector = copy.deepcopy(vector_feature)
                cur_vector.extend([x, y])
                cur_vector.extend(agent_type)
                cur_vector = tensor(cur_vector)
                # 处理cnn_feature
                cur_cnn = copy.deepcopy(cnn_feature)
                cur_cnn.append(distance)
                cur_cnn = tensor(cur_cnn).reshape(-1)
                
                cur_feature = torch.cat([cur_vector, cur_cnn])
                cur_feature = cur_feature.numpy().tolist()
                middle_data.append(cur_feature)
                labels.append(label)
                agent_ids.append(agent_id)
        assert len(middle_data) == len(labels) == len(agent_ids)
        length = len(middle_data)
        for start in range(0, length, batch_size):
            end = start + batch_size if start + batch_size <= length else length
            cur_features = middle_data[start: end]
            cur_labels = labels[start: end]
            cur_ids = agent_ids[start: end]
            final_batches.append((cur_features, cur_labels, cur_ids))
        return final_batches




class ActionImitation:
    def __init__(self, device='cuda:0', lr=2e-4) -> None:
        if 'cuda' in device and not torch.cuda.is_available():
            self.device = 'cpu'
        else:
            self.device = device
        self.model = Model(is_actor=True)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.total = 0
        self.losses = tensor(0).float().to(self.device)
        


    def update_loss(self, loss, batch_size):
        self.total += batch_size
        self.losses += loss * batch_size
        self.ans = self.losses / self.total
        return self.ans

    def resume(self):
        self.total = 0
        self.losses = tensor(0).float()



    def train(self, batches: list, epoch=22):
        # 每个batch是两个list，feature_list和target_list
        self.model.train()
        for e in tqdm(epoch + 1):
            print('epoch:', e)
            self.resume()
            for batch in batches:
                feature, target, _ = batch
                feature = tensor(feature).float()
                target=  tensor(target)
                feature.to(self.device)
                target.to(self.device)
                predict = self.model(feature)
                loss = func.cross_entropy(predict, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.update_loss(loss, batch_size=len(batch))
            print(self.ans)
            self.save('models/epoch_' + str(e))

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path))

    # 只输入一个batch，就是当前的环境
    def predict(self, batch):
        self.model.eval()
        feature, _, agent_ids = batch
        feature = tensor(feature).float()
        actor_probs = self.model(feature)
        final_dict = {}
        actions = actor_probs.detach().numpy().argmax(axis=1)
        agent_num = len(agent_ids)
        for index in range(agent_num):
            cur_id = agent_ids[index]
            cur_choice = actions[index]
            if 'recrtCenter' in cur_id:
                # 转化中心预测的结果只有三种
                if cur_choice > 2:
                    cur_choice = 0 
                cur_action =  BaseActions[cur_choice]
                
            else:
                cur_action = WorkerActions[cur_choice]
            if cur_action:
                cur_action = cur_action.name
            final_dict[cur_id] = cur_action
        return final_dict


data_loader = DataLoader()
model = ActionImitation()
try:
    model.load('models/best')
except:
    print('加载失败!')
    pass


def agent(obs, configuration):
    obs = Board(obs, configuration)
    obs_parser = ObservationParser()
    obs_feature, _, _ = obs_parser.obs_transform(obs)
    cur_feature = transfer_ob_feature_to_model_feature(obs_feature)
    batch = data_loader.process_data([cur_feature])[0]
    commands = model.predict(batch)
    del_list = []
    for eve in commands:
        if commands[eve] is None:
            del_list.append(eve)
    for eve in del_list:
        commands.pop(eve)
    return commands

if __name__ == '__main__':
    read_data = eval(open('/Users/yupeng/Desktop/data.txt', 'r').read())
    batches = data_loader.process_data(read_data)
    # cur = batches[0][0][20:25], batches[0][1][20:25], batches[0][2][20:25]
    # print(cur[1], cur[2])

    # print(model.predict(cur))
    model.train(batches)
    model.save('models/best')
    