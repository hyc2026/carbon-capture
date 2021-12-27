from copy import deepcopy
from posixpath import commonpath
import torch
from torch import tensor
import torch.nn.functional as func
from torch.optim import optimizer
from algorithms.model import Model
from numpy import array
import copy
import logging
from zerosum_env.envs.carbon.helpers import \
    (Board, Player, Cell, Collector, Planter, Point, \
        RecrtCenter, Tree, Worker, RecrtCenterAction, WorkerAction)
from tqdm import tqdm
from submission import ObservationParser
import pickle
import os


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

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

    @staticmethod
    def process_data(data: list, batch_size=256):
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
    def __init__(self, device='cuda:0', lr=1e-3) -> None:
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
        self.losses += loss.detach().cpu() * batch_size
        self.ans = self.losses / self.total
        return self.ans

    def resume(self):
        self.total = 0
        self.losses = tensor(0).float()

    def train(self, batches=None, epoch=22, eval_batches=None, eval_per_epoch=1, data_dir=None):
        # 每个batch是两个list，feature_list和target_list
        model = self.model
        optimizer = self.optimizer
        loss_func = func.cross_entropy
        best_eval_result = 0
        step_per_epoch = len(batches)
        model.train()
        # eval_steps = [int(len(batches) / eval_per_epoch * i) for i in range(eval_per_epoch)]
        cur_index = 1
        flag = 0
        for e in tqdm(range(epoch + 1), desc="epochs"):
            self.resume()
            while 1:
                if data_dir is not None:
                    cur_name = data_dir.strip('/') + '/' + 'data' + str(cur_index)
                    cur_index += 1
                    if os.path.exists(cur_name):
                        with open(cur_name, 'rb') as f:
                            batches = pickle.load(f)
                            batches = DataLoader.process_data(batches)
                            flag = 1
                    else:
                        break
                if batches is None:
                    break
                print('cur_batches_size:', len(batches))
                for i, batch in enumerate(tqdm(batches, total=len(batches), desc="step")):

                    feature, target, _ = batch
                    feature = tensor(feature).float()
                    target = tensor(target)

                    feature = feature.to(self.device)
                    target = target.to(self.device)

                    logits = model(feature)
                    loss = loss_func(logits, target)

                    loss.backward()
                    optimizer.step()
                    # scheduler.step()
                    optimizer.zero_grad()
                    self.update_loss(loss, batch_size=len(batch))
                    if eval_batches and (i % int(step_per_epoch / eval_per_epoch) == 0 or i == step_per_epoch - 1):
                        eval_result = self.eval(eval_batches)
                        logger.info(f"eval_result at epoch {e} step {i}: {eval_result} best result: {best_eval_result}")
                        if eval_result > best_eval_result:
                            best_eval_result = eval_result
                            self.save('models/best')
                logger.info(f"epoch loss: {self.ans.item()}")
                self.save('models/epoch_' + str(e))
                if flag == 0:
                    break

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path))
    
    def eval(self, batches):
        model = self.model
        model.eval()
        logger.info(f"begin eval: batches count {len(batches)}")
        correct = 0
        total = 0
        for batch in batches:
            feature, target, _ = batch
            feature = tensor(feature).float()
            target = tensor(target)
            
            feature = feature.to(self.device)
            target = target.to(self.device)

            logits = model(feature)
            predict = torch.argmax(logits, 1)
            label = target
            correct += (predict == label).sum().item()
            total += len(batch)
        return correct / total

    # 只输入一个batch，就是当前的环境
    def predict(self, batch):
        self.model.eval()
        feature, _, agent_ids = batch
        feature = tensor(feature).float()
        feature = feature.to(self.device)
        actor_probs = self.model(feature)
        final_dict = {}
        actions_probs = actor_probs.detach().cpu().numpy()
        agent_num = len(agent_ids)
        for index in range(agent_num):
            cur_id = agent_ids[index]
            if 'recrtCenter' in cur_id:
                # 转化中心预测的结果只有三种
                cur_choice = actions_probs[index][:3].argmax() 
                cur_action =  BaseActions[cur_choice]
            else:
                cur_choice = actions_probs[index].argmax()
                cur_action = WorkerActions[cur_choice]
            if cur_action:
                cur_action = cur_action.name
            final_dict[cur_id] = cur_action
        return final_dict


data_loader = DataLoader()
model = ActionImitation()
try:
    model.load('models/best')
    logger.info('trian from checkpoint!!')
except:
    logger.info('train from scratch!!')
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


def read_train_data_pickle(data_path):
    if os.path.isdir(data_path):
        file_name_list = os.listdir(data_path)
        file_name_list = [os.path.join(data_path, _) for _ in file_name_list]
    else:
        file_name_list = [data_path]
    read_data = []
    for file_path in file_name_list:
        with open(file_path, 'rb') as f:
            read_data.extend(pickle.load(f))
    return read_data  


def split_file(content_list: list, split_size=1000, data_dir='tmp_data/'):
    count = 1
    name = 'data'
    length = len(content_list)
    global test_batches
    for start in range(0, length, split_size):
        end = start + split_size if start + split_size <= length else length
        cur = content_list[start: end]
        if end != length:
            with open(data_dir + name + str(count), 'wb') as f:
                pickle.dump(cur, f)
        else:
            test_batches = DataLoader.process_data(cur)
        count += 1


if __name__ == '__main__':
    import random
    random.seed(2021)
    data_path = "data1"
    train_batches = []
    test_batches = []
    read_data = read_train_data_pickle(data_path)
    # read_data = eval(open('datasets_200.txt', 'r').read())
    data_directory = 'tmp_data/'
    split_file(read_data, data_dir=data_directory)
    logger.info(f"preprocessing: data count {len(read_data)}")
    # batches = data_loader.process_data(read_data)
    # logger.info(f"preprocess finish: batches count {len(batches)}")
    # random.shuffle(batches)
    # trian_size = int(len(batches) * 0.9)
    # train_batches = batches[:trian_size]
    # eval_batches = batches[trian_size:]

    logger.info(f"train batches count: {len(train_batches)} eval batches count: {len(test_batches)}")
    model.train(train_batches, epoch=30, eval_batches=test_batches, eval_per_epoch=2, data_dir=data_directory)
    
    # cur = batches[0][0][20:25], batches[0][1][20:25], batches[0][2][20:25]
    # print(cur[1], cur[2])
    # print(model.predict(cur))
