from copy import deepcopy
from posixpath import commonpath
from typing import Dict

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
        for e in tqdm(range(epoch), desc="epochs"):
            self.resume()
            cur_index = 1
            flag = 0
            while 1:
                if data_dir is not None:
                    cur_name = data_dir.strip('/') + '/' + 'data' + str(cur_index)
                    logger.info(f'cur_name:  {cur_name}')
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
                logger.info(f'cur_batches_size: {len(batches)}')
                random.shuffle(batches)
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
                # if eval_batches and (i % int(step_per_epoch / eval_per_epoch) == 0 or i == step_per_epoch - 1):
                eval_result = self.eval(eval_batches)
                logger.info(f"eval_result at epoch {e}: {eval_result} best result: {best_eval_result}")
                # 修改模型保存路径来适应rl
                if not os.path.exists('runs'):
                    os.mkdir('runs')
                if not os.path.exists(os.path.join('runs', 'run1')):
                    os.mkdir(os.path.join('runs', 'run1'))
                if not os.path.exists(os.path.join('runs', 'run1', 'models')):
                    os.mkdir(os.path.join('runs', 'run1', 'models'))
                if eval_result > best_eval_result:
                    best_eval_result = eval_result
                    self.save(os.path.join('runs', 'run1', 'models', 'model_best.pth'))
                logger.info(f"epoch loss: {self.ans.item()}")
                self.save(os.path.join('runs', 'run1', 'models', 'model_' + str(e) + '.pth'))
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
                cur_action = BaseActions[cur_choice]
            else:
                cur_choice = actions_probs[index].argmax()
                cur_action = WorkerActions[cur_choice]
            # if cur_action:
            #     cur_action = cur_action.name
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


def safety_detect(commands: Dict[str, WorkerAction or RecrtCenterAction], obs: Board) -> Dict[str, WorkerAction or RecrtCenterAction]:
    def get_new_position(point1: Point, action: WorkerAction):
        new_x = point1.x
        new_y = point1.y
        if action == WorkerAction.RIGHT:
            new_x += 1
        elif action == WorkerAction.LEFT:
            new_x -= 1
        if action == WorkerAction.UP:
            new_y -= 1
        elif action == WorkerAction.DOWN:
            new_y += 1
        new_x %= 15
        new_y %= 15
        return new_x, new_y

    def get_cross(point1: Point):
        return (point1.x, point1.y), \
               ((point1.x + 1) % 15, point1.y), \
               ((point1.x - 1) % 15, point1.y), \
               (point1.x, (point1.y + 1) % 15), \
               (point1.x, (point1.y - 1) % 15)

    centers = obs.recrtCenters
    positions = {}
    danger_zones = {}
    current_player_id = obs.current_player_id
    for rec_id in centers:
        if centers[rec_id].player_id == current_player_id:
            positions[rec_id] = centers[rec_id].position

    for worker_id in obs.workers:
        cur_worker = obs.workers[worker_id]
        if cur_worker.player_id == current_player_id:
            positions[worker_id] = cur_worker.position
        else:
            # 用地方单位打一遍危险区，分值为0.5
            cross_positions = get_cross(cur_worker.position)
            for eve_pos in cross_positions:
                if eve_pos not in danger_zones:
                    danger_zones[eve_pos] = 0.5
                else:
                    danger_zones[eve_pos] += 0.5

    agents = list(commands.keys())
    print(agents)
    import random
    # 目的是每轮中，agent的优先级都不一样
    random.shuffle(agents)
    print(agents)
    for cur_agent_id in agents:
        cur_action = commands[cur_agent_id]
        cur_pos = positions[cur_agent_id]
        if 'recrtCenter' in cur_agent_id:
            if cur_action is not None:
                if positions[cur_agent_id] in danger_zones:
                    if danger_zones[cur_pos] >= 1:
                        commands[cur_agent_id] = None
                    else:
                        danger_zones[cur_pos] += 1
                else:
                    danger_zones[cur_pos] = 1
        else:

            copy_action_list = copy.deepcopy(WorkerActions)
            random.shuffle(copy_action_list)
            new_action_list = [cur_action]
            for eve_action in copy_action_list:
                if eve_action != cur_action:
                    new_action_list.append(eve_action)
            flag = 0
            final_action = cur_action
            # 优先选择危险区以外的
            for eve_action in new_action_list:
                new_pos = get_new_position(cur_pos, eve_action)
                if new_pos not in danger_zones:
                    final_action = eve_action
                    flag = 1
                    break
            # 如果都在危险区，则选择0.5分一下的危险区执行
            if flag == 0:
                for eve_action in new_action_list:
                    new_pos = get_new_position(cur_pos, eve_action)
                    if danger_zones[new_pos] <= 0.5:
                        final_action = eve_action
                        break
            # 更新危险区，选择最终action
            new_pos = get_new_position(cur_pos, final_action)
            if new_pos in danger_zones:
                danger_zones[new_pos] += 1
            else:
                danger_zones[new_pos] = 1
            # if final_action != cur_action:
            #     print(final_action, cur_action, cur_agent_id)
            commands[cur_agent_id] = final_action
    # print(danger_zones)
    return commands


def agent(obs, configuration):
    obs = Board(obs, configuration)
    obs_parser = ObservationParser()
    obs_feature, _, _ = obs_parser.obs_transform(obs)
    cur_feature = transfer_ob_feature_to_model_feature(obs_feature)
    batch = data_loader.process_data([cur_feature])[0]
    commands = model.predict(batch)

    # print_commands = {}
    # for eve_id in commands:
    #     if commands[eve_id] is not None:
    #         print_commands[eve_id] = commands[eve_id].name
    # print('过滤前:', print_commands)
    commands_action = safety_detect(commands, obs)
    final_commands = {}
    for eve_id in commands_action:
        if commands_action[eve_id] is not None:
            final_commands[eve_id] = commands_action[eve_id].name
    # print('过滤后:', final_commands)
    return final_commands


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


def split_file(src_data_path: str, split_size=1000, data_dir='tmp_data/'):
    count = 1
    name = 'data'
    size = 0
    if os.path.isdir(src_data_path):
        file_name_list = os.listdir(src_data_path)
        file_name_list = [os.path.join(src_data_path, _) for _ in file_name_list]
    else:
        file_name_list = [src_data_path]
    # length = len(content_list)
    global test_batches
    print(file_name_list)
    for eve_file in file_name_list:
        with open(eve_file, 'rb') as f:
            content_list = pickle.load(f)
            length = len(content_list)
            size += length
        random.shuffle(content_list)
        for start in range(0, length, split_size):
            end = start + split_size if start + split_size <= length else length
            cur = content_list[start: end]
            if end != length:
                with open(data_dir + name + str(count), 'wb') as f:
                    pickle.dump(cur, f)
                    count += 1
            else:
                test_batches += DataLoader.process_data(cur)
    print(count, size)
    return size


if __name__ == '__main__':
    import random
    random.seed(2021)
    data_path = "data"
    train_batches = []
    test_batches = []
    # read_data = read_train_data_pickle(data_path)
    # read_data = eval(open('datasets_200.txt', 'r').read())
    data_directory = 'tmp_data/'
    if not os.path.exists(data_directory):
        os.mkdir(data_directory)
    data_num = split_file(data_path, split_size=1500, data_dir=data_directory)
    logger.info(f"preprocessing: data count {data_num}")
    # batches = data_loader.process_data(read_data)
    # logger.info(f"preprocess finish: batches count {len(batches)}")
    # random.shuffle(batches)
    # trian_size = int(len(batches) * 0.9)
    # train_batches = batches[:trian_size]
    # eval_batches = batches[trian_size:]

    logger.info(f"train batches count: {data_num - len(test_batches)} eval batches count: {len(test_batches)}")
    model.train(train_batches, epoch=30, eval_batches=test_batches, eval_per_epoch=2, data_dir=data_directory)
    
    # cur = batches[0][0][20:25], batches[0][1][20:25], batches[0][2][20:25]
    # print(cur[1], cur[2])
    # print(model.predict(cur))
