import json
from easydict import EasyDict as edict, EasyDict
import pandas
import copy
import jstyleson
import sys
from collections import defaultdict
import nni
import logging

sys.path.append('./')
sys.path.append('../')
sys.path.append('../..')

from algorithms.planning_policy.planning_policy import PlanningPolicy

from batch_test import run_experiments

#需要自动调超参数的类
#它有一个要求，就是这个类有config这个参数，并且搜索空间已经放在configs/search_space.json中或者configs/planning_policy_mother.json
fine_tuned_policy_class = PlanningPolicy


def PlanningPolicyWithConfig(config):
    #x是被装饰的函数
    #或者x是被装饰的类(的__init__函数)
    def wrapper(x):
        #y是被装饰的函数的参数
        #或者y是被装饰的类的__init__函数的参数
        def inner(*args, **kwargs):
            planning_policy = x(*args, **kwargs)
            planning_policy.config = config
            return planning_policy

        return inner

    return wrapper


def unflatten_dict(dictionary):
    '''
    将一个平面的字典转换为树状的字典
    '''
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict


def flattern_dict(dic, sep):
    '''
    将一个树状的字典转换为平面的字典
    '''
    df = pandas.json_normalize(dic, sep=sep)
    return df.to_dict(orient='records')[0]


def make_nni_search_space(config):
    '''
    将一个<修改后的配置文件>转换为nni的搜索空间
    - NNI要求的search_space.json文件的每一个值都必须定义'_type'和'_value'，但是我
    们可能并不需要自动调那么多参数，所以<修改后的配置文件>中的参数可以是一个固定
    的值，也可以是带有'_type'和'_value'的字典
    - NNI的sample里面给的是一个单级字典，但是我们不知道NNI是否允许嵌套字典，无论如何，这里先将我们的Policy使用的嵌套字典，转化为了单级字典，再交给NNI
    '''
    config = flattern_dict(config, '.')
    nni_param_dict = defaultdict(dict)
    for key, value in config.items():
        if key.endswith('._type'):
            base_key = key.replace('._type', '')
            nni_param_dict[base_key]['_type'] = value
        elif key.endswith('._value'):
            base_key = key.replace('._value', '')
            nni_param_dict[base_key]['_value'] = value
        else:
            config[key] = {"_type": "choice", "_value": [value]}
    config.update(nni_param_dict)
    for key in nni_param_dict:
        config.pop(key + '._type')
        config.pop(key + '._value')
    return config


def make_nni_search_space_file():
    with open(
            'algorithms/planning_policy/planning_policy_nni/configs/planning_policy_mother.json'
    ) as fin:
        config = jstyleson.load(fin)

    config = make_nni_search_space(config)
    with open(
            'algorithms/planning_policy/planning_policy_nni/configs/search_space.json',
            'w') as fout:
        jstyleson.dump(config, fout, indent=2, ensure_ascii=False)


def main(Debugging=False):

    if Debugging:
        with open(
                'algorithms/planning_policy/planning_policy_nni/configs/planning_policy.json'
        ) as fin:
            config = jstyleson.load(fin)
    else:
        config = nni.get_next_parameter()
        config = unflatten_dict(config)
        print(config)

    planning_policy_class_with_new_config = PlanningPolicyWithConfig(config)(
        fine_tuned_policy_class)

    total_result = run_experiments(planning_policy_class_with_new_config,
                                   fine_tuned_policy_class, 1)

    relative_win_rate = total_result.A_relative_win / total_result.experiment_count

    print('relative_win_rate: ', relative_win_rate)

    nni.report_final_result(relative_win_rate)


if __name__ == '__main__':
    main(Debugging=False)
    # make_nni_search_space_file()
    # policy=PlanningPolicyWithConfig({})(PlanningPolicy)()
