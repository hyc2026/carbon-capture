import json
from easydict import EasyDict as edict, EasyDict
import pandas
import copy
import sys
from collections import defaultdict
import nni

sys.path.append('./')
sys.path.append('../')
sys.path.append('../..')

from algorithms.planning_policy.planning_policy import PlanningPolicy

from batch_test import run_experiments

epoch_count = 30
config_value_decrease_to_ratio = 0.95
config_value_increase_to_ratio = 1.05
relative_win_rate_threshold = 0.15
fine_tuned_policy_class=PlanningPolicy

def judge_if_new_config_is_better_than_config_by_result(result):
    if result.A_relative_win / result.experiment_count > relative_win_rate_threshold:
        return True
    else:
        return False


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


def flattern_dict(dic,sep):
    df = pandas.json_normalize(dic, sep=sep)
    return df.to_dict(orient='records')[0]

def make_nni_search_space(config):
    config =  flattern_dict(config,'|')
    nni_param_dict = defaultdict(dict)
    for key,value in config.items():
        if key.endswith('|_type'):
            base_key = key.replace('|_type','')
            nni_param_dict[base_key]['_type']=value
        elif key.endswith('|_value'):
            base_key = key.replace('|_value','')
            nni_param_dict[base_key]['_value']=value
        else:
            config[key]={"_type":"choice", "_value": [value]}
    config.update(nni_param_dict)
    for key in nni_param_dict:
        config.pop(key+'|_type')
        config.pop(key+'|_value')
    return config

        
def main():
    
    with open('algorithms/planning_policy/planning_policy_nni/configs/planning_policy_step_0.json') as fin:
        config=json.load(fin)
    config = make_nni_search_space(config)
    with open('algorithms/planning_policy/planning_policy_nni/configs/search_space.json','w') as fout:
        json.dump(config,fout,indent=2,ensure_ascii=False)

    for i in range(epoch_count):

        planning_policy_config = nni.get_next_parameter()
        planning_policy_class_with_new_config = PlanningPolicyWithConfig(config)(fine_tuned_policy_class)
        
        total_result=run_experiments(planning_policy_class_with_new_config,fine_tuned_policy_class)

        relative_win_rate=total_result.A_relative_win/total_result.experiment_count
        
        nni.report_intermediate_result(relative_win_rate)
        # nni.


if __name__ == '__main__':
    main()
    # policy=PlanningPolicyWithConfig({})(PlanningPolicy)()
