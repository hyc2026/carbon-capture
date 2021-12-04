authorName: default
experimentName: PlanningPolicy
trialConcurrency: 32
maxExecDuration: 24h
maxTrialNum: 99999
trainingServicePlatform: local
searchSpacePath: configs/search_space.json #relative path to this file
useAnnotation: false
tuner:
  builtinTunerName: TPE
trial: # reletive path to $PWD
  command: python
  algorithms/planning_policy/planning_policy_nni/planning_policy_auto_test_make_best_parameter.py
  codeDir: .
  gpuNum: 0
