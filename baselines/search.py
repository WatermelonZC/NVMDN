"""Run the released search operators on physical task rows."""
import ctypes
import importlib
import json
import random
import signal
import time


def solve(data,method,seconds,seed):
    import numpy as np
    import torch
    from portable import ROOT,observation,replay,routes_to_actions
    if (data['mother'],data['daughter'])!=(4,8):raise ValueError('Search baselines use a 4/8 fleet')
    from baselines.heuristics.utils.Config import Config
    from baselines.heuristics import runtime
    module=importlib.import_module('baselines.heuristics.'+method)
    Config.M_parent,Config.M_child,Config.M=4,8,12
    Config.M_parent_list=list(range(1,5));Config.M_child_list=list(range(5,13))
    Config.M_position_map=observation(data)['positions'];Config.Graph_size=len(data['tasks'])
    original_loader=module.read_excel
    original_population=module.generate_multiweight_population
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
    if data.get('scenario')=='industrial':
        from baselines.edd import rollout_policy
        from baselines.heuristics.utils.Solution import Solution
        from baselines.heuristics.utils.util import path_map2sequence_map
        def population(instance,population_size=1):
            actions=rollout_policy(observation(data));routes={v:[] for v in range(1,13)}
            for task,mother,daughter in actions:routes[mother].append(task);routes[daughter].append(task)
            sequence,initial=path_map2sequence_map(routes);edd=Solution(instance,sequence,initial)
            runtime.observe(edd,0.,0,'edd_seed_ready')
            return [edd]+(original_population(instance,population_size-1) if population_size>1 else [])
    config=json.loads((ROOT/'configs/search.json').read_text())
    if seconds is None:seconds=config['budget_seconds']
    target=module
    if method=='ALNS':
        from baselines.heuristics.utils.ALNS_config import ALNSConfig
        target=ALNSConfig
    for key,value in config[method.lower()].items():setattr(target,key,value)
    runtime.best_solution=None;runtime.last_iteration=0;started=time.monotonic()
    strict=hasattr(signal,'setitimer')
    def expired(*_):raise runtime.BudgetExpired()
    def observe(*_):
        if time.monotonic()-started>=seconds:raise runtime.BudgetExpired()
    runtime.callback=observe
    if strict:
        old=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        module.read_excel=lambda _: [row.copy() for row in data['tasks']]
        if data.get('scenario')=='industrial':module.generate_multiweight_population=population
        returned=getattr(module,method)('instance',max_runtime=seconds)
        solution=returned[1] if method=='ALNS' else returned[0]
    except runtime.BudgetExpired:solution=runtime.best_solution
    except ctypes.ArgumentError as error:
        if 'BudgetExpired' not in str(error):raise
        solution=runtime.best_solution
    finally:
        if strict:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        runtime.callback=None
        module.read_excel=original_loader
        module.generate_multiweight_population=original_population
    if solution is None:raise RuntimeError('The search budget expired before initialization completed')
    actions=routes_to_actions(solution.get_path_map(),len(data['tasks']),4,8)
    result=replay(data,actions)
    if abs(result['objective']-solution.get_fitness())>1e-5:raise RuntimeError('Search replay mismatch')
    return dict(method=method,seed=seed,actions=actions,result=result,
        elapsed_seconds=time.monotonic()-started,budget_seconds=seconds,strict_timer=strict,independently_verified=True)
