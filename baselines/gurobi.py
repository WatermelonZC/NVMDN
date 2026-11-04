"""Gurobi baseline for synthetic and industrial NVSP instances.

Use scripts/predict.py --method gurobi for both scenarios. Synthetic instances
use the compact shared-depot model. Industrial instances use distinct start
nodes for the vehicles. Both formulations preserve the published constraints.
"""
import contextlib
import math
import json
import sys
import time

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as error:
    raise RuntimeError('Install requirements.txt and configure a Gurobi license') from error

__all__ = ['solve']


def solve(data,seconds,seed):
    """Solve one NVSP instance with the formulation used for its scenario."""
    from portable import ROOT,observation,replay,routes_to_actions
    if (data['mother'],data['daughter'])!=(4,8):raise ValueError('The released Gurobi models use a 4/8 fleet')
    config=json.loads((ROOT/'configs/gurobi.json').read_text())
    if seconds is None:seconds=config['budget_seconds']
    started=time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr), gp.Env(empty=True) as env:
        env.start()
        if data.get('scenario')=='industrial':
            model,variables=_build_distinct_starts(data['tasks'],observation(data)['positions'],env)
        else:
            model,variables=_build_shared_depot(data['tasks'],env=env)
        try:
            model.Params.MIPGap=config['MIPGap']
            model.Params.Seed=seed;model.Params.TimeLimit=max(.001,seconds-(time.perf_counter()-started));model.optimize()
            if model.SolCount==0:raise RuntimeError('Gurobi returned no feasible incumbent')
            if data.get('scenario')=='industrial':
                values={k:({a:v.X for a,v in val.items()} if isinstance(val,dict) else val) for k,val in variables.items()}
                routes=_extract_distinct_routes(values)
            else:routes=_extract_shared_routes(variables)
            actions=routes_to_actions(routes,len(data['tasks']),4,8);result=replay(data,actions)
            if abs(result['objective']-model.ObjVal)>.05:raise RuntimeError('Gurobi replay mismatch')
            return dict(method='Gurobi',actions=actions,result=result,elapsed_seconds=time.perf_counter()-started,
                budget_seconds=seconds,lower_bound=model.ObjBound,solver_mip_gap=model.MIPGap,
                solver_status=int(model.Status),optimality_proven=model.Status==gp.GRB.OPTIMAL,independently_verified=True)
        finally:model.dispose()


def _distance(a, b):
    return sum(abs(float(x) - float(y)) for x, y in zip(a, b))


def _shared_arcs(routes, vehicles, terminal):
    return {(i, j) for k in vehicles
            for route in [list(routes.get(k, routes.get(str(k), [])))] if route
            for i, j in zip([0] + route, route + [terminal])}


def _time_bounds(tasks, transport, own, pickup_max, speed):
    """Bound earliest completion times using instance data alone.

    A precedence path visits at most all requests. Summing the largest
    per-request time increments bounds its completion without a feasible
    route, an initial solution, or an objective cutoff.
    """
    increments = []
    for j in range(1, len(tasks) + 1):
        handling = float(tasks[j - 1][6])
        if not math.isfinite(handling) or handling < 0:
            raise ValueError('Handling times must be finite and nonnegative')
        longest_transport = max(value for (i, k), value in transport.items() if k == j)
        increments.append(46.0 + (pickup_max + longest_transport + own[j]) / speed + handling)
    horizon = math.nextafter(math.fsum(increments), math.inf)
    if not math.isfinite(horizon):
        raise ValueError('The instance must have finite coordinates and times')
    upper_d = {j: horizon for j in range(1, len(tasks) + 1)}
    upper_m = {j: horizon - float(tasks[j - 1][6]) for j in upper_d}
    upper_s = {j: upper_m[j] - 46.0 - (own[j] + min(value for (i, k), value in transport.items() if k == j)) / speed for j in upper_d}
    upper_late = {j: max(0.0, horizon - float(tasks[j - 1][5])) for j in upper_d}
    return upper_s, upper_m, upper_d, upper_late


def _build_shared_depot(instance, mothers=4, daughters=8,
                speed=1.2, weight=.1, env=None, fixed_routes=None,
                distance_cuts=False):
    assert mothers >= 1 and daughters >= 1 and speed > 0 and 0 < weight < 1
    n, terminal = len(instance), len(instance) + 1
    assert n > 0
    tasks = list(range(1, n + 1))
    pred = {j: [i for i in range(n + 1) if i != j] for j in tasks}
    arcs = [(i, j) for j in tasks for i in pred[j]] + [(i, terminal) for i in tasks]
    destination = {0: (0., 0.), **{j: tuple(map(float, instance[j - 1][3:5])) for j in tasks}}
    own = {j: _distance(instance[j - 1][1:3], destination[j]) for j in tasks}
    transport = {(i, j): _distance(destination[i], instance[j - 1][1:3]) for j in tasks for i in pred[j]}
    pickup_max = sum(max(p[c] for p in destination.values()) - min(p[c] for p in destination.values()) for c in (0, 1))
    min_transport = {j: min(transport[i, j] for i in pred[j]) for j in tasks}
    lower_d = {j: 46 + (min_transport[j] + own[j]) / speed + float(instance[j - 1][6]) for j in tasks}
    lower_late = {j: max(0., lower_d[j] - float(instance[j - 1][5])) for j in tasks}
    upper_s, upper_m, upper_d, upper_late = _time_bounds(instance, transport, own, pickup_max, speed)
    model = gp.Model('NVSP_compact', env=env)
    a = model.addVars(arcs, vtype=GRB.BINARY, name='mother_arc')
    b = model.addVars(arcs, vtype=GRB.BINARY, name='daughter_arc')
    sync = model.addVars(tasks, lb=0, ub=upper_s, name='sync')
    mt = model.addVars(tasks, lb={j: lower_d[j] - float(instance[j - 1][6]) for j in tasks}, ub=upper_m, name='mother_time')
    dt = model.addVars(tasks, lb=lower_d, ub=upper_d, name='daughter_time')
    late = model.addVars(tasks, lb=lower_late, ub=upper_late, name='tardiness')
    px = model.addVars(tasks, lb=0, ub=pickup_max, name='pickup_x')
    py = model.addVars(tasks, lb=0, ub=pickup_max, name='pickup_y')
    pickup = {j: px[j] + py[j] for j in tasks}
    daughter_travel = {j: gp.quicksum(transport[r, j] * b[r, j] for r in pred[j]) for j in tasks}
    objective = weight * gp.quicksum(pickup[j] + daughter_travel[j] + own[j] for j in tasks) + (1 - weight) * late.sum()
    model.setObjective(objective)
    for variables, count, label in [(a, mothers, 'mother'), (b, daughters, 'daughter')]:
        for j in tasks:
            model.addConstr(gp.quicksum(variables[i, j] for i in pred[j]) == 1, name=label + '_in')
            model.addConstr(gp.quicksum(variables[j, r] for r in tasks + [terminal] if r != j) == 1, name=label + '_out')
        model.addConstr(gp.quicksum(variables[0, j] for j in tasks) <= count, name=label + '_fleet_limit')
        model.addConstr(gp.quicksum(variables[0, j] for j in tasks) == gp.quicksum(variables[j, terminal] for j in tasks), name=label + '_paths')
    for j in tasks:
        for coordinate, variable in [(0, px[j]), (1, py[j])]:
            difference = gp.quicksum(destination[i][coordinate] * a[i, j] for i in pred[j]) - gp.quicksum(destination[r][coordinate] * b[r, j] for r in pred[j])
            model.addConstr(variable >= difference)
            model.addConstr(variable >= -difference)
        model.addConstr(pickup[j] <= pickup_max)
        model.addConstr(mt[j] == sync[j] + 46 + (daughter_travel[j] + own[j]) / speed)
        model.addConstr(dt[j] == mt[j] + float(instance[j - 1][6]))
        model.addConstr(late[j] >= dt[j] - float(instance[j - 1][5]))
        for i in pred[j]:
            previous = mt[i] if i else 0.
            big_m = (upper_m[i] if i else 0.) + pickup_max / speed
            model.addConstr(sync[j] >= previous + pickup[j] / speed - big_m * (1 - a[i, j]))
        for r in pred[j]:
            previous = dt[r] if r else 0.
            big_m = upper_d[r] if r else 0.
            model.addConstr(sync[j] >= previous - big_m * (1 - b[r, j]))
        if distance_cuts:
            for i in pred[j]:
                exact = gp.quicksum(_distance(destination[i], destination[r]) * b[r, j] for r in pred[j])
                model.addConstr(pickup[j] >= exact - pickup_max * (1 - a[i, j]))
    if fixed_routes is not None:
        for variables, vehicles in [(a, range(1, mothers + 1)), (b, range(mothers + 1, mothers + daughters + 1))]:
            selected = _shared_arcs(fixed_routes, vehicles, terminal)
            for arc, variable in variables.items():
                variable.LB = variable.UB = int(arc in selected)
    model.update()
    return model, dict(mother=a, daughter=b, tasks=tasks, terminal=terminal,
                       mothers=mothers, daughters=daughters, bounds=dict(upper_d=upper_d, upper_m=upper_m))


def _extract_shared_routes(variables):
    n, end = len(variables['tasks']), variables['terminal']
    result = {}
    for label, count, offset in [('mother', variables['mothers'], 0),
                                 ('daughter', variables['daughters'], variables['mothers'])]:
        active = {(i, j) for (i, j), variable in variables[label].items() if variable.X > .5}
        starts = sorted(j for i, j in active if i == 0)
        assert len(starts) <= count
        visited = []
        for k, first in enumerate(starts, 1):
            route, current = [], first
            while current != end:
                assert current not in route and 1 <= current <= n
                route.append(current)
                successors = [j for i, j in active if i == current]
                assert len(successors) == 1
                current = successors[0]
            result[offset + k] = route
            visited.extend(route)
        assert sorted(visited) == variables['tasks']
        for k in range(len(starts) + 1, count + 1):
            result[offset + k] = []
    return result


def _build_distinct_starts(tasks,starts,env):
    starts={int(k):tuple(v) for k,v in starts.items()};n=len(tasks);end=n+1
    dest={**{-k:p for k,p in starts.items()},**{j:tuple(tasks[j-1][3:5]) for j in range(1,n+1)}}
    predm={j:list(range(-4,0))+[i for i in range(1,n+1) if i!=j] for j in range(1,n+1)}
    predd={j:list(range(-12,-4))+[i for i in range(1,n+1) if i!=j] for j in range(1,n+1)}
    own={j:_distance(tasks[j-1][1:3],dest[j]) for j in range(1,n+1)}
    travel={(r,j):_distance(dest[r],tasks[j-1][1:3]) for j in predd for r in predd[j]}
    mintravel={j:min(travel[r,j] for r in predd[j]) for j in predd}
    maxpickup=sum(max(p[c] for p in dest.values())-min(p[c] for p in dest.values()) for c in [0,1])
    lowerd={j:46+(mintravel[j]+own[j])/1.2+tasks[j-1][6] for j in predd}
    lowerlate={j:max(0.,lowerd[j]-tasks[j-1][5]) for j in predd}
    uppers, upperm, upperd, upperlate = _time_bounds(tasks, travel, own, maxpickup, 1.2)
    model=gp.Model('NVSP_distinct_factory_starts',env=env)
    arcs_m=[(i,j) for j in predm for i in predm[j]]+[(j,end) for j in predm]
    arcs_d=[(i,j) for j in predd for i in predd[j]]+[(j,end) for j in predd]
    a=model.addVars(arcs_m,vtype=GRB.BINARY,name='mother_arc')
    b=model.addVars(arcs_d,vtype=GRB.BINARY,name='daughter_arc')
    sync=model.addVars(list(predm),lb=0,ub=uppers,name='sync')
    mt=model.addVars(list(predm),lb={j:lowerd[j]-tasks[j-1][6] for j in predm},ub=upperm,name='mother_time')
    dt=model.addVars(list(predm),lb=lowerd,ub=upperd,name='daughter_time')
    late=model.addVars(list(predm),lb=lowerlate,ub=upperlate,name='tardiness')
    px=model.addVars(list(predm),lb=0,ub=maxpickup,name='pickup_x')
    py=model.addVars(list(predm),lb=0,ub=maxpickup,name='pickup_y')
    pickup={j:px[j]+py[j] for j in predm}
    transport={j:gp.quicksum(travel[r,j]*b[r,j] for r in predd[j]) for j in predd}
    objective=.1*gp.quicksum(pickup[j]+transport[j]+own[j] for j in predm)+.9*late.sum()
    model.setObjective(objective)
    for variables,pred,vehicles in [(a,predm,range(1,5)),(b,predd,range(5,13))]:
        for j in pred:
            model.addConstr(gp.quicksum(variables[i,j] for i in pred[j])==1)
            model.addConstr(gp.quicksum(variables[j,r] for r in list(pred)+[end] if r!=j)==1)
        for k in vehicles:model.addConstr(gp.quicksum(variables[-k,j] for j in pred)<=1)
    for j in predm:
        for coordinate,var in [(0,px[j]),(1,py[j])]:
            diff=gp.quicksum(dest[i][coordinate]*a[i,j] for i in predm[j])-gp.quicksum(dest[r][coordinate]*b[r,j] for r in predd[j])
            model.addConstr(var>=diff);model.addConstr(var>=-diff)
        model.addConstr(pickup[j]<=maxpickup)
        model.addConstr(mt[j]==sync[j]+46+(transport[j]+own[j])/1.2)
        model.addConstr(dt[j]==mt[j]+tasks[j-1][6]);model.addConstr(late[j]>=dt[j]-tasks[j-1][5])
        for i in predm[j]:
            previous=mt[i] if i>0 else 0.
            big=(upperm[i] if i>0 else 0.)+maxpickup/1.2
            model.addConstr(sync[j]>=previous+pickup[j]/1.2-big*(1-a[i,j]))
        for r in predd[j]:
            previous=dt[r] if r>0 else 0.
            model.addConstr(sync[j]>=previous-(upperd[r] if r>0 else 0.)*(1-b[r,j]))
    model.update()
    return model,dict(mother=a,daughter=b,n=n,end=end)

def _extract_distinct_routes(values):
    routes={};n=values['n'];end=values['end']
    for label,vehicles in [('mother',range(1,5)),('daughter',range(5,13))]:
        selected={i:j for (i,j),v in values[label].items() if float(v)>.5}
        visited=[]
        for k in vehicles:
            route=[];node=selected.get(-k,end)
            while node!=end:
                assert node not in route and len(route)<n
                route.append(node);node=selected[node]
            routes[k]=route;visited+=route
        assert sorted(visited)==list(range(1,n+1))
    return routes
