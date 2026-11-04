"""Physical execution uses the main experiment's anticipatory synchronization."""
def metric(a,b):return sum(abs(float(x)-float(y)) for x,y in zip(a,b))

def schedule(tasks,actions,now=0.,prefix=(),delay=0.):
    positions={v:[0.,0.] for v in range(1,13)}
    times={v:0. for v in positions};log=[]
    for d in prefix:
        m,w=d['mother'],d['daughter'];target=tasks[d['task']]['row'][3:5]
        positions[m]=positions[w]=list(target)
        times[m]=d['mother_end']+delay;times[w]=d['daughter_end']+delay;log.append(dict(d))
    for task,m,w in actions:
        t=tasks[task];row=t['row'];join=metric(positions[m],positions[w])
        transport=metric(positions[w],row[1:3])+metric(row[1:3],row[3:5])
        start=max(float(now),float(t['release']),times[m])
        coupled=max(start+join/1.2*t['tt_factor'],times[w])
        me=coupled+transport/1.2*t['tt_factor']+46.*t['et_factor']
        de=me+float(row[6])*t['et_factor']
        log.append(dict(task=task,mother=m,daughter=w,mother_start=start,coupled=coupled,
            mother_end=me,daughter_end=de,distance=join+transport,
            tardiness=max(0.,de-float(row[5])),release=t['release']))
        positions[m]=positions[w]=list(row[3:5]);times[m]=me+delay;times[w]=de+delay
    return log

def committed_at(log,now):
    """Once travel starts, its planned vehicle predecessors stay reserved."""
    chosen={d['task'] for d in log if d['mother_start']<now-1e-9}
    last={};dependencies={}
    for d in log:
        key=d['task'];dependencies[key]=set()
        for v in (d['mother'],d['daughter']):
            if v in last:dependencies[key].add(last[v])
            last[v]=key
    while True:
        expanded=chosen|set().union(*(dependencies[k] for k in chosen)) if chosen else set()
        if expanded==chosen:break
        chosen=expanded
    return [d for d in log if d['task'] in chosen]

def observation(tasks,prefix,now):
    positions={v:[0.,0.] for v in range(1,13)}
    times={v:float(now) for v in positions};last={v:None for v in positions}
    for d in prefix:
        for v,end in ((d['mother'],d['mother_end']),(d['daughter'],d['daughter_end'])):
            positions[v]=list(tasks[d['task']]['row'][3:5]);times[v]=max(float(now),end);last[v]=d['task']
    return dict(known={k:list(t['row']) for k,t in tasks.items()},committed={d['task'] for d in prefix},
        positions=positions,times=times,vehicle_last_task=last,time=now)

def replay_scenario(scenario,initial_actions,policy):
    all_tasks={t['id']:t for t in scenario['tasks']}
    known={k:t for k,t in all_tasks.items() if t['release']<=0}
    log=schedule(known,initial_actions,delay=scenario['communication_delay'])
    plans=[]
    # Duration perturbations execute the initially selected static Greedy plan.
    # Arrival cases are separate, with nominal physical durations known exactly.
    for now in sorted({float(t['release']) for t in all_tasks.values() if t['release']>0}):
        prefix=committed_at(log,now)
        known={k:t for k,t in all_tasks.items() if t['release']<=now}
        o=observation(known,prefix,now)
        actions,seconds=policy(o)
        assert {a[0] for a in actions}==set(known)-o['committed']
        log=schedule(known,actions,now,prefix,scenario['communication_delay'])
        plans.append(dict(time=now,known_ids=sorted(known),committed_ids=sorted(o['committed']),actions=actions,seconds=seconds))
    assert len(log)==len(all_tasks) and {d['task'] for d in log}==set(all_tasks)
    distance=sum(d['distance'] for d in log);tardiness=sum(d['tardiness'] for d in log)
    # Independent recurrence from recorded vehicle assignments and dispatch times.
    # Arrival repair introduces an explicit decision time for unstarted tasks.
    # Verify each individual transition using its logged start and previous state.
    positions={v:[0.,0.] for v in range(1,13)};times={v:0. for v in positions}
    for d in log:
        row=all_tasks[d['task']]['row'];t=all_tasks[d['task']];m,w=d['mother'],d['daughter']
        assert d['mother_start']>=max(times[m],t['release'])-1e-7
        join=metric(positions[m],positions[w]);transport=metric(positions[w],row[1:3])+metric(row[1:3],row[3:5])
        me=max(d['mother_start']+join/1.2*t['tt_factor'],times[w])+transport/1.2*t['tt_factor']+46*t['et_factor']
        assert abs(me-d['mother_end'])<1e-7 and abs(me+row[6]*t['et_factor']-d['daughter_end'])<1e-7
        positions[m]=positions[w]=list(row[3:5]);times[m]=me+scenario['communication_delay'];times[w]=d['daughter_end']+scenario['communication_delay']
    return dict(method=None,scenario_index=scenario['index'],kind=scenario['kind'],level=scenario['level'],
        task_count=len(all_tasks),objective=.1*distance+.9*tardiness,
        distance=distance,tardiness=tardiness,makespan=max(d['daughter_end'] for d in log),
        decisions=log,plans=plans,plan_calls=1+len(plans),planning_seconds=sum(p['seconds'] for p in plans),
        protocol='main_static_synchronization; duration replay; causal arrival repair',
        decoding='greedy',independently_verified=True)
