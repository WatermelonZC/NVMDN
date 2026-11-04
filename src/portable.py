"""Portable physical instances, model selection, and independent replay."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = {'nvmdn':'NVMDN','am':'AM','hadrl':'HA-DRL','tdrl':'TDRL'}


def read_instance(path):
    data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(data,dict) or not isinstance(data.get('tasks'),list) or not data['tasks']:
        raise ValueError('Expected an instance with a nonempty task list')
    rows=data['tasks'];n=len(rows)
    if data.get('size')!=n or not all(isinstance(r,list) and len(r)==7 for r in rows):
        raise ValueError('Expected size tasks with seven columns per row')
    if any(not isinstance(data.get(key),int) or data[key]<1 for key in ('mother','daughter')):
        raise ValueError('Vehicle counts must be positive integers')
    if [int(r[0]) for r in rows] != list(range(1,n+1)):
        raise ValueError('Task IDs must be consecutive and start at one')
    if data.get('speed',1.2)!=1.2 or data.get('params',[8,8,30])!=[8,8,30] or data.get('distance_weight',.1)!=.1:
        raise ValueError('The released environment uses speed 1.2, durations 8/8/30, and distance weight 0.1')
    return data


def observation(data):
    m,d=data['mother'],data['daughter']
    positions={v:[0.,0.] for v in range(1,m+d+1)}
    if 'initial_positions' in data:
        positions={int(k):list(v) for k,v in data['initial_positions'].items()}
    return dict(mothers=list(range(1,m+1)),daughters=list(range(m+1,m+d+1)),
        times={v:0. for v in positions},positions=positions,
        known={int(r[0]):r for r in data['tasks']},committed=set())


def replay(data,actions):
    o=observation(data);pos=o['positions'];ready=o['times'];distance=tardiness=0.;seen=[]
    metric=lambda a,b:sum(abs(float(x)-float(y)) for x,y in zip(a,b))
    m,d=data['mother'],data['daughter']
    for task,mother,daughter in actions:
        if not (1<=mother<=m and m<daughter<=m+d):raise ValueError('Invalid vehicle assignment')
        row=data['tasks'][task-1];source,target=row[1:3],row[3:5]
        join=metric(pos[mother],pos[daughter]);transport=metric(pos[daughter],source)+metric(source,target)
        finish=max(ready[mother]+join/1.2,ready[daughter])+8.+transport/1.2+30.+8.
        ready[mother],ready[daughter]=finish,finish+float(row[6])
        pos[mother],pos[daughter]=list(target),list(target)
        distance+=join+transport;tardiness+=max(0.,ready[daughter]-float(row[5]));seen.append(task)
    if sorted(seen)!=list(range(1,len(data['tasks'])+1)):raise ValueError('Each task must occur once')
    return dict(objective=.1*distance+.9*tardiness,distance=distance,tardiness=tardiness,feasible=True)


def routes_to_actions(routes,n,m,d):
    routes={int(k):v for k,v in routes.items()};pre={};assign={}
    for kind,vehicles in (('m',range(1,m+1)),('d',range(m+1,m+d+1))):
        seen=[]
        for vehicle in vehicles:
            previous=0
            for task in routes.get(vehicle,[]):
                pre[task,kind]=previous;assign[task,kind]=vehicle;previous=task;seen.append(task)
        if sorted(seen)!=list(range(1,n+1)):raise ValueError('Incomplete vehicle routes')
    pending=set(range(1,n+1));done={0};actions=[]
    while pending:
        ready=[t for t in sorted(pending) if pre[t,'m'] in done and pre[t,'d'] in done]
        if not ready:raise ValueError('Cyclic vehicle routes')
        for task in ready:
            actions.append([task,assign[task,'m'],assign[task,'d']]);done.add(task);pending.remove(task)
    return actions


def select_model(method,data,checkpoint=None,training_size=None,variant='full'):
    models=[]
    for folder in sorted((ROOT/'pretrained').glob('N=*')):
        if not folder.name[2:].isdigit():continue
        for label in METHODS.values():
            path=folder/('' if label=='NVMDN' else label)/'epoch-99.pt'
            if path.is_file():models.append(dict(method=label,size=int(folder.name[2:]),category='normal',checkpoint=path.relative_to(ROOT).as_posix()))
    if not models:raise FileNotFoundError('No released checkpoints in pretrained/N=*/')
    if checkpoint:
        path=Path(checkpoint).resolve()
        if path.is_dir():path=path/('model.pt' if (path/'model.pt').is_file() else 'epoch-99.pt')
        matches=[r for r in models if (ROOT/r['checkpoint']).resolve()==path]
        if matches:
            if matches[0]['method']!=METHODS[method]:raise ValueError('--method does not match the checkpoint')
            return matches[0],path
        if not path.is_file():raise FileNotFoundError(path)
        opts=json.loads((path.parent/'args.json').read_text())
        expected='attention' if method=='nvmdn' else method
        if opts.get('model','attention')!=expected:raise ValueError('--method does not match the checkpoint')
        return dict(method=METHODS[method],size=opts['graph_size']),path
    if variant=='flat_decoder':
        raise ValueError('Flat Decoder weights are not distributed. Supply --checkpoint with your own trained model')
    category='real-world' if data.get('scenario')=='industrial' else 'normal'
    distribution=data.get('distribution','uniform')
    if distribution!='uniform' and category=='normal':category='spatial'
    candidates=[r for r in models if r['method']==METHODS[method] and r['category']==category
        and (category!='spatial' or r.get('distribution')==distribution)]
    size=training_size or len(data['tasks'])
    if not candidates:raise ValueError('Only models trained on the main distribution are included. Supply --checkpoint for a separately trained model in this scenario')
    chosen=min(candidates,key=lambda r:abs(r['size']-size))
    if training_size is not None and chosen['size']!=training_size:
        raise ValueError('No released model at the requested training scale')
    return chosen,ROOT/chosen['checkpoint']
