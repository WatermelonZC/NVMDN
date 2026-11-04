"""Solve one released instance and independently replay its schedule."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from portable import METHODS,read_instance,replay,observation,select_model


def learned(args,data):
    entry,path=select_model(args.method,data,args.checkpoint,args.training_size,getattr(args,'variant','full'))
    import torch
    from utils import load_model
    from utils.data_utils import MAX_VALUES
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is not available')
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Keep stdout machine readable; the preserved loader reports to stderr.
    with contextlib.redirect_stdout(sys.stderr):model,config=load_model(str(path))
    if getattr(args,'variant','full')=='flat_decoder' and model.decoder_variant!='flat_decoder':
        raise ValueError('--variant flat_decoder requires a Flat Decoder checkpoint')
    if model.decoder_variant=='flat_decoder':entry['method']='Flat Decoder (FD)'
    m,d=data['mother'],data['daughter'];n=len(data['tasks'])
    for module in model.modules():
        for name,value in (('mom_vehicle_size',m),('sub_vehicle_size',d),('vehicle_size',m+d),
                ('mothers',m),('daughters',d)):
            if hasattr(module,name):setattr(module,name,value)
        for name,value in (('compile_steps',False),('compile_decoder',False),('compile_encoder',False),
                ('use_sdpa',True),('fast_sampling',False),('debug_checks',False),('vectorized_state',True),
                ('checkpoint_dynamic_steps',False)):
            if hasattr(module,name):setattr(module,name,value)
    device=torch.device(args.device);model=model.to(device).float().eval()
    MAX_VALUES['time']=340.+40*n
    samples=1 if args.decode=='greedy' else 1024
    model.set_decode_type('greedy' if samples==1 else 'sampling')
    rows=data['tasks']
    batch=dict(sor_loc=torch.tensor([[r[1:3] for r in rows]],dtype=torch.float32),
        tar_loc=torch.tensor([[r[3:5] for r in rows]],dtype=torch.float32),
        designated_time=torch.tensor([[[r[5]] for r in rows]],dtype=torch.float32),
        tau_h=torch.tensor([[[r[6]] for r in rows]],dtype=torch.float32),
        mom_size=torch.tensor([m]),sub_size=torch.tensor([d]),
        params=torch.tensor([[8.,8.,30.]]),v=torch.tensor([[1.2]]))
    if 'initial_positions' in data:
        positions=data['initial_positions']
        batch['initial_positions']=torch.tensor([[positions[str(v)] for v in list(range(m+1,m+d+1))+list(range(1,m+1))]],dtype=torch.float32)
    expanded={k:v.expand(samples,*v.shape[1:]).contiguous().to(device) for k,v in batch.items()}
    if device.type=='cuda':torch.cuda.synchronize()
    started=time.perf_counter()
    with torch.inference_mode():
        costs,_,actions=model(expanded,return_pi=True)
        objective=.1*costs[0].float()+.9*costs[1].float()
        chosen=int(objective.argmin());network=float(objective[chosen])
        raw=actions[chosen].cpu().tolist()
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    plan=[[int(t)+1,int(mother)-d+1,int(daughter)+m+1] for t,daughter,mother in raw]
    result=replay(data,plan)
    if abs(result['objective']-network)>.05+2e-5*abs(network):
        raise RuntimeError('Independent physical replay does not match the model objective')
    return dict(method=entry['method'],decode=args.decode,checkpoint=path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        seed=args.seed,
        actions=plan,result=result,elapsed_seconds=elapsed,network_objective=network,
        timing_scope='single FP32 inference call, excludes model loading and input transfer; no warmup',
        independently_verified=True)


def solve(args):
    data=read_instance(args.instance)
    if args.method in METHODS:return learned(args,data)
    if args.method=='edd':
        from baselines.edd import rollout_policy
        started=time.perf_counter();plan=rollout_policy(observation(data));elapsed=time.perf_counter()-started
        return dict(method='EDD',actions=plan,result=replay(data,plan),elapsed_seconds=elapsed,independently_verified=True)
    if args.method=='gurobi':
        from baselines.gurobi import solve as solver
        return solver(data,args.timeout,args.seed)
    from baselines.search import solve as search
    return search(data,args.method.upper(),args.timeout,args.seed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instance',type=Path,required=True)
    p.add_argument('--method',choices=[*METHODS,'edd','ga','alns','diwo','gurobi'],default='nvmdn')
    p.add_argument('--decode',choices=['greedy','sample1024'],default='greedy')
    p.add_argument('--variant',choices=['full','flat_decoder'],default='full')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--training-size',type=int,help='Select a training scale for zero-shot evaluation')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--timeout',type=float,help='Override the search budget in configs/')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.variant!='full' and args.method!='nvmdn':p.error('--variant applies to --method nvmdn')
    if args.timeout is not None and args.timeout<=0:p.error('--timeout must be positive')
    record=solve(args)
    text=json.dumps(record,indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text+'\n',encoding='utf-8')
    print(text)


if __name__=='__main__':main()
