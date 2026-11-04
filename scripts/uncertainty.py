"""Replay one fixed travel-time or operation-time scenario."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from execution import replay_scenario
from predict import solve


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',type=Path,required=True)
    p.add_argument('--method',choices=['nvmdn','edd'],default='nvmdn')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    scenario=json.loads(args.scenario.read_text())
    if scenario['kind'] not in ('BASE','TT','ET'):p.error('Expected a fixed duration scenario')
    index=scenario['index']
    base=ROOT/f'instances/main/n20/{index:02d}.json'
    run=SimpleNamespace(instance=base,method=args.method,device=args.device,seed=args.seed,
        decode='greedy',checkpoint=None,training_size=20,timeout=3600)
    nominal=solve(run)
    def unexpected(_):raise ValueError('Duration scenarios must not contain new task arrivals')
    result=replay_scenario(scenario,nominal['actions'],unexpected)
    result.update(method=args.method,seed=args.seed,planning_seconds=nominal['elapsed_seconds'])
    text=json.dumps(result,indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text+'\n',encoding='utf-8')
    print(text)


if __name__=='__main__':main()
