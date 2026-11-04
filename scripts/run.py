"""Batch evaluation of released instances."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    evaluate=sub.add_parser('evaluate');evaluate.add_argument('--instances',type=Path,required=True)
    evaluate.add_argument('--method',choices=['nvmdn','am','hadrl','tdrl','edd','ga','alns','diwo','gurobi'],default='nvmdn')
    evaluate.add_argument('--output',type=Path,required=True)
    evaluate.add_argument('--decode',choices=['greedy','sample1024'],default='greedy')
    evaluate.add_argument('--variant',choices=['full','flat_decoder'],default='full')
    evaluate.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    evaluate.add_argument('--timeout',type=float);evaluate.add_argument('--seed',type=int,default=1234)
    evaluate.add_argument('--checkpoint',type=Path);evaluate.add_argument('--training-size',type=int)
    args=p.parse_args()
    if args.variant!='full' and args.method!='nvmdn':p.error('--variant applies to --method nvmdn')
    files=sorted(x for x in args.instances.glob('*.json') if x.stem.isdigit()
        or (x.stem.startswith('n') and x.stem[1:].isdigit()))
    if not files:p.error('No instance JSON files in this directory')
    args.output.mkdir(parents=True,exist_ok=True);records=[]
    for path in files:
        target=args.output/(path.stem+'.json')
        if target.exists():raise FileExistsError(target)
        command=[sys.executable,str(ROOT/'scripts/predict.py'),'--instance',str(path),'--method',args.method,
            '--decode',args.decode,'--variant',args.variant,'--device',args.device,
            '--seed',str(args.seed),'--output',str(target)]
        if args.checkpoint:command+=['--checkpoint',str(args.checkpoint)]
        if args.timeout is not None:command+=['--timeout',str(args.timeout)]
        if args.training_size:command+=['--training-size',str(args.training_size)]
        subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
        record=json.loads(target.read_text());records.append(dict(instance=path.name,**record['result'],elapsed_seconds=record['elapsed_seconds']))
        print(path.name,record['result']['objective'],flush=True)
    with (args.output/'results.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


if __name__=='__main__':main()
