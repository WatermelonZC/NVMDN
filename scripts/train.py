"""Train the paper models using their released architecture configurations."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from portable import METHODS,select_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=int,choices=[10,20,50,100],required=True)
    p.add_argument('--method',choices=METHODS,default='nvmdn')
    p.add_argument('--variant',choices=['full','flat_decoder'],default='full')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--epochs',type=int,help='Total training epochs; default from configs/nvmdn.json')
    p.add_argument('--batch-size',type=int)
    p.add_argument('--epoch-size',type=int)
    p.add_argument('--val-size',type=int)
    args=p.parse_args()
    if args.epochs is not None and args.epochs<1:p.error('--epochs must be positive')
    if args.variant!='full' and args.method!='nvmdn':p.error('--variant applies to --method nvmdn')
    data=dict(tasks=[None]*args.tasks,mother=4,daughter=8)
    if args.variant=='flat_decoder' and args.tasks!=20:p.error('The Flat Decoder ablation uses --tasks 20')
    entry,path=select_model(args.method,data,training_size=args.tasks)
    import torch
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA is not available; use --device cpu')
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        p.error('Output directory must be empty for a fresh training run')
    output.mkdir(parents=True,exist_ok=True)
    from options import get_options
    opts=get_options(['--launcher','none'])
    config=json.loads(path.with_name('args.json').read_text())
    if args.variant=='flat_decoder':
        config.update(decoder_variant='flat_decoder',cache_cross_query=False,dist_threshold=None,time_threshold=None)
        entry['method']='Flat Decoder (FD)'
    common=json.loads((ROOT/'configs/nvmdn.json').read_text())
    config.update({key:value for key,value in common.items() if key not in ('epochs','normalization')})
    for key,value in config.items():setattr(opts,key,value)
    opts.graph_size=args.tasks;opts.epoch_start=0
    opts.n_epochs=args.epochs if args.epochs is not None else common['epochs']
    opts.batch_size=args.batch_size or config.get('batch_size',2048)
    opts.epoch_size=args.epoch_size or config.get('epoch_size',1280000)
    opts.val_size=args.val_size or config.get('val_size',10000)
    if opts.batch_size<1 or opts.epoch_size%opts.batch_size:p.error('epoch-size must be divisible by batch-size')
    opts.val_dataset=None;opts.load_path=None;opts.resume=str(args.resume.resolve()) if args.resume else None
    if args.resume:
        state=torch.load(args.resume,map_location='cpu',weights_only=False)
        if 'optimizer' not in state:p.error('--resume requires a full training checkpoint')
        from utils.training_resume import checkpoint_epoch
        opts.epoch_start=checkpoint_epoch(args.resume)+1
        opts.n_epochs=max(0,opts.n_epochs-opts.epoch_start)
    opts.training_profile='standard';opts._shared_profile_applied=False
    opts.launcher='none';opts.local_rank=0;opts.no_cuda=args.device=='cpu';opts.use_cuda=not opts.no_cuda
    opts.save_dir=str(output);opts.output_dir=str(output);opts.log_dir=str(output/'events')
    opts.run_name=f'{entry["method"]}_N{args.tasks}'
    # Use the preserved numerical implementation without machine-specific kernels.
    opts.speed_profile='standard';opts.resolved_speed_profile='standard';opts.precision='fp32'
    for key in ('compile_steps','compile_decoder','compile_encoder','cuda_graph_train','cuda_graph_rollout',
                'prefetch_batches','small_attention','native_amp_norm','checkpoint_dynamic_steps',
                'baseline_eval_amp','eval_amp','tdrl_single_query_attention','tdrl_direct_fast_decode'):
        setattr(opts,key,False)
    opts.fast_train=False;opts.tensor_loader=False;opts.vectorized_state=True;opts.fast_sampling=False
    opts.cpu_threads=4;opts.num_workers=0;opts.train_microbatch_size=None
    opts.data_distribution=None
    os.chdir(output)
    from run import run
    run(opts)


if __name__=='__main__':main()
