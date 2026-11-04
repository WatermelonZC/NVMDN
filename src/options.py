import os
import time
import argparse
import torch

launcher = 'torchrun' if 'TORCHRUN' in os.environ else 'pytorch'
def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Train NVMDN for nested-vehicle scheduling")

    # Data

    parser.add_argument('--problem', choices=['nvsp'], default='nvsp', help='Nested-Vehicle Scheduling Problem')
    parser.add_argument('--graph_size', type=int, default=50, help="The size of the problem graph")
    parser.add_argument('--mom_size', type=int, default=4, help="The size of the mom vehicle")
    parser.add_argument('--sub_size', type=int, default=8, help="The size of the sub vehicle")
    parser.add_argument('--batch_size', type=int, default=1024, help='Number of instances per batch during training')
    parser.add_argument('--epoch_size', type=int, default=1280000, help='Number of instances per epoch during training')
    parser.add_argument('--val_size', type=int, default=10000,
                        help='Number of instances used for reporting validation performance')
    parser.add_argument('--val_dataset', type=str, default=None, help='Dataset file to use for validation')

    # Model parameter 1 (add 1)
    parser.add_argument('--sychonization_cost', type=bool, default=True, help="cost add travel distance and tardiness'")
    parser.add_argument('--use_vehicle_context', type=bool, default=True, help='Use the current vehicle state in the decoder')
    parser.add_argument('--use_cross_attention', type=bool, default=True, help='Use task and vehicle cross-attention')
    parser.add_argument('--designated_driver', type=bool, default=True, help='if true use designated time as node embedding')
    parser.add_argument('--variable_instance', type=bool, default=True, help="Model, 'attention' (default) or 'pointer'")
    parser.add_argument('--dist_threshold', type=float, default=None, help="NVMDN standardized distance threshold")
    parser.add_argument('--time_threshold', type=float, default=None, help="NVMDN standardized time-difference threshold")
    parser.add_argument('--encoder_variant', choices=['nvmdn','am','hadrl','tdrl'], default='nvmdn', help='Policy encoder')
    parser.add_argument('--request_encoder', choices=['task_robot_cross','standard','tdrl'], default='task_robot_cross', help='Separate task/vehicle self-attention followed by dynamic cross-attention')
    parser.add_argument('--checkpoint_dynamic_steps', action=argparse.BooleanOptionalAction, default=None,
                        help='Recompute dynamic encoding/decoding during backward to reduce activation memory')
    parser.add_argument('--cache_cross_query', action=argparse.BooleanOptionalAction, default=None,
                        help='Reuse static task Q projection within each NVMDN episode; auto on in throughput mode')
    parser.add_argument('--train_microbatch_size', type=int, default=None,
                        help='CUDA graph microbatch; accumulate gradients before one effective-batch update')
    parser.add_argument('--decoder_variant', choices=['nvmdn', 'flat_decoder', 'standard', 'tdrl'], default='nvmdn')
    parser.add_argument('--decoder_ablation', choices=['none'], default='none')

    # Model parameter 2
    parser.add_argument('--model', default='attention', choices=['attention','am','hadrl','tdrl'], help='Policy model')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_encode_layers', type=int, default=1,
                        help='Number of layers in the encoder/critic network')
    parser.add_argument('--tanh_clipping', type=float, default=10.,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' (default) or 'instance'")

    # Training
    parser.add_argument('--speed_profile', choices=['auto', 'standard', 'throughput'], default='auto',
                        help='Default auto uses the accelerated runtime on supported single-GPU NVSP training')
    parser.add_argument('--tensor_loader', action=argparse.BooleanOptionalAction, default=None,
                        help='Iterate compact tensor batches directly without Python lists or DataLoader dispatch')
    parser.add_argument('--reuse_baseline', action=argparse.BooleanOptionalAction, default=None,
                        help='Refresh rollout baseline weights in place and reuse its evaluation graphs')
    parser.add_argument('--fast_train', action='store_true',
                        help='Use compact tensor datasets, batched loading, SDPA and fused Adam on CUDA')
    parser.add_argument('--debug_checks', action='store_true',
                        help='Enable synchronous decoder checks for debugging')
    parser.add_argument('--compile_decoder', action='store_true',
                        help='Compile decoder tensor operations with TorchInductor (CUDA/Linux, startup cost)')
    parser.add_argument('--compile_encoder', action='store_true', help='Compile encoder independently of decoder steps')
    parser.add_argument('--compile_mode', choices=['default', 'reduce-overhead', 'max-autotune'], default='default')
    parser.add_argument('--train_compile_mode', choices=['default', 'max-autotune-no-cudagraphs'], default='default',
                        help='Compilation mode used only when building the captured training graph')
    parser.add_argument('--fast_sampling', action='store_true', help='Categorical sampling without device-to-host checks')
    parser.add_argument('--vectorized_state', action='store_true', help='Use compact vectorized GPU environment state')
    parser.add_argument('--preload_mb', type=int, default=None, help='Maximum GPU-resident dataset size in MiB; auto profile defaults to 1024, zero disables')
    parser.add_argument('--prefetch_batches', action='store_true', help='Asynchronously copy the next batch on a CUDA stream')
    parser.add_argument('--cuda_graph_train', action='store_true', help='Capture forward/backward after rollout baseline warmup (fixed full batches)')
    parser.add_argument('--cuda_graph_rollout', action='store_true', help='Capture greedy baseline/validation forward for fixed full batches')
    parser.add_argument('--optimized_attention', action='store_true', help='Pack vehicle attention vehicle projections and fuse relation masks')
    parser.add_argument('--attention_backend', choices=['auto', 'sdpa', 'math', 'triton'], default='auto',
                        help='Short vehicle attention attention backend; auto selects tested Triton kernels on Turing CUDA GPUs')
    parser.add_argument('--compile_steps', action='store_true',
                        help='Compile complete NVSP decision steps, including state updates (CUDA/Linux)')
    parser.add_argument('--compile_chunk_size', type=int, choices=[1, 2, 4, 8, 16], default=None,
                        help='Decisions per compiled graph; auto selects the measured vehicle attention setting, 1 uses single steps')
    parser.add_argument('--precision', choices=['auto', 'fp32', 'bf16', 'fp16'], default='auto')
    parser.add_argument('--cpu_threads', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--eval_amp', action='store_true', help='Also use mixed precision for greedy baseline and validation')
    parser.add_argument('--baseline_eval_amp', action='store_true',
                        help='Mixed precision only for training baseline labels; keep validation and baseline selection FP32')
    # 三个分布式执行的参数
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default=launcher,
        help='job launcher'),
    parser.add_argument('--sync_bn', action='store_true', default=False, help='Use synchronized batch normalization')

    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')
    parser.add_argument('--eval_only', action='store_true', help='Set this value to only evaluate model')
    parser.add_argument('--n_epochs', type=int, default=100, help='The number of epochs to train')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed to use')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument('--max_clip_value', type=float, default=0.5,
                        help='clip_grad_value, default 1.0 (0 to disable clipping)')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--exp_beta', type=float, default=0.8,
                        help='Exponential moving average baseline decay (default 0.8)')
    parser.add_argument('--baseline', default="rollout",
                        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help='Significance in the t-test for updating rollout baseline')
    parser.add_argument('--bl_warmup_epochs', type=int, default=None,
                        help='Number of epochs to warmup the baseline, default None means 1 for rollout (exponential '
                             'used for warmup phase), 0 otherwise. Can only be used with rollout baseline.')
    parser.add_argument('--eval_batch_size', type=int, default=1024,
                        help="Batch size to use during (baseline) evaluation")
    parser.add_argument('--checkpoint_encoder', action='store_true',
                        help='Set to decrease memory usage by checkpointing encoder')
    parser.add_argument('--shrink_size', type=int, default=None,
                        help='Shrink the batch size if at least this many instances in the batch are finished'
                             ' to save memory (default None means no shrinking)')
    parser.add_argument('--data_distribution', type=str, default=None,
                        help='Data distribution to use during training, defaults and options depend on problem.')

    # Misc
    parser.add_argument('--log_step', type=int, default=50, help='Log info every log_step steps')
    parser.add_argument('--log_dir', default='', help='Directory to write TensorBoard information to')
    parser.add_argument('--run_name', default='run', help='Name to identify the run')
    parser.add_argument('--output_dir', default='outputs', help='Directory to write output models to')
    parser.add_argument('--epoch_start', type=int, default=0,
                        help='Start at epoch # (relevant for learning rate decay)')
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help='Save checkpoint every n epochs (default 1), 0 to save no checkpoints')
    parser.add_argument('--load_path', help='Path to load model parameters and optimizer state from')
    parser.add_argument('--resume', help='Resume from previous checkpoint file')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')

    opts = parser.parse_args(args)
    if opts.debug_checks and (opts.compile_steps or opts.compile_decoder):
        parser.error('--debug_checks cannot be combined with decoder compilation')

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        opts.output_dir,
        "{}_{}".format(opts.problem, opts.graph_size),
        opts.run_name,
        f"rank_{opts.local_rank}" if opts.launcher else None  # 为每个进程添加一个唯一的目录
    )
    if opts.bl_warmup_epochs is None:
        opts.bl_warmup_epochs = 1 if opts.baseline == 'rollout' else 0
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    assert (opts.bl_warmup_epochs == 0) or (opts.baseline == 'rollout')
    assert opts.epoch_size % opts.batch_size == 0, "Epoch size must be integer multiple of batch size!"
    return opts
