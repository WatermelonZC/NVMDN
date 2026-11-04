import logging
import os
import time
import json

from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import torch
import math

from torch.nn import DataParallel
import torch.distributed as dist
from nets.attention_model import set_decode_type
from problems.nvsp.state import omega
from utils.log_utils import log_values
from utils import move_to, dist_utils, ModelData
from utils.training_performance import make_loader, configure_precision, autocast_context, preload_dataset
from utils.cuda_graph_runtime import TrainingGraph, AccumulatedTrainingGraph, graph_compatible, get_rollout_graph


# 是否开启混合精度训练


def get_inner_model(model):
    return model.module if isinstance(model, (DataParallel, DistributedDataParallel)) else model


def validate(model, dataset, opts):
    # Validate
    if not opts.distributed or (opts.distributed and opts.local_rank == 0):
        print('[Rank 0] Validating...')
    cost = rollout(model, dataset, opts)
    # if opts.distributed and opts.local_rank != 0:
    #     return None
    avg_cost = cost.mean()
    if not opts.distributed or (opts.distributed and opts.local_rank == 0):
        print('[Rank 0] Validation overall avg_cost: {} +- {}'.format(
            avg_cost, torch.std(cost) / math.sqrt(len(cost))))
    return avg_cost


def rollout(model, dataset, opts):
    set_decode_type(model, "greedy")
    model.eval()
    dataset = preload_dataset(dataset, opts)
    graph = None
    use_graph = (opts.use_cuda and getattr(opts, 'cuda_graph_rollout', False)
                 and graph_compatible(model) and len(dataset) >= 4 * opts.eval_batch_size)

    def eval_model_bat(bat):
        nonlocal graph
        with torch.no_grad(), autocast_context(opts, evaluation=True):
            data = move_to(bat, opts.device, non_blocking=True)
            if use_graph and data['sor_loc'].size(0) == opts.eval_batch_size:
                if graph is None:
                    amp_key = (getattr(opts, 'eval_amp', False), getattr(opts, 'precision', 'auto'),
                               getattr(ModelData, 'amp_dtype', None))
                    graph = get_rollout_graph(model, data, lambda: autocast_context(opts, evaluation=True), amp_key)
                cost = graph(data)
            elif opts.problem == 'nvsp' and getattr(get_inner_model(model), 'vectorized_state', False):
                cost = model(data, cost_only=True)
            else:
                cost, _ = model(data)
            if opts.problem == 'nvsp' and opts.sychonization_cost:
                cost = omega * cost[0] + (1 - omega) * cost[1]
        return cost.detach() if opts.use_cuda and getattr(opts, 'fast_train', False) else cost.detach().cpu()

    # 分布式采样器
    sampler = None
    # if opts.distributed and opts.local_rank != 0:
    #     sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)

    # BaselineDataset relies on these values having exactly dataset order.
    dataloader = make_loader(dataset, opts.eval_batch_size, opts, shuffle=False)

    results = []
    host_results, offset = None, 0

    # 必须所有的GPU都参与运算,否则会卡住
    # if opts.distributed and opts.local_rank != 0:
    #     return None

    for bat in tqdm(dataloader, disable=opts.local_rank != 0 if opts.distributed else opts.no_progress_bar):
        batch_cost = eval_model_bat(bat)
        if batch_cost.is_cuda:
            if host_results is None:
                host_results = torch.empty(len(dataset), dtype=batch_cost.dtype, pin_memory=True)
            count = batch_cost.size(0)
            host_results[offset:offset + count].copy_(batch_cost, non_blocking=True)
            offset += count
        else:
            results.append(batch_cost)
    if host_results is not None:
        torch.cuda.current_stream(opts.device).synchronize()
        results = host_results
    else:
        results = torch.cat(results, 0)
    # Ensure only rank 0 returns the result
    # 最终返回 CPU tensor，确保主进程返回结果
    return results


def clip_grad_norms(param_groups, max_norm=math.inf, clip_value=None):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param clip_value:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = []
    for group in param_groups:
        # 先进行范数裁剪
        group_norm = torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm=max_norm if max_norm > 0 else math.inf,
            norm_type=2
        )
        grad_norms.append(group_norm.detach())

        # 可选：逐元素梯度裁剪（应对极端异常值）
        if clip_value is not None:
            torch.nn.utils.clip_grad_value_(
                group['params'],
                clip_value=clip_value
            )

    grad_norms_clipped = [g_norm.clamp(max=max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("rank {} Start train epoch {}, lr={} for run {}".format(opts.local_rank, epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)

    configure_precision(opts)

    if not opts.no_tensorboard:
        tb_logger.add_scalar('learnrate_pg0', optimizer.param_groups[0]['lr'], step)
    # Generate new training data for each epoch
    start_time = time.time()
    baseline_seconds = 0.0
    if opts.problem == 'nvsp':
        training_dataset = problem.make_dataset(
            size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution,
            mom_vehicle_size=opts.mom_size, sub_vehicle_size=opts.sub_size, designated_driver=opts.designated_driver
            , variable_instance=opts.variable_instance,
            **({"compact": True} if opts.problem == "nvsp" and getattr(opts, "fast_train", False) else {}))
        training_dataset = preload_dataset(training_dataset, opts)
        generation_seconds = time.time() - start_time
        baseline_start = time.time()
        training_dataset = baseline.wrap_dataset(training_dataset)
        baseline_seconds = time.time() - baseline_start
        training_dataset = preload_dataset(training_dataset, opts)
    else:
        training_dataset = baseline.wrap_dataset(problem.make_dataset(
            size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution))
    train_sampler = None
    if opts.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset, shuffle=True)
        train_sampler.set_epoch(epoch)
    training_dataloader = make_loader(
        training_dataset, opts.batch_size, opts, shuffle=True,
        sampler=train_sampler, drop_last=opts.distributed)

    model.train()
    set_decode_type(model, "sampling")

    # 打印各进程分配到的batch数
    print(f"Rank {opts.local_rank} will process {len(training_dataloader)} batches")
    training_start = time.time()
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=(opts.local_rank != 0 if opts.distributed else opts.no_progress_bar))):
        try:
            train_batch(
                model,
                optimizer,
                baseline,
                epoch,
                batch_id,
                step,
                batch,
                tb_logger,
                opts
            )
            step += 1
        except Exception as e:
            if opts.distributed:
                print(f"Rank {opts.local_rank} error in batch {batch_id}: {str(e)}")
                torch.distributed.destroy_process_group()
                raise
            raise

    if opts.use_cuda:
        torch.cuda.synchronize(opts.device)
    training_seconds = time.time() - training_start
    epoch_duration = time.time() - start_time
    print("rank {}, Finished epoch {}, took {} s \r\n".format(opts.local_rank, epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    last_epoch = opts.epoch_start + opts.n_epochs - 1
    should_save = (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == last_epoch
    checkpoint_start = time.time()
    if should_save and (not opts.distributed or opts.local_rank == 0):
        print('Saving model and state at ' + opts.save_dir)
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if opts.use_cuda else [],
                'baseline': baseline.state_dict(),
                'scaler': ModelData.scaler.state_dict() if ModelData.use_scaler else None,
                'encoder_variant': getattr(get_inner_model(model), 'encoder_variant', 'standard'),
                'request_encoder': getattr(get_inner_model(model), 'request_encoder', 'vehicle_encoder'),
                'decoder_variant': getattr(get_inner_model(model), 'decoder_variant', 'standard'),
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch)))

    checkpoint_seconds = time.time() - checkpoint_start
    validation_start = time.time()
    avg_reward = validate(model, val_dataset, opts)
    validation_seconds = time.time() - validation_start
    if opts.no_tensorboard or (opts.distributed and opts.local_rank != 0):
        pass
    else:
        tb_logger.add_scalar('val_avg_reward', avg_reward, step)

    callback_start = time.time()
    baseline.epoch_callback(model, epoch)
    baseline_callback_seconds = time.time() - callback_start
    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()
    if not opts.distributed or opts.local_rank == 0:
        metrics = dict(epoch=epoch, generation_seconds=locals().get('generation_seconds', 0),
                       baseline_seconds=baseline_seconds, training_seconds=training_seconds,
                       checkpoint_seconds=checkpoint_seconds, validation_seconds=validation_seconds,
                       baseline_callback_seconds=baseline_callback_seconds,
                       checkpoint_validation_callback_seconds=time.time() - start_time - epoch_duration,
                       total_seconds=time.time() - start_time,
                       samples_per_second=opts.epoch_size / training_seconds)
        print('EPOCH_TIMING ' + json.dumps(metrics))
        with open(os.path.join(opts.save_dir, 'epoch_timings.jsonl'), 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(metrics) + '\n')



def train_batch(
    model,
    optimizer,
    baseline,
    epoch,
    batch_id,
    step,
    batch,
    tb_logger,
    opts
):
    try:
        x, bl_val = baseline.unwrap_batch(batch)
        x = move_to(x, opts.device, non_blocking=True)
        bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

        amp_enabled = opts.use_cuda and getattr(opts, 'precision', 'auto') != 'fp32'
        use_graph = (opts.use_cuda and getattr(opts, 'cuda_graph_train', False)
                     and opts.problem == 'nvsp' and opts.sychonization_cost
                     and graph_compatible(model) and getattr(model, 'fast_sampling', False)
                     and bl_val is not None and x['sor_loc'].size(0) == opts.batch_size)
        if use_graph:
            runtime = getattr(opts, '_training_graph', None)
            if runtime is None:
                runtime_class = AccumulatedTrainingGraph if getattr(opts, 'train_microbatch_size', None) else TrainingGraph
                previous_mode = model.compile_mode
                model.compile_mode = getattr(opts, 'train_compile_mode', 'default')
                try:
                    runtime = runtime_class(model, x, bl_val, opts,
                                            ModelData.scaler if ModelData.use_scaler and amp_enabled else None,
                                            lambda: autocast_context(opts))
                finally:
                    model.compile_mode = previous_mode
                opts._training_graph = runtime
                opts._used_training_graph = True
            costs, cost, log_likelihood, reinforce_loss, loss = runtime(x, bl_val)
            cost_dist, cost_delay = costs
            bl_loss = 0
        else:
            optimizer.zero_grad(set_to_none=True)
            # During the first rollout-baseline warmup epoch, its exponential
            # baseline needs the full effective-batch mean. Keep the original
            # warmup update and temporarily use recomputation for this batch.
            recompute = getattr(model, 'checkpoint_dynamic_steps', False)
            if getattr(opts, 'train_microbatch_size', None) and bl_val is None:
                model.checkpoint_dynamic_steps = True
            try:
                with autocast_context(opts):
                    cost, log_likelihood = model(x)
                    if opts.problem == 'nvsp' and opts.sychonization_cost:
                        cost_dist, cost_delay = cost
                        cost = omega * cost_dist + (1 - omega) * cost_delay
                    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
            finally:
                model.checkpoint_dynamic_steps = recompute
            if getattr(model, 'uses_heterogeneous_attention', False):
                from utils.ha_policy_loss import ha_reinforce_loss
                reinforce_loss = ha_reinforce_loss(cost, bl_val, log_likelihood)
            else:
                reinforce_loss = ((cost.float() - bl_val) * log_likelihood.float()).mean()
            loss = reinforce_loss + bl_loss
            if not amp_enabled and getattr(model, 'request_encoder', None) != 'tdrl':
                loss = loss / opts.graph_size
        if not use_graph:
            (ModelData.scaler.scale(loss) if ModelData.use_scaler and amp_enabled else loss).backward()
        if ModelData.use_scaler and amp_enabled:
            ModelData.scaler.unscale_(optimizer)
            grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
            ModelData.scaler.step(optimizer)
            ModelData.scaler.update()
        else:
            grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
            optimizer.step()

        #
        # >>> CORRECT PLACEMENT FOR THE SCHEDULER STEP <<<
        # Update the learning rate *after* the optimizer has updated the parameters

        # Logging
        if step % int(opts.log_step) == 0:
            extra_metrics = ({'cost_dist': cost_dist.mean(), 'cost_delay': cost_delay.mean()}
                             if not opts.no_tensorboard and opts.problem == 'nvsp' and opts.sychonization_cost else None)
            if opts.distributed:
                avg_cost = dist_utils.reduce_tensor(cost, opts)
                avg_reinforce_loss = dist_utils.reduce_tensor(reinforce_loss, opts)
                # if opts.local_rank == 0:
                log_values(avg_cost, grad_norms, epoch, batch_id, step,
                           log_likelihood, avg_reinforce_loss, bl_loss, tb_logger, opts, extra_metrics)
                # log_values(cost, grad_norms, epoch, batch_id, step,
                #            log_likelihood, reinforce_loss, avg_bl_loss, tb_logger, opts)
            else:
                log_values(cost, grad_norms, epoch, batch_id, step,
                           log_likelihood, reinforce_loss, bl_loss, tb_logger, opts, extra_metrics)
        return loss.detach()

    except Exception as e:
        logging.error(f"[Step {step}] ❌ Error during training batch: {e}")
        # torch.save(x, f"debug_x_step_{step}.pt")  # 保存错误样本以便复现
        raise e

