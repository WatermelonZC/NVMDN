import utils.dist_utils
from utils import dist_utils
import torch


def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts, extra_metrics=None):
    grad_norms, grad_norms_clipped = grad_norms
    metrics = {'avg_cost': cost.mean(), 'grad_norm': grad_norms[0],
               'grad_norm_clipped': grad_norms_clipped[0]}
    if not opts.no_tensorboard:
        metrics.update(actor_loss=reinforce_loss, nll=-log_likelihood.mean())
        if opts.baseline == 'critic':
            metrics.update(critic_loss=bl_loss, critic_grad_norm=grad_norms[1],
                           critic_grad_norm_clipped=grad_norms_clipped[1])
        metrics.update(extra_metrics or {})
    # One small host transfer per logging event rather than one synchronization
    # for every scalar and formatted CUDA tensor.
    values = torch.stack([torch.as_tensor(value, device=cost.device).detach().reshape(())
                          for value in metrics.values()]).cpu().tolist()
    metrics = dict(zip(metrics, values))
    cost = metrics['avg_cost']
    # Log values to screen
    if (opts.distributed and opts.local_rank == 0) or not opts.distributed:
    #     print('rank: {}, epoch: {}, train_batch_id: {}, avg_cost: {}'.format(opts.local_rank, epoch, batch_id, cost))
    #     print('rank: {}, grad_norm: {}, clipped: {}'.format(opts.local_rank, grad_norms[0], grad_norms_clipped[0]))
    # else:
        print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, cost))
        print('grad_norm: {}, clipped: {}'.format(metrics['grad_norm'], metrics['grad_norm_clipped']))
    # Log values to tensorboard
    if not opts.no_tensorboard:
        for name, value in metrics.items():
            tb_logger.add_scalar(name, value, step)


import logging
import os
from datetime import datetime

loggers = {}


def createLogger(name):
    global loggers
    if loggers.get(name) and len(loggers[name].handlers) != 0:
        return loggers.get(name)
    else:
        if not os.path.exists('./logs'):
            os.mkdir('./logs')
        # 创建一个logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        # 创建一个文件处理器，用于处理info级别的日志
        info_handler = logging.FileHandler(
            './logs/info_{}.log'.format(datetime.now().strftime('%Y_%m_%d_%H')), encoding="utf-8", mode="a")
        info_handler.setLevel(logging.INFO)

        # 创建一个格式化器
        formatter = logging.Formatter('%(asctime)s | %(module)s | %(message)s')

        # 为处理器添加格式化器
        info_handler.setFormatter(formatter)

        # 将处理器添加到logger
        logger.addHandler(info_handler)

    loggers[name] = logger

    return logger


