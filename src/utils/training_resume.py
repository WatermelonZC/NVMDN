"""Preserve the original epoch schedule and validation data when resuming."""
from contextlib import contextmanager
from pathlib import Path
import re

import torch


def checkpoint_epoch(path):
    match = re.fullmatch(r'epoch-(\d+)\.pt', Path(path).name)
    if match is None:
        raise ValueError('Resume checkpoint must be named epoch-<number>.pt')
    return int(match.group(1))


@contextmanager
def preserve_cpu_rng():
    saved = torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(saved)


def create_epoch_scheduler(optimizer, decay, resume=None):
    # Checkpoints are saved before the end-of-epoch scheduler step. Initializing
    # at the saved epoch advances to exactly the next epoch's learning rate.
    last_epoch = checkpoint_epoch(resume) if resume else -1
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: decay ** epoch,
                                            last_epoch=last_epoch)
