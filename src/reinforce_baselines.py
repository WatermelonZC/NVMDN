import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import copy
from train import rollout, get_inner_model
from problems.nvsp.state import omega
import torch.distributed as dist



class Baseline(object):

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class WarmupBaseline(Baseline):

    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8, ):
        super(Baseline, self).__init__()

        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c):

        if self.alpha == 1:
            return self.baseline.eval(x, c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c)
        v, l = self.baseline.eval(x, c)
        vw, lw = self.warmup_baseline.eval(x, c)
        # Return convex combination of baseline and of loss
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha) * lw

    def epoch_callback(self, model, epoch):
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, epoch)
        next_alpha = min(1.0, (epoch + 1) / float(self.n_epochs))
        if self.alpha != next_alpha:
            self.alpha = next_alpha
            print("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self):
        # Checkpointing within warmup stage makes no sense, only save inner baseline
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        # Checkpointing within warmup stage makes no sense, only load inner baseline
        self.baseline.load_state_dict(state_dict)


class NoBaseline(Baseline):

    def eval(self, x, c):
        return 0, 0  # No baseline, no loss


class ExponentialBaseline(Baseline):

    def __init__(self, beta):
        super(Baseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, x, c):

        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss

    def state_dict(self):
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict):
        self.v = state_dict['v']


class CriticBaseline(Baseline):

    def __init__(self, critic):
        super(Baseline, self).__init__()

        self.critic = critic

    def eval(self, x, c):
        v = self.critic(x)
        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach(), F.mse_loss(v, c.detach())

    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict):
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})


class RolloutBaseline(Baseline):

    def __init__(self, model, problem, opts, epoch=0):
        super(Baseline, self).__init__()

        self.problem = problem
        self.opts = opts

        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        if getattr(self.opts, 'reuse_baseline', False) and hasattr(self, 'model'):
            # Keep parameter storage stable so greedy CUDA graphs remain valid.
            # load_state_dict copies weights and BN statistics without replacing them.
            get_inner_model(self.model).load_state_dict(get_inner_model(model).state_dict())
        else:
            self.model = copy.deepcopy(model)
        # Always generate baseline dataset when updating model to prevent overfitting to the baseline dataset

        if dataset is not None:
            if len(dataset) != self.opts.val_size:
                print("Warning: not using saved baseline dataset since val_size does not match")
                dataset = None
            elif (self.problem.NAME != 'nvsp') and ((dataset[0] if self.problem.NAME == 'tsp' else dataset[0]['loc']).size(0) != self.opts.graph_size):
                print("Warning: not using saved baseline dataset since graph_size does not match")
                dataset = None

        if dataset is None:
            if self.opts.problem == 'nvsp':
                self.dataset = self.problem.make_dataset(
                    size=self.opts.graph_size, num_samples=self.opts.val_size, distribution=self.opts.data_distribution,
                    mom_vehicle_size=self.opts.mom_size, sub_vehicle_size=self.opts.sub_size, designated_driver=self.opts.designated_driver, variable_instance=self.opts.variable_instance,
                    **({'compact': True} if self.problem.NAME == 'nvsp' and getattr(self.opts, 'fast_train', False) else {}))
            else:
                self.dataset = self.problem.make_dataset(
                    size=self.opts.graph_size, num_samples=self.opts.val_size, distribution=self.opts.data_distribution)
        else:
            self.dataset = dataset
        # print("Evaluating baseline model on evaluation dataset")
        self.bl_vals = (rollout(self.model, self.dataset, self.opts))
        if self.bl_vals is not None:
            self.bl_vals = self.bl_vals.cpu().numpy()
            self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def wrap_dataset(self, dataset):
        print(f"[Rank {self.opts.local_rank}] Evaluating baseline on dataset...")

        is_distributed = self.opts.distributed
        is_main = not is_distributed or self.opts.local_rank == 0

        # if is_main:
        #     baseline_vals = rollout(self.model, dataset, self.opts).view(-1, 1)
        # else:
        #     baseline_vals = torch.zeros((len(dataset), 1), dtype=torch.float32, device=self.opts.device)

        # 都进行评估再广播，否则会导致其他GPU等待超时
        label_opts = self.opts
        if getattr(self.opts, 'baseline_eval_amp', False):
            label_opts = copy.copy(self.opts)
            label_opts.eval_amp = True
        baseline_vals = rollout(self.model, dataset, label_opts).view(-1, 1)

        if is_distributed:
            baseline_vals = baseline_vals.to(self.opts.device)
            dist.barrier()
            # print(f"[Rank {self.opts.local_rank}] Reached before broadcast \r\n")
            # 所有进程准备好 buffer 后同步
            dist.broadcast(baseline_vals, src=0)
        return BaselineDataset(dataset, baseline_vals.cpu())

    def unwrap_batch(self, batch):
        return batch['data'], batch['baseline'].view(-1)  # Flatten result to undo wrapping as 2D

    def eval(self, x, c):
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            v, _ = self.model(x)
            if self.opts.problem == 'nvsp' and self.opts.sychonization_cost:
                v = omega * v[0] + (1 - omega) * v[1]

        # There is no loss
        return v, 0

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        Only rank 0 performs evaluation and broadcasts results to other ranks.
        """
        is_distributed = self.opts.distributed
        is_main = not is_distributed or self.opts.local_rank == 0

        if is_main:
            candidate_vals = rollout(model, self.dataset, self.opts).cpu().numpy()
            candidate_mean = candidate_vals.mean()
        else:
            candidate_vals = None
            candidate_mean = 0.0

        # 同步 candidate_mean 到所有 rank
        if is_distributed:
            candidate_mean_tensor = torch.tensor(candidate_mean, dtype=torch.float32, device=self.opts.device)
            dist.broadcast(candidate_mean_tensor, src=0)
            candidate_mean = candidate_mean_tensor.item()

        if is_main:
            print("[rank{}] Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
                self.opts.local_rank, epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))

        if candidate_mean - self.mean < 0:
            if is_main:  # 只有 rank 0 执行 T 检验
                from scipy.stats import ttest_rel
                t, p = ttest_rel(candidate_vals, self.bl_vals)
                p_val = p / 2  # one-sided
                if t >= 0:
                    print("p-value: {} (baseline not updated; t-statistic={})".format(p_val, t))
                    update_baseline = False
                else:
                    print("p-value: {}".format(p_val))
                    update_baseline = p_val < self.opts.bl_alpha
            else:
                update_baseline = False

            # 广播是否更新 baseline
            if is_distributed:
                update_tensor = torch.tensor(int(update_baseline), device=self.opts.device)
                dist.broadcast(update_tensor, src=0)
                update_baseline = bool(update_tensor.item())

            if update_baseline:
                # print('Update baseline')
                self._update_model(model, epoch)

    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        # We make it such that it works whether model was saved as data parallel or not
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(get_inner_model(state_dict['model']).state_dict())
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])


class BaselineDataset(Dataset):

    def __init__(self, dataset=None, baseline=None):
        super(BaselineDataset, self).__init__()

        self.dataset = dataset
        self.baseline = baseline

        assert (len(self.dataset) == len(self.baseline))

    def __getitem__(self, item):
        label_item = item.to(self.baseline.device) if torch.is_tensor(item) and torch.is_tensor(self.baseline) else item
        return {
            'data': self.dataset[item],
            'baseline': self.baseline[label_item]
        }

    def __len__(self):
        return len(self.dataset)
