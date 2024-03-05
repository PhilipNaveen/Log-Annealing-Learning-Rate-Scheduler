#imports and dependencies
import math
import torch
import numpy 
from torch.optim.lr_scheduler import LRScheduler
import warnings

class LogAnnealingLR(LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::
\begin{equation}
    \eta_t = \eta^{i}_{min} + \frac{1}{2} \left( \eta^{i}_{max} - \eta^{i}_{min}\right) 
    \left( 1 + \log_{\underset{}{}\underset{\eta_j \in \eta^{i}_{min}}{\max}}(\frac{T_{i}}{T_{cur}} \pi) \right)
\end{equation}

    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, verbose="deprecated"):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.log((self.T_max) * math.pi / self.last_epoch)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - self.eta_min) *
                    (1 - math.log(math.pi / self.T_max)) / 2
                    for base_lr, group in
                    zip(self.base_lrs, self.optimizer.param_groups)]
        return [(1 + math.log(math.pi * self.T_max / self.last_epoch)) /
                (1 + math.log(math.pi * (self.T_max - 1) / self.last_epoch)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.log(math.pi * self.T_max / self.last_epoch)) / 2
                for base_lr in self.base_lrs]
        
        
        
from collections.abc import Iterable
from math import log, cos, pi, floor

from torch.optim.lr_scheduler import _LRScheduler


class CyclicLogDecayLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 init_decay_epochs,
                 min_decay_lr,
                 restart_interval=None,
                 restart_interval_multiplier=None,
                 restart_lr=None,
                 warmup_epochs=None,
                 warmup_start_lr=None,
                 last_epoch=-1,
                 verbose=False):
        """
        Initialize new CyclicCosineDecayLR object.

        :param optimizer: (Optimizer) - Wrapped optimizer.
        :param init_decay_epochs: (int) - Number of initial decay epochs.
        :param min_decay_lr: (float or iterable of floats) - Learning rate at the end of decay.
        :param restart_interval: (int) - Restart interval for fixed cycles.
            Set to None to disable cycles. Default: None.
        :param restart_interval_multiplier: (float) - Multiplication coefficient for geometrically increasing cycles.
            Default: None.
        :param restart_lr: (float or iterable of floats) - Learning rate when cycle restarts.
            If None, optimizer's learning rate will be used. Default: None.
        :param warmup_epochs: (int) - Number of warmup epochs. Set to None to disable warmup. Default: None.
        :param warmup_start_lr: (float or iterable of floats) - Learning rate at the beginning of warmup.
            Must be set if warmup_epochs is not None. Default: None.
        :param last_epoch: (int) - The index of the last epoch. This parameter is used when resuming a training job. Default: -1.
        :param verbose: (bool) - If True, prints a message to stdout for each update. Default: False.
        """

        if not isinstance(init_decay_epochs, int) or init_decay_epochs < 1:
            raise ValueError("init_decay_epochs must be positive integer, got {} instead".format(init_decay_epochs))

        if isinstance(min_decay_lr, Iterable) and len(min_decay_lr) != len(optimizer.param_groups):
            raise ValueError("Expected len(min_decay_lr) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(min_decay_lr), len(optimizer.param_groups)))

        if restart_interval is not None and (not isinstance(restart_interval, int) or restart_interval < 1):
            raise ValueError("restart_interval must be positive integer, got {} instead".format(restart_interval))

        if restart_interval_multiplier is not None and \
                (not isinstance(restart_interval_multiplier, float) or restart_interval_multiplier <= 0):
            raise ValueError("restart_interval_multiplier must be positive float, got {} instead".format(
                restart_interval_multiplier))

        if isinstance(restart_lr, Iterable) and len(restart_lr) != len(optimizer.param_groups):
            raise ValueError("Expected len(restart_lr) to be equal to len(optimizer.param_groups), "
                             "got {} and {} instead".format(len(restart_lr), len(optimizer.param_groups)))

        if warmup_epochs is not None:
            if not isinstance(warmup_epochs, int) or warmup_epochs < 1:
                raise ValueError(
                    "Expected warmup_epochs to be positive integer, got {} instead".format(type(warmup_epochs)))

            if warmup_start_lr is None:
                raise ValueError("warmup_start_lr must be set when warmup_epochs is not None")

            if not (isinstance(warmup_start_lr, float) or isinstance(warmup_start_lr, Iterable)):
                raise ValueError("warmup_start_lr must be either float or iterable of floats, got {} instead".format(
                    warmup_start_lr))

            if isinstance(warmup_start_lr, Iterable) and len(warmup_start_lr) != len(optimizer.param_groups):
                raise ValueError("Expected len(warmup_start_lr) to be equal to len(optimizer.param_groups), "
                                 "got {} and {} instead".format(len(warmup_start_lr), len(optimizer.param_groups)))

        group_num = len(optimizer.param_groups)
        self._warmup_start_lr = [warmup_start_lr] * group_num if isinstance(warmup_start_lr, float) else warmup_start_lr
        self._warmup_epochs = 0 if warmup_epochs is None else warmup_epochs
        self._init_decay_epochs = init_decay_epochs
        self._min_decay_lr = [min_decay_lr] * group_num if isinstance(min_decay_lr, float) else min_decay_lr
        self._restart_lr = [restart_lr] * group_num if isinstance(restart_lr, float) else restart_lr
        self._restart_interval = restart_interval
        self._restart_interval_multiplier = restart_interval_multiplier
        super(CyclicLogDecayLR, self).__init__(optimizer, last_epoch, verbose=verbose)
        


    def get_lr(self):

        if self._warmup_epochs > 0 and self.last_epoch < self._warmup_epochs:
            return self._calc(self.last_epoch,
                              self._warmup_epochs,
                              self._warmup_start_lr,
                              self.base_lrs)

        elif self.last_epoch < self._init_decay_epochs + self._warmup_epochs:
            return self._calc(self.last_epoch - self._warmup_epochs,
                              self._init_decay_epochs,
                              self.base_lrs,
                              self._min_decay_lr)
        else:
            if self._restart_interval is not None:
                if self._restart_interval_multiplier is None:
                    cycle_epoch = (self.last_epoch - self._init_decay_epochs - self._warmup_epochs) % self._restart_interval
                    lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
                    return self._calc(cycle_epoch,
                                      self._restart_interval,
                                      lrs,
                                      self._min_decay_lr)
                else:
                    n = self._get_n(self.last_epoch - self._warmup_epochs - self._init_decay_epochs)
                    sn_prev = self._partial_sum(n)
                    cycle_epoch = self.last_epoch - sn_prev - self._warmup_epochs - self._init_decay_epochs
                    interval = self._restart_interval * self._restart_interval_multiplier ** n
                    lrs = self.base_lrs if self._restart_lr is None else self._restart_lr
                    return self._calc(cycle_epoch,
                                      interval,
                                      lrs,
                                      self._min_decay_lr)
            else:
                return self._min_decay_lr

    def _calc(self, t, T, lrs, min_lrs):
        return [abs(min_lr + (lr - min_lr) * ((1 + log((pi * T / (t + 0.000000001)), 1 / (min(min_lrs)))) / 2))
                for lr, min_lr in zip(lrs, min_lrs)]

    def _get_n(self, epoch):
        _t = 1 - (1 - self._restart_interval_multiplier) * epoch / self._restart_interval
        return floor(log(_t, self._restart_interval_multiplier))

    def _partial_sum(self, n):
        return self._restart_interval * (1 - self._restart_interval_multiplier ** n) / (
                    1 - self._restart_interval_multiplier)
