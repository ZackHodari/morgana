from functools import partial

from torch.optim.lr_scheduler import _LRScheduler, LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, \
    ReduceLROnPlateau


EPOCH_LR_SCHEDULES = ['constant', 'lambda', 'step', 'multi_step', 'exponential', 'cosine_annealing']
BATCH_LR_SCHEDULES = ['noam']


def init_lr_schedule(lr_name, **kwargs):
    r"""Partially initialises the learning rate schedule with kwargs (optimiser is required for full initialisation)."""
    supported = {
        'constant': DummyLR,
        'lambda': LambdaLR,
        'step': StepLR,
        'multi_step': MultiStepLR,
        'exponential': ExponentialLR,
        'cosine_annealing': CosineAnnealingLR,
        'plateau': ReduceLROnPlateau,
        'noam': NoamLR,
    }

    lr_schedule = supported[lr_name]
    return partial(lr_schedule, **kwargs)


class DummyLR(_LRScheduler):
    r"""Used to perform no learning rate changes, i.e. a constant learning rate."""
    def __init__(self, optimizer):
        super(DummyLR, self).__init__(optimizer)

    def get_lr(self):
        return self.base_lrs


class NoamLR(_LRScheduler):
    r"""Noam Learning rate schedule.

    Increases the learning rate linearly for the first `warmup_steps` training steps, then decreases it proportional to
    the inverse square root of the step number.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimiser instance to modify the learning rate of.
    warmup_steps : int
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_steps=4000):
        self.warmup_steps = warmup_steps
        super(NoamLR, self).__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]

