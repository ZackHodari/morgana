from functools import partial

from torch.optim import lr_scheduler


EPOCH_LR_SCHEDULES = ['constant', 'lambda', 'step', 'multi_step', 'exponential', 'cosine_annealing',
                      'cosine_annealing_warm_restarts']
BATCH_LR_SCHEDULES = ['cyclic', 'noam', 'cyclic_noam']

SUPPORTED = {
    'lambda': lr_scheduler.LambdaLR,
    'step': lr_scheduler.StepLR,
    'multi_step': lr_scheduler.MultiStepLR,
    'exponential': lr_scheduler.ExponentialLR,
    'cosine_annealing': lr_scheduler.CosineAnnealingLR,
    'plateau': lr_scheduler.ReduceLROnPlateau,
    'cyclic': lr_scheduler.CyclicLR,
    'cosine_annealing_warm_restarts': lr_scheduler.CosineAnnealingWarmRestarts,
    ###
    # ADDED BELOW
    ###
    # 'constant': DummyLR,
    # 'noam': NoamLR,
    # 'cyclic_noam': CyclicNoamLR,
}


def init_lr_schedule(lr_name, **kwargs):
    r"""Partially initialises the learning rate schedule with kwargs (optimiser is required for full initialisation)."""
    return partial(SUPPORTED[lr_name], **kwargs)


class DummyLR(lr_scheduler._LRScheduler):
    r"""Used to perform no learning rate changes, i.e. a constant learning rate."""
    def __init__(self, optimizer):
        super(DummyLR, self).__init__(optimizer)

    def get_lr(self):
        return self.base_lrs


SUPPORTED['constant'] = DummyLR


class NoamLR(lr_scheduler._LRScheduler):
    r"""Noam Learning rate schedule.

    Increases the learning rate linearly for the first `warmup_steps` training steps, then decreases it proportional to
    the inverse square root of the step number.

              ^
             / \
            /   `
           /     `
          /         `
         /               `
        /                       `
       /                                   `
      /                                                    `
     /                                                                              `
    /                                                                                                                  `

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimiser instance to modify the learning rate of.
    warmup_steps : int
        The number of steps to linearly increase the learning rate.

    Notes
    -----
    If step <= warmup_steps,
        scale = step / warmup_steps
    If step > warmup_steps,
        scale = (warmup_steps ^ 0.5) / (step ^ 0.5)
    """
    def __init__(self, optimizer, warmup_steps=4000):
        self.warmup_steps = warmup_steps
        super(NoamLR, self).__init__(optimizer)

    def scale(self, step):
        return self.warmup_steps ** 0.5 * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.scale(last_epoch)
        return [base_lr * scale for base_lr in self.base_lrs]


SUPPORTED['noam'] = NoamLR


class CyclicNoamLR(NoamLR):
    r"""Cyclical Noam learning rate schedule.

    Same increase and decrease pattern, but it repeats when the scale gets to cycle_trigger (after cycle_steps batches).

              ^                                                        ^
             / \                                                      / \
            /   `                                                    /   `
           /     `                                                  /     `
          /         `                                              /         `
         /               `                                        /               `
        /                       `                                /                       `
       /                                   `                    /                                   `
      /                                                    `   /                                                    `
     /                                                        /
    /                                                        /

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimiser instance to modify the learning rate of.
    warmup_steps : int
        The number of steps to linearly increase the learning rate.
    cycle_trigger : float
        Scale of learning rate that will trigger cycle to repeat.
    cycle_steps : int, optional
        The number of steps at which the pattern will repeat. If given, overrides cycle_trigger.

    Notes
    -----
    The cycle is triggered based on a given scale during the decreasing stage:
    cycle_trigger = (warmup_steps ^ 0.5) / (cycle_steps ^ 0.5)

    Becomes,
    cycle_steps = (cycle_trigger / warmup_steps ^ 0.5) ^ -2
    """
    def __init__(self, optimizer, warmup_steps=4000, cycle_trigger=0.2, cycle_steps=None):
        self.warmup_steps = warmup_steps

        if cycle_steps is None:
            self.cycle_steps = int((cycle_trigger / self.warmup_steps ** 0.5) ** -2)
        else:
            self.cycle_steps = cycle_steps

        super(CyclicNoamLR, self).__init__(optimizer, warmup_steps=warmup_steps)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch % self.cycle_steps)
        scale = self.scale(last_epoch)
        return [base_lr * scale for base_lr in self.base_lrs]


SUPPORTED['cyclic_noam'] = CyclicNoamLR

