import torch


def KLD_standard_normal(mean, log_variance):
    r"""Calculates KL-divergence of :math:`\mathbb{N}` (`mean`, `log_variance`) with :math:`\mathbb{N}(0, 1)`."""
    kld = -0.5 * torch.sum(1 + log_variance - mean ** 2 - torch.exp(log_variance), dim=-1)
    return torch.mean(kld)

