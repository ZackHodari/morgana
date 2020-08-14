import functools

import torch
import torch.nn.functional as F

from morgana import utils


def sequence_loss(loss_fn):
    r"""Sequence loss wrapper, adds the optional `seq_len` key-word argument which masks padded elements.

    The loss value between two frames of the target and prediction is given by :func:`~loss_fn`.

    Parameters
    ----------
    target : torch.Tensor, shape (batch_size, seq_len, feat_dim)
        Ground truth tensor.
    prediction : torch.Tensor, shape (batch_size, seq_len, feat_dim)
        Prediction tensor.
    seq_len : None or torch.Tensor, shape (batch_size,)
        Sequence lengths.

    Returns
    -------
    float
        Masked loss.
    """

    @functools.wraps(loss_fn)
    def wrapped_loss(predictions, targets, seq_len=None):
        feature_loss = loss_fn(predictions, targets)

        if seq_len is None:
            max_num_frames = feature_loss.shape[1]
            feature_loss = torch.sum(feature_loss, dim=1) / max_num_frames
        else:
            mask = utils.sequence_mask(seq_len, max_len=feature_loss.shape[1], dtype=feature_loss.dtype)
            num_valid_frames = torch.sum(mask, dim=1)
            feature_loss = torch.sum(feature_loss * mask, dim=1) / num_valid_frames

        # Average across all batch items and all feature dimensions.
        feature_loss = torch.mean(feature_loss)

        return feature_loss

    return wrapped_loss


@sequence_loss
def mse(predictions, targets):
    return F.mse_loss(predictions, targets, reduction='none')


@sequence_loss
def bce(predictions, targets):
    return F.binary_cross_entropy(predictions, targets, reduction='none')


@sequence_loss
def ce(predictions, targets):
    return F.cross_entropy(predictions.transpose(1, 2), targets, reduction='none').unsqueeze(dim=-1)


def KLD_standard_normal(mean, log_variance):
    r"""Calculates KL-divergence of :math:`\mathbb{N}` (`mean`, `log_variance`) with :math:`\mathbb{N}(0, 1)`."""
    kld = -0.5 * torch.sum(1 + log_variance - mean ** 2 - torch.exp(log_variance), dim=-1)
    return torch.mean(kld)

