from collections.abc import Mapping, Iterable, Sized
import re

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def listify(object_or_list):
    r"""Converts input to an iterable if it is not already one."""
    if not isinstance(object_or_list, (list, tuple)):
        object_or_list = [object_or_list]
    return object_or_list


def format_float_tensor(tensor):
    r"""Formats the a single value or a 1-dimensional vector as a string."""
    if isinstance(tensor, Sized):
        try:
            feat_dim = len(tensor)
        except TypeError:
            # Length of a zero-dimensional tensor/array is undefined.
            feat_dim = 0
    else:
        feat_dim = 0

    if feat_dim <= 1:
        return tqdm.format_num(tensor)
    elif feat_dim <= 4:
        return '[{}]'.format(', '.join(tqdm.format_num(val) for val in tensor))
    else:
        return '[{first}, {second}, ..., {last}]'.format(
            first=tqdm.format_num(tensor[0]), second=tqdm.format_num(tensor[1]), last=tqdm.format_num(tensor[-1]))


def map_nested(func, data):
    r"""Recursively applies a function on a nested data structure. Base cases: `np.ndarray` :class:`torch.Tensor`."""
    if isinstance(data, (np.ndarray, torch.Tensor)):
        mapped = func(data)

    elif isinstance(data, Mapping):
        mapped = {}
        for k, v in data.items():
            mapped[k] = map_nested(func, v)

    elif isinstance(data, Iterable) and not isinstance(data, str):
        mapped = [map_nested(func, v) for v in data]

    else:
        mapped = func(data)

    return mapped


def infer_device(tensor):
    r"""Gets the device from a :class:`torch.Tensor` instance."""
    if tensor.is_cuda:
        device = 'cuda:{}'.format(tensor.get_device())
    else:
        device = 'cpu'

    return torch.device(device)


def detach_batched_seqs(*sequence_features, seq_len=None, squeeze=True):
    r"""Converts :class:`torch.Tensor` to `np.ndarray`. Moves data to CPU, detaches gradients, and removes padding.

    Parameters
    ----------
    sequence_features : list[torch.Tensor] or torch.Tensor, shape (batch_size, max_seq_len, feat_dim)
        List of batched sequence features to be detached.
    seq_len : np.ndarray or torch.Tensor, shape (batch_size,)
        Sequence length used to remove padding from each batch item.
    squeeze : bool
        If True, try to squeeze 1-dimensional features

    Returns
    -------
    list[list[np.ndarray]] or list[np.ndarray], shape (batch_size, (seq_len, feat_dim))
        Sequence features as `np.ndarray` without paddingWhat.
    """
    if isinstance(seq_len, torch.Tensor):
        seq_len = seq_len.cpu().detach().numpy()

    detached = []
    for sequence_feature_batch in sequence_features:

        # Convert to numpy.
        if isinstance(sequence_feature_batch, torch.Tensor):
            sequence_feature_batch = sequence_feature_batch.cpu().detach().numpy()

        # Remove padding.
        if seq_len is not None and sequence_feature_batch[0].ndim > 1:
            sequence_feature_batch = [sequence_feature[:len_].squeeze() if squeeze else sequence_feature[:len_]
                                      for sequence_feature, len_ in zip(sequence_feature_batch, seq_len)]

        detached.append(sequence_feature_batch)

    if len(detached) == 1:
        return detached[0]
    return detached


def get_epoch_from_checkpoint_path(checkpoint_path):
    r"""Extracts the epoch number from a checkpoint path of the form `.*checkpoints/epoch_(NUM)_.*.pt`"""
    epoch_regex = re.compile(r'.*checkpoints/epoch_(?P<epoch>\d+)(_\w+)?\.\w+')
    match = epoch_regex.match(checkpoint_path)
    if match is None:
        return 0
    else:
        return int(match['epoch'])


def sequence_mask(seq_len, max_len=None, dtype=torch.ByteTensor, device=None):
    r"""Creates a sequence mask with a given type.

    Parameters
    ----------
    seq_len : torch.Tensor, shape (batch_size,)
        Sequence lengths.
    max_len : int, optional
        Maximum sequence length. If None, `max(seq_len)` will be used to infer the `max_len`.
    dtype : type or dtype
        Type for the mask that will be returned.
    device : str or `torch.device`
        Name of the device to place the mask on.

    Returns
    -------
    mask : torch.Tensor, shape (batch_size, max_len, 1)
        Sequence mask.
    """
    if max_len is None:
        max_len = torch.max(seq_len).item()

    if device is None:
        device = infer_device(seq_len)

    range = torch.arange(max_len).type(seq_len.dtype).to(device)

    mask = range[None, :] < seq_len[:, None]

    return mask[:, :, None].type(dtype)


def batched_masked_select(sequence_feature, seq_len):
    r"""Gets the feature vectors for all items in the batch that are within the sequence, according to `seq_len`.

    Performs the same operation as `torch.masked_select`, but on a batch (i.e. returning a 2-d tensor).

    Parameres
    ---------
    sequence_feature : torch.Tensor, shape (batch_size, max_seq_len, feat_dim)
        Sequence feature at some lower frame-rate, this will be upsampled.
    seq_len : np.ndarray or torch.Tensor, shape (batch_size,)
        Sequence lengths used to crop each batch item.

    Returns : torch.Tensor, shape (sum(seq_len), feat_dim)
        Features from `sequence_feature` that are within each batch items sequence length.
    """
    mask = sequence_mask(seq_len, sequence_feature.shape[1], dtype=torch.long)
    mask = mask.squeeze(dim=2)

    idxs = mask.nonzero(as_tuple=True)
    return sequence_feature[idxs]


def both_voiced_mask(*sequence_features, dtype=torch.ByteTensor):
    r"""Calculates whether the sequence features are non-zero at the same time."""
    is_voiced = [~torch.eq(sequence_feature, 0.) for sequence_feature in sequence_features]
    return torch.prod(torch.stack(is_voiced), dim=0).type(dtype)


def upsample_to_repetitions(sequence_feature, repeats):
    r"""Copies sequence items according to some number of repetitions. Functionality is the same as `np.repeat`.

    This is useful for upsampling phone-level linguistic features to frame-level, where `repeats` would be durations.

    Parameters
    ----------
    sequence_feature : torch.Tensor, shape (batch_size, max_seq_len, feat_dim)
        Sequence feature at some lower frame-rate, this will be upsampled.
    repeats : torch.Tensor, shape (batch_size, max_seq_len, 1)
        Number of repetitions of each sequence item.

    Returns
    -------
    upsampled_sequence_feature : torch.Tensor, shape (batch_size, max_repeated_len, feat_dim)
        Sequence feature upsampled using repetitions of individual sequence items.
    """
    device = infer_device(sequence_feature)

    batch_size = sequence_feature.shape[0]
    max_seq_len = sequence_feature.shape[1]
    feat_dim = sequence_feature.shape[2]

    repeated_lens = torch.sum(repeats, dim=1)
    max_repeated_len = torch.max(repeated_lens).item()

    # Remove the trailing single dimension axis if it exists.
    repeats = repeats.reshape((batch_size, -1))

    # Pad the sequence features with an extra frame, then the index array `repeated_idxs` can be created using
    # `np.repeat` using the value -1 for positions where we need to insert the padder.
    padder = torch.zeros((batch_size, 1, feat_dim), dtype=sequence_feature.dtype).to(device)
    sequence_feature_with_padder = torch.cat((sequence_feature, padder), dim=1)

    # The batch indexes are of shape (batch_size, max_repeated_len), with each row containing the index of that row.
    batch_idx = torch.arange(batch_size)[:, None]
    batch_idxs = batch_idx.repeat(1, max_repeated_len)

    # We create the sequence indexes such that any positions that are not modified below will index the padding frame
    repeated_idxs = -1 * np.ones((batch_size, max_repeated_len), dtype=np.int64)

    # For each item in the batch we use `np.repeat` to create our indices, this will vary in length for each item, so
    # we need to ensure proper indexing of `repeated_idxs` is done using `repeated_len`.
    seq_feats_idx = np.arange(max_seq_len)
    for b, (repeat, repeated_len) in enumerate(zip(repeats.cpu(), repeated_lens.cpu())):
        repeated_idxs[b, :repeated_len] = np.repeat(seq_feats_idx, repeat)

    repeated_idxs = torch.tensor(repeated_idxs).to(device)

    # Now we can easily index our PyTorch tensor using the indexes created in NumPy. This will have no interaction with
    # actual values, since it is creating a view to the original tensor, so using NumPy has no effect on backprop.
    upsampled_sequence_feature = sequence_feature_with_padder[batch_idxs, repeated_idxs]

    return upsampled_sequence_feature


def split_to_segments(sequence_feature, segment_lens):
    r"""Splits sequence into shorter segments according to some lengths.

    This is useful for splitting sentence level features into lower-level sequences, such as phone or word.

    Parameters
    ----------
    sequence_feature : torch.Tensor, shape (batch_size, max_seq_len, feat_dim)
        Sequence feature at some lower frame-rate, this will be upsampled.
    segment_lens : torch.Tensor, shape (batch_size, max_num_segments, 1)
        Lengths of segments for each sequence item.

    Returns
    -------
    segmented_sequence_feature : torch.Tensor, shape (batch_size, max_num_segments, max_segment_len, feat_dim)
        Sequence feature split using segment lengths of individual sequence items.
    """
    device = infer_device(sequence_feature)

    batch_size = sequence_feature.shape[0]
    feat_dim = sequence_feature.shape[-1]

    max_num_segments = segment_lens.shape[1]
    max_segment_len = segment_lens.max().item()

    # Ensure `segment_lens` has 2 dimensions.
    segment_lens = segment_lens.reshape((batch_size, -1))

    # Pad the sequence features with an extra frame, then the index array `segment_idxs` can be created using using the
    # value -1 for positions where we need to insert the padder.
    padder = torch.zeros((batch_size, 1, feat_dim), dtype=sequence_feature.dtype).to(device)
    sequence_feature_with_padder = torch.cat((sequence_feature, padder), dim=1)

    # Batch indexes have shape (batch_size, max_num_segments, max_segment_len), first axis contains batch index of item.
    batch_idx = torch.arange(batch_size)[:, None, None]
    batch_idxs = batch_idx.repeat(1, max_num_segments, max_segment_len)

    # Create indexes such that positions that are not modified below will index the padding frame.
    segment_idxs = -1 * np.ones((batch_size, max_num_segments, max_segment_len), dtype=np.int64)

    # Populate the `segment_idxs` tensor with indices corresponding to the length of all segments in each batch item.
    for b, segment_len in enumerate(segment_lens.cpu()):
        seq_idx = 0
        for seg_idx, seg_len in enumerate(segment_len):
            segment_idxs[b, seg_idx, :seg_len] = np.arange(seq_idx, seq_idx + seg_len, dtype=np.int64)
            seq_idx += seg_len

    segment_idxs = torch.tensor(segment_idxs).to(device)

    # Now we can easily index our PyTorch tensor using the indexes created in NumPy. This will have no interaction with
    # actual values, since it is creating a view to the original tensor, so using NumPy has no effect on backprop.
    segmented_sequence_feature = sequence_feature_with_padder[batch_idxs, segment_idxs]

    return segmented_sequence_feature


def get_segment_ends(sequence_feature, segment_lens):
    r"""Gets the feature at the last position for each segment, degined by the lengths `segment_lens`.

    This is useful for clockwork RNN models, given the lengths of segments, we can get the outputs of the current
    time-scale (e.g. frame) needed for the next time-scale (e.g. word).

    Parameters
    ----------
    sequence_feature : torch.Tensor, shape (batch_size, max_seq_len, feat_dim)
        Sequence feature at some lower frame-rate, this will be upsampled.
    segment_lens : torch.Tensor, shape (batch_size, max_num_segments, 1)
        Lengths of segments for each sequence item.

    Returns
    -------
    segment_feature : torch.Tensor, shape (batch_size, max_num_segments, feat_dim)
        Features from `sequence_feature` for the indices corresponding to the ends of each segment.
    """
    device = infer_device(sequence_feature)

    batch_size = sequence_feature.shape[0]
    feat_dim = sequence_feature.shape[-1]

    max_num_segments = segment_lens.shape[1]

    # Ensure `segment_lens` has 2 dimensions.
    segment_lens = segment_lens.reshape((batch_size, -1))

    # Pad the sequence features with an extra frame, then the index array `cumulative_lens` can be created using using
    # the value -1 for positions where we need to insert the padder.
    padder = torch.zeros((batch_size, 1, feat_dim), dtype=sequence_feature.dtype).to(device)
    sequence_feature_with_padder = torch.cat((sequence_feature, padder), dim=1)

    # Batch indexes have shape (batch_size, max_num_segments), rows contain batch index of item.
    batch_idx = torch.arange(batch_size)[:, None]
    batch_idxs = batch_idx.repeat(1, max_num_segments)

    # From the segment lengths calculate the indices at the end of the segments (cumulative sum), masking the padding.
    segment_mask = (segment_lens > 0).type(torch.long)
    segment_idxs = torch.cumsum(segment_lens, dim=1, dtype=torch.long) * segment_mask

    segment_feature = sequence_feature_with_padder[batch_idxs, segment_idxs - 1]

    return segment_feature


class RecurrentCuDNNWrapper(nn.Module):
    r"""Wraps a torch layer with sequence packing. This requires the sequence lengths to sort, pack and unpack.

    Parameters
    ----------
    layer : torch.nn.RNNBase
        PyTorch recurrent layer to be wrapped.
    """
    def __init__(self, layer):
        super(RecurrentCuDNNWrapper, self).__init__()
        self.layer = layer

    def forward(self, inputs, hx=None, seq_len=None):
        # Sort the batch items by sequence length and pack the sequence.
        sorted_idxs = torch.argsort(seq_len, descending=True)
        sorted_inputs = inputs[sorted_idxs, ...]
        sorted_seq_len = seq_len[sorted_idxs]
        packed_inputs = nn.utils.rnn.pack_padded_sequence(sorted_inputs, sorted_seq_len, batch_first=True)

        # Sort the initial hidden state of each batch item by sequence length.
        if hx is not None:
            # Hidden shape is (num_layers * num_directions, batch_size, hidden_size).
            if self.layer.mode == 'LSTM':
                hx = (hx[0][:, sorted_idxs, :], hx[1][:, sorted_idxs, :])
            else:
                hx = hx[:, sorted_idxs, :]

        # Run the recurrent layer.
        packed_outputs, hidden = self.layer(packed_inputs, hx)

        # Unpack and unsort the outputs.
        sorted_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        unsorting_idxs = torch.argsort(sorted_idxs)
        outputs = sorted_outputs[unsorting_idxs, ...]

        # Unsort the final hidden state of each batch item.
        if self.layer.mode == 'LSTM':
            hidden = (hidden[0][:, unsorting_idxs, :], hidden[1][:, unsorting_idxs, :])
        else:
            hidden = hidden[:, unsorting_idxs, :]

        return outputs, hidden


class SequentialWithRecurrent(nn.Sequential):
    r"""Wraps :class:`torch.nn.Sequential` to take custom forward arguments used by :class:`RecurrentCuDNNWrapper`."""
    def __init__(self, *args):
        super(SequentialWithRecurrent, self).__init__(*args)

    def forward(self, input, hx=None, seq_len=None):
        hidden = None
        for module in self._modules.values():

            if isinstance(module, RecurrentCuDNNWrapper):
                input, hidden = module(input, hx, seq_len)

            elif isinstance(module, nn.RNNBase):
                input, hidden = module(input, hx)

            else:
                input = module(input)

        if isinstance(module, (RecurrentCuDNNWrapper, nn.RNNBase)):
            return input, hidden
        else:
            return input


class ExponentialMovingAverage(object):
    """Exponential moving average helper to apply gradient updates to an EMA model.

    Parameters
    ----------
    model : torch.nn.Module
    decay : float
        Decay rate of previous parameter values. Parameter updates are also scaled by `1 - decay`.
    """
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay

        # Use shadow to link to all parameters in the averaged model.
        self.shadow = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data

                # The following is necessary, as `self.model` is a separate EMA model.
                # self.shadow[name] = param.data.clone()

    def _update_param(self, name, x):
        """Performs update on one parameter. `shadow = decay * shadow + (1 - decay) * x`."""
        assert name in self.shadow

        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta

    def update_params(self, other_model):
        """Updates all parameters of `self.model` using a separate model's updated parameters."""
        assert other_model is not self.model

        for name, param in other_model.named_parameters():
            if name in self.shadow:
                self._update_param(name, param.data)

    # The following is not necessary, as `morgana.experiment_builder.ExperimentBuilder` creates a separate EMA model.
    # def clone_average_model(self):
    #     for name, param in self.model.named_parameters():
    #         if name in self.shadow:
    #             param.data = self.shadow[name].clone()
    #
    #     return self.model

