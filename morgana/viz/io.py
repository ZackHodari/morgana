import os

import numpy as np

import tts_data_tools as tdt

from morgana import utils


def save_batched_seqs(sequence_features, names, out_dir, seq_len=None, feat_names=None):
    """Saves multiple sequence features for multiple sentences, handling sequence length clipping and device detachment.

    Parameters
    ----------
    sequence_features : dict[str, torch.Tensor], list[torch.Tensor], or torch.Tensor
        shape (batch_size, max_seq_len, feat_dim)
        Batched sequence features to be saved.
        If it is a dict, the keys are used as the subdirectory names and `feat_names` can be used to select a subset.
        If it is a list (or singleton), `feat_names` must be provided in order to determine the subdirectory names.
    names : list[str], shape (batch_size, )
        Utterance names to use to save each item in the batch
    out_dir : str
        Path of directory under which to save each feature type.
    seq_len : np.ndarray or torch.Tensor, shape (batch_size,)
        Sequence length used to remove padding from each batch item.
    feat_names : list[str]
        Names of features to be saved, these determine the subdirectory names for saving under `out_dir`.
        If `sequence_features` is a dict, this can be used to select a subset of the features for saving.

    Notes
    -----
    Each feature for each sentence is saved at, {out_dir}/{feat_name}/{name}.npy
    """
    pred_dir = os.path.join(out_dir, 'feats')
    os.makedirs(pred_dir, exist_ok=True)

    if isinstance(sequence_features, dict):
        if feat_names is None:
            feat_names = sequence_features.keys()

        sequence_features = [sequence_features[feat_name] for feat_name in feat_names]

    else:
        if feat_names is None:
            raise ValueError('If sequences features is not a dictionary, then feat_names must be provided.')

    sequence_features = utils.detach_batched_seqs(*sequence_features, seq_len=seq_len)
    sequence_features = utils.listify(sequence_features)

    for feat_name, values in zip(feat_names, sequence_features):

        if isinstance(values[0], np.ndarray):
            tdt.file_io.save_dir(tdt.file_io.save_bin,
                                 path=os.path.join(pred_dir, feat_name),
                                 data=values,
                                 file_ids=names)

