import os

import numpy as np

import tts_data_tools as tdt

from morgana import utils


def save_batched_seqs(sequence_features, names, out_dir, seq_len=None, feat_names=None):
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

