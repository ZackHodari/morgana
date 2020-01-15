import os

import numpy as np
from scipy.signal import savgol_filter
import torch

from morgana.base_models import BaseSPSS
from morgana.experiment_builder import ExperimentBuilder
from morgana import utils

from tts_data_tools import data_sources
from tts_data_tools import file_io
from tts_data_tools.utils import make_dirs
from tts_data_tools.wav_gen import world_with_reaper_f0


class CopySynth(BaseSPSS):
    def __init__(self):
        r"""Initialises acoustic model parameters and settings."""
        super(CopySynth, self).__init__()

    def train_data_sources(self):
        return {
            'n_frames': data_sources.TextSource('n_frames', sentence_level=True),
            'lf0': data_sources.NumpyBinarySource('lf0', use_deltas=True),
            'vuv': data_sources.NumpyBinarySource('vuv'),
            'mcep': data_sources.NumpyBinarySource('mcep', use_deltas=True),
            'bap': data_sources.NumpyBinarySource('bap', use_deltas=True),
        }

    def predict(self, features):
        return {}

    def loss(self, features, output_features):
        return 0.

    def analysis_for_train_batch(self, features, output_features, out_dir, sample_rate=16000, **kwargs):
        kwargs['sample_rate'] = sample_rate
        super(CopySynth, self).analysis_for_train_batch(features, output_features, out_dir, **kwargs)

        # Synthesise outputs using WORLD.
        synth_dir = os.path.join(out_dir, 'synth')
        make_dirs(synth_dir, features['name'])

        lf0, vuv, mcep, bap = utils.detach_batched_seqs(
            features['lf0'], features['vuv'], features['mcep'], features['bap'],
            seq_len=features['n_frames'])

        for _lf0, _vuv, _mcep, _bap, _file_id in zip(lf0, vuv, mcep, bap, features['name']):
            _vuv = _vuv > 0.5

            _f0 = np.exp(_lf0)
            _f0 = savgol_filter(_f0, 7, 1)

            wav_path = os.path.join(synth_dir, f'{_file_id}.wav')
            wav = world_with_reaper_f0.synthesis(_f0, _vuv, _mcep, _bap, sample_rate)
            file_io.save_wav(wav, wav_path, sample_rate=sample_rate)


def main():
    torch.random.manual_seed(1234567890)
    args = ExperimentBuilder.get_experiment_args()
    args['device'] = 'cpu'
    experiment = ExperimentBuilder(CopySynth, **args)
    experiment.run_experiment()


if __name__ == "__main__":
    main()

