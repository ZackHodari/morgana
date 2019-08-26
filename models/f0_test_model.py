import os

import numpy as np
import pyworld
from scipy.signal import savgol_filter
import torch
import torch.nn as nn

from morgana.base_models import BaseSPSS
from morgana.experiment_builder import ExperimentBuilder
from morgana import viz
from morgana import metrics
from morgana import utils

from tts_data_tools import data_sources
from tts_data_tools import file_io


class F0Model(BaseSPSS):
    def __init__(self, dropout_prob=0., input_dim=600+9, output_dim=1*3):
        r"""Initialises acoustic model parameters and settings."""
        super(F0Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.recurrent_layers = utils.SequentialWithRecurrent(
            nn.Linear(self.input_dim, 256),
            nn.Sigmoid(),
            nn.Dropout(p=dropout_prob),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(256, 64, batch_first=True)),
            nn.Dropout(p=dropout_prob),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(64, 64, batch_first=True)),
            nn.Dropout(p=dropout_prob),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(64, 64, batch_first=True)),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, self.output_dim),
        )

        self.metrics.add_metrics('all',
                                 LF0_RMSE_Hz=metrics.LF0Distortion())

    def train_data_sources(self):
        return {
            'n_frames': data_sources.TextSource('n_frames'),
            'n_phones': data_sources.TextSource('n_phones'),
            'dur': data_sources.TextSource('dur', normalisation='mvn'),
            'lab': data_sources.NumpyBinarySource('lab', normalisation='minmax'),
            'counters': data_sources.NumpyBinarySource('counters', normalisation='minmax'),
            'lf0': data_sources.NumpyBinarySource('lf0', normalisation='mvn', use_deltas=True),
            'vuv': data_sources.NumpyBinarySource('vuv'),
        }

    def valid_data_sources(self):
        sources = self.train_data_sources()
        sources['sp'] = data_sources.NumpyBinarySource('sp')
        sources['ap'] = data_sources.NumpyBinarySource('ap')

        return sources

    def predict(self, features):
        # Prepare inputs.
        norm_lab_at_frame_rate = utils.upsample_to_repetitions(features['normalised_lab'], features['dur'])
        model_inputs = torch.cat((norm_lab_at_frame_rate, features['normalised_counters']), dim=-1)
        n_frames = features['n_frames']

        # Run the encoder.
        pred_norm_lf0_deltas = self.recurrent_layers(model_inputs, seq_len=n_frames)

        # Prepare the outputs.
        pred_lf0_deltas = self.normalisers['lf0'].denormalise(pred_norm_lf0_deltas, deltas=True)

        # MLPG to select the most probable trajectory given the delta and delta-delta features.
        global_variance = self.normalisers['lf0'].delta_params['std_dev'] ** 2
        pred_lf0 = viz.synthesis.MLPG(pred_lf0_deltas, global_variance, padding_size=100, seq_len=n_frames)

        outputs = {
            'lf0_deltas': pred_lf0_deltas,
            'lf0': pred_lf0
        }

        return outputs

    def loss(self, features, output_features):
        inputs = features['lf0_deltas']
        outputs = output_features['lf0_deltas']
        seq_len = features['n_frames']

        self.metrics.accumulate(
            self.mode,
            LF0_RMSE_Hz=(features['lf0'], output_features['lf0'], seq_len, features['vuv']))

        return self._loss(inputs, outputs, seq_len)

    def analysis_for_valid_batch(self, features, output_features, out_dir, sample_rate=16000, **kwargs):
        super(F0Model, self).analysis_for_valid_batch(features, output_features, out_dir, **kwargs)

        # Synthesise outputs using WORLD.
        synth_dir = os.path.join(out_dir, 'synth')
        os.makedirs(synth_dir, exist_ok=True)

        lf0 = output_features['lf0'].cpu().detach().numpy()

        vuv = features['vuv'].cpu().detach().numpy()
        sp = features['sp'].cpu().detach().numpy()
        ap = features['ap'].cpu().detach().numpy()

        n_frames = features['n_frames'].cpu().detach().numpy()
        for i, (n_frame, name) in enumerate(zip(n_frames, features['name'])):

            f0_i = np.exp(lf0[i, :n_frame, 0])
            f0_i = savgol_filter(f0_i, 7, 1)
            f0_i = f0_i * vuv[i, :n_frame, 0]

            f0_i = f0_i.astype(np.float64)
            sp_i = sp[i, :n_frame].astype(np.float64)
            ap_i = ap[i, :n_frame].astype(np.float64)

            wav_path = os.path.join(synth_dir, '{}.wav'.format(name))
            wav = pyworld.synthesize(f0_i, sp_i, ap_i, sample_rate)
            file_io.save_wav(wav, wav_path, sample_rate=sample_rate)


def main():
    torch.random.manual_seed(1234567890)
    args = ExperimentBuilder.get_experiment_args()
    experiment = ExperimentBuilder(F0Model, **args)
    experiment.run_experiment()


if __name__ == "__main__":
    main()

