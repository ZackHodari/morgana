import os

import numpy as np
import torch
import torch.nn as nn

from morgana.base_models import BaseSPSS
from morgana.experiment_builder import ExperimentBuilder
from morgana import viz
from morgana import data
from morgana import metrics
from morgana import utils


class F0Model(BaseSPSS):
    def __init__(self, normalisers=None, dropout_prob=0., input_dim=600+9, output_dim=1*3):
        r"""Initialises acoustic model parameters and settings."""
        super(F0Model, self).__init__(normalisers=normalisers)
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

    @classmethod
    def train_data_sources(cls):
        return {
            'n_frames': data.TextSource('n_frames'),
            'n_phones': data.TextSource('n_phones'),
            'dur': data.TextSource('dur', normalisation='mvn'),
            'lab': data.NumpyBinarySource('lab', normalisation='minmax'),
            'counters': data.NumpyBinarySource('counters', normalisation='minmax'),
            'lf0': data.NumpyBinarySource('lf0', normalisation='mvn', use_deltas=True),
            'vuv': data.NumpyBinarySource('vuv', dtype=np.bool),
        }

    @classmethod
    def valid_data_sources(cls):
        data_sources = cls.train_data_sources()
        data_sources['sp'] = data.NumpyBinarySource('sp')
        data_sources['ap'] = data.NumpyBinarySource('ap')

        return data_sources

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
        inputs = features['normalised_lf0_deltas']
        outputs = output_features['normalised_lf0_deltas']
        seq_len = features['n_frames']

        self.metrics.accumulate(
            self.mode,
            LF0_RMSE_Hz=(features['lf0'], output_features['lf0'], seq_len, features['vuv']))

        return self._loss(inputs, outputs, seq_len)

    def analysis_for_valid_batch(self, output_features, features, names, out_dir, sample_rate=16000, **kwargs):
        super(F0Model, self).analysis_for_valid_batch(output_features, features, names, out_dir, **kwargs)

        # Synthesise outputs using WORLD.
        synth_dir = os.path.join(out_dir, 'synth')
        viz.synthesis.synth_batch_predictions(
            output_features, features, names, out_dir=synth_dir, sample_rate=sample_rate, **kwargs)


def main():
    torch.random.manual_seed(1234567890)
    args = ExperimentBuilder.get_experiment_args()
    experiment = ExperimentBuilder(F0Model, **args)
    experiment.run_experiment()


if __name__ == "__main__":
    main()

