import numpy as np
import torch
import torch.nn as nn

from morgana.base_models import BaseAcousticModel
from morgana.experiment_builder import ExperimentBuilder
from morgana.viz.synthesis import MLPG
from morgana import data
from morgana import metrics
from morgana import utils


class LSTMAcousticModel(BaseAcousticModel):
    def __init__(self, normalisers=None, dropout_prob=0.):
        """Initialises acoustic model parameters and settings."""
        super(LSTMAcousticModel, self).__init__(normalisers=normalisers)

        self.recurrent_layers = utils.SequentialWithRecurrent(
            nn.Linear(self.conditioning_dim, 512),
            nn.Sigmoid(),
            nn.Dropout(p=dropout_prob),
            utils.RecurrentCuDNNWrapper(
                nn.LSTM(512, 256, dropout=dropout_prob, batch_first=True)),
            utils.RecurrentCuDNNWrapper(
                nn.LSTM(256, 128, dropout=dropout_prob, batch_first=True)),
            utils.RecurrentCuDNNWrapper(
                nn.LSTM(128, 64, dropout=dropout_prob, batch_first=True)),
            nn.Linear(64, 16),
            nn.Sigmoid(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(16, self.output_dim),
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
            'mgc': data.NumpyBinarySource('mgc', normalisation='mvn', use_deltas=True),
            'bap': data.NumpyBinarySource('bap', normalisation='mvn', use_deltas=True),
        }

    @classmethod
    def valid_data_sources(cls):
        data_sources = cls.train_data_sources()
        data_sources['sp'] = data.NumpyBinarySource('sp')
        data_sources['ap'] = data.NumpyBinarySource('ap')

        return data_sources

    @property
    def output_dim(self):
        return 1 * 3

    @property
    def conditioning_dim(self):
        return 600 + 9

    def predict(self, features):
        # Prepare inputs.
        norm_lab = features['normalised_lab']
        dur = features['dur']
        norm_lab_at_frame_rate = utils.upsample_to_repetitions(norm_lab, dur)

        norm_counters = features['normalised_counters']
        model_inputs = torch.cat((norm_lab_at_frame_rate, norm_counters), dim=-1)

        # Run the encoder.
        n_frames = features['n_frames']
        pred_norm_lf0_deltas, _ = self.recurrent_layers(model_inputs, seq_len=n_frames)

        # Prepare the outputs.
        pred_lf0_deltas = self.normalisers['lf0'].denormalise(pred_norm_lf0_deltas, deltas=True)

        # MLPG to select the most probable trajectory given the delta and delta-delta features.
        device = pred_lf0_deltas.device
        pred_lf0 = MLPG(pred_lf0_deltas.detach().cpu().numpy(),
                        self.normalisers['lf0'].delta_params['std_dev'] ** 2,
                        padding_size=100, seq_len=n_frames)
        pred_lf0 = torch.tensor(pred_lf0).type(torch.float).to(device)

        outputs = {
            'normalised_lf0_deltas': pred_norm_lf0_deltas,
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
            LF0_RMSE_Hz=(features['lf0'], output_features['lf0'], features['vuv'], seq_len))

        return self._loss(inputs, outputs, seq_len)


def main():
    torch.random.manual_seed(1234567890)
    args = ExperimentBuilder.get_experiment_args()
    experiment = ExperimentBuilder(LSTMAcousticModel, **args)
    experiment.run_experiment()


if __name__ == "__main__":
    main()

