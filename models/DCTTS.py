import logging

import torch
import torch.nn as nn

from morgana.base_models import BaseSPSS
from morgana.experiment_builder import ExperimentBuilder
from morgana.metrics import LF0Distortion
from morgana.viz.synthesis import MLPG
from morgana import data
from morgana import utils

from tts_data_tools import data_sources


logger = logging.getLogger('morgana')


class DCTTS(BaseSPSS):
    def __init__(self, input_dim=600+9, output_dim=1*3, num_heads=1):
        """Initialises VAE parameters and settings."""
        super(DCTTS, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.encoder_layer = utils.SequentialWithRecurrent(
            nn.Linear(self.input_dim, 256),
            nn.Sigmoid(),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(256, 64, batch_first=True)),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(64, 64, batch_first=True)),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(64, 64, batch_first=True)),
            nn.Linear(64, 64),
            nn.Sigmoid(),
        )

        self.attention_mechanism = nn.MultiHeadAttention(embed_dim=64, num_heads=num_heads)

        self.decoder_layer = utils.SequentialWithRecurrent(
            nn.Linear(64, 64),
            nn.Sigmoid(),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(64, 64, batch_first=True)),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(64, 64, batch_first=True)),
            utils.RecurrentCuDNNWrapper(
                nn.GRU(64, 64, batch_first=True)),
            nn.Linear(64, 64),
            nn.Sigmoid(),
            nn.Linear(64, self.output_dim),
        )

        self.metrics.add_metrics('all',
                                 LF0_RMSE_Hz=LF0Distortion())

    def normaliser_sources(self):
        return {
            'dur': data.MeanVarianceNormaliser('dur'),
            'lf0': data.MeanVarianceNormaliser('lf0', use_deltas=True),
        }

    def train_data_sources(self):
        return {
            'n_frames': data_sources.TextSource('n_frames', sentence_level=True),
            'n_phones': data_sources.TextSource('n_phones', sentence_level=True),
            'dur': data_sources.TextSource('dur'),
            'phone': data_sources.StringSource('phone'),
            'lf0': data_sources.NumpyBinarySource('lf0', use_deltas=True),
            'vuv': data_sources.NumpyBinarySource('vuv'),
        }

    def valid_data_sources(self):
        sources = self.train_data_sources()
        sources['sp'] = data_sources.NumpyBinarySource('sp')
        sources['ap'] = data_sources.NumpyBinarySource('ap')

        return sources

    def encode(self, features):
        phones_one_hot = features['phone']
        n_frames = features['n_frames']
        embeddings = self.encoder_layer(phones_one_hot, seq_len=n_frames)

        return embeddings

    def decode(self, embeddings, features):
        # Teacher forcing.
        targets = features['normalised_lf0_deltas']
        n_phones = features['n_phones']
        decoder_inputs = self.attention_mechanism(targets, embeddings, embeddings, key_padding_mask=n_phones)

        # Run the decoder.
        n_frames = features['n_frames']
        pred_norm_lf0_deltas = self.decoder_layer(decoder_inputs, seq_len=n_frames)

        # Prepare the outputs.
        pred_lf0_deltas = self.normalisers['lf0'].denormalise(pred_norm_lf0_deltas, deltas=True)

        # MLPG to select the most probable trajectory given the delta and delta-delta features.
        pred_lf0 = MLPG(means=pred_lf0_deltas,
                        variances=self.normalisers['lf0'].delta_params['std_dev'] ** 2)

        outputs = {
            'normalised_lf0_deltas': pred_norm_lf0_deltas,
            'lf0_deltas': pred_lf0_deltas,
            'lf0': pred_lf0
        }

        return outputs

    def predict(self, features):
        embeddings = self.encode(features)
        return self.decode(embeddings, features)

    def loss(self, input_features, output_features):
        inputs = input_features['normalised_lf0_deltas']
        outputs = output_features['normalised_lf0_deltas']
        seq_len = input_features['n_frames']

        latent = output_features['latent']
        mean = output_features['mean']
        log_variance = output_features['log_variance']

        self.metrics.accumulate(
            self.mode,
            LF0_RMSE_Hz=(input_features['lf0'], output_features['lf0'], input_features['vuv'], seq_len))

        return self._loss(inputs, outputs, latent, mean, log_variance, seq_len)


def main():
    torch.random.manual_seed(1234567890)
    args = ExperimentBuilder.get_experiment_args()
    experiment = ExperimentBuilder(DCTTS, **args)
    experiment.run_experiment()


if __name__ == "__main__":
    main()

