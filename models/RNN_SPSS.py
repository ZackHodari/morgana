import numpy as np
import os
from scipy.signal import savgol_filter
import torch
import torch.nn as nn

from morgana.base_models import BaseSPSS
from morgana.experiment_builder import ExperimentBuilder
from morgana.viz.synthesis import MLPG
from morgana import data
from morgana import losses
from morgana import metrics
from morgana import utils

from tts_data_tools import data_sources
from tts_data_tools import file_io
from tts_data_tools.wav_gen import world_with_reaper_f0


class LSTMAcousticModel(BaseSPSS):
    def __init__(self, input_dim=600 + 9, output_dims=None, dropout_prob=0., num_layers=8):
        """Initialises acoustic model parameters and settings."""
        if output_dims is None:
            output_dims = {'lf0': 1 * 3, 'vuv': 1, 'mcep': 60 * 3, 'bap': 5 * 3}

        super(LSTMAcousticModel, self).__init__()
        self.input_dim = input_dim
        self.output_dims = output_dims
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers

        self.layers = utils.SequentialWithRecurrent(
            nn.Linear(self.input_dim, 512),
            nn.Sigmoid(),
            nn.Dropout(p=self.dropout_prob),
            *[utils.RecurrentCuDNNWrapper(nn.LSTM(512, 512, dropout=self.dropout_prob, batch_first=True))
              for _ in range(self.num_layers)],
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(256, sum(self.output_dims.values())),
        )

        self.metrics.add_metrics('all',
                                 LF0_RMSE_Hz=metrics.LF0Distortion(),
                                 VUV_accuracy=metrics.Mean(),
                                 MCEP_distortion=metrics.MelCepDistortion(),
                                 BAP_distortion=metrics.Distortion())

    def normaliser_sources(self):
        return {
            'dur': data.MeanVarianceNormaliser('dur'),
            'lab': data.MinMaxNormaliser('lab'),
            'counters': data.MinMaxNormaliser('counters'),
            'lf0': data.MeanVarianceNormaliser('lf0', use_deltas=True),
            'mcep': data.MeanVarianceNormaliser('mcep', use_deltas=True),
            'bap': data.MeanVarianceNormaliser('bap', use_deltas=True),
        }

    def train_data_sources(self):
        return {
            'n_frames': data_sources.TextSource('n_frames', sentence_level=True),
            'dur': data_sources.TextSource('dur'),
            'lab': data_sources.NumpyBinarySource('lab'),
            'counters': data_sources.NumpyBinarySource('counters'),
            'lf0': data_sources.NumpyBinarySource('lf0', use_deltas=True),
            'vuv': data_sources.NumpyBinarySource('vuv'),
            'mcep': data_sources.NumpyBinarySource('mcep', use_deltas=True),
            'bap': data_sources.NumpyBinarySource('bap', use_deltas=True),
        }

    def predict(self, features):
        # Prepare inputs.
        norm_lab = features['normalised_lab']
        dur = features['dur']
        norm_lab_at_frame_rate = utils.upsample_to_repetitions(norm_lab, dur)

        norm_counters = features['normalised_counters']
        model_inputs = torch.cat((norm_lab_at_frame_rate, norm_counters), dim=-1)

        # Run the model.
        n_frames = features['n_frames']
        pred_norm_deltas = self.layers(model_inputs, seq_len=n_frames)

        # Prepare the outputs.
        output_dims = [self.output_dims[n] for n in ['lf0', 'vuv', 'mcep', 'bap']]
        pred_norm_lf0_deltas, pred_vuv, pred_norm_mcep_deltas, pred_norm_bap_deltas = \
            torch.split(pred_norm_deltas, output_dims, dim=-1)

        pred_lf0 = self._prepare_output('lf0', pred_norm_lf0_deltas)
        pred_mcep = self._prepare_output('mcep', pred_norm_mcep_deltas)
        pred_bap = self._prepare_output('bap', pred_norm_bap_deltas)

        pred_vuv = torch.sigmoid(pred_vuv)

        outputs = {
            'normalised_lf0_deltas': pred_norm_lf0_deltas,
            'normalised_mcep_deltas': pred_norm_mcep_deltas,
            'normalised_bap_deltas': pred_norm_bap_deltas,
            'lf0': pred_lf0,
            'vuv': pred_vuv,
            'mcep': pred_mcep,
            'bap': pred_bap,
        }

        return outputs

    def _prepare_output(self, name, pred_norm_deltas, seq_len=None):
        pred_deltas = self.normalisers[name].denormalise(pred_norm_deltas, deltas=True)

        pred_deltas = pred_deltas.detach().cpu().numpy()

        pred = MLPG(means=pred_deltas,
                    variances=self.normalisers[name].delta_params['std_dev'] ** 2,
                    padding_size=100, seq_len=seq_len)
        pred = torch.tensor(pred).type(pred_norm_deltas.dtype).to(pred_norm_deltas.device)

        return pred

    def loss(self, features, output_features):
        n_frames = features['n_frames']
        vuv = output_features['vuv'] > 0.5

        self.metrics.accumulate(
            self.mode,
            LF0_RMSE_Hz=(features['lf0'], output_features['lf0'], vuv, n_frames),
            VUV_accuracy=((features['vuv'] == vuv).type(torch.float), n_frames),
            MCEP_distortion=(features['mcep'], output_features['mcep'], n_frames),
            BAP_distortion=(features['bap'], output_features['bap'], n_frames))

        loss = 0.

        loss += losses.mse(output_features['normalised_lf0_deltas'], features['normalised_lf0_deltas'], n_frames)
        loss += losses.mse(output_features['normalised_mcep_deltas'], features['normalised_mcep_deltas'], n_frames)
        loss += losses.mse(output_features['normalised_bap_deltas'], features['normalised_bap_deltas'], n_frames)

        loss += losses.bce(output_features['vuv'].type(torch.float), features['vuv'].type(torch.float), n_frames)

        return loss / 4.

    def analysis_for_valid_batch(self, features, output_features, out_dir, sample_rate=16000, **kwargs):
        kwargs['sample_rate'] = sample_rate
        super(LSTMAcousticModel, self).analysis_for_valid_batch(features, output_features, out_dir, **kwargs)

        # Synthesise outputs using WORLD.
        synth_dir = os.path.join(out_dir, 'synth')
        os.makedirs(synth_dir, exist_ok=True)

        lf0, vuv, mcep, bap = utils.detach_batched_seqs(
            output_features['lf0'], output_features['vuv'], output_features['mcep'], output_features['bap'],
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
    experiment = ExperimentBuilder(LSTMAcousticModel, **args)
    experiment.run_experiment()


if __name__ == "__main__":
    main()

