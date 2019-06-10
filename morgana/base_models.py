import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from morgana import losses
from morgana import metrics
from morgana import utils

import tts_data_tools as tdt


class BaseModel(nn.Module):
    r"""Creates an abstract model class with utility functions.

    Any additional kwargs specified in `__init__` should be passed to the command line argument `model_kwargs`.

    Attributes
    ----------
    normalisers : None or dict[str, morgana.data._FeatureNormaliser]
        Normalisers specified within the :class:`morgana.data._DataSource` in `self.train_data_sources`.
    mode : {'', 'train', 'valid', 'test'}
        Stage of training, set in `morgana.experiment_builder.ExperimentBuilder.*_epoch`, for use with `self.metrics`.
    metrics : morgana.metrics.Handler
        Handler for tracking metrics in an online fashion (over multiple batches).
    step : int
        Step in training, calculated using epoch number, batch number, and number of batches per epoch. This is updated
        automatically by :code:`morgana.experiment_builder.ExperimentBuilder`. Useful for logging to `self.tensorboard`.
    tensorboard : tensorboardX.SummaryWriter
    """
    def __init__(self):
        super(BaseModel, self).__init__()

        self.normalisers = {}
        self.mode = ''
        self.metrics = metrics.Handler(loss=metrics.Mean())
        self.step = 0
        self.tensorboard = None

    def train_data_sources(self):
        r"""Specifies the data that will be loaded and used in training.

        Only specifies what data will be loaded, but not where from.

        Returns
        -------
        features
            The data sources used by :class:`morgana.experiment_builder.ExperimentBuilder` for the training data, can be
            any data structure containing :class:`morgana.data._DataSource` instances.
        """
        raise NotImplementedError

    def valid_data_sources(self):
        r"""Specifies the data that will be loaded and used in validation.

        Only specifies what data will be loaded, but not where from.

        Returns
        -------
        features
            The data sources used by :class:`morgana.experiment_builder.ExperimentBuilder` for the validation data, can
            be any data structure containing :class:`morgana.data._DataSource` instances.
        """
        return self.train_data_sources()

    def test_data_sources(self):
        r"""Specifies the data that will be loaded and used in testing.

        Only specifies what data will be loaded, but not where from.

        Returns
        -------
        features
            The data sources used by :class:`morgana.experiment_builder.ExperimentBuilder` for the testing data, can be
            any data structure containing :class:`morgana.data._DataSource` instances.
        """
        return self.valid_data_sources()

    def forward(self, features):
        r"""Defines the computation graph, including calculation of loss.

        Parameters
        ----------
        features : dict[str, torch.Tensor]
            The ground truth features produced by `self.*_data_sources`.

        Returns
        -------
        loss : float
            Loss of the model, as defined by `self.loss`.
        output_features
            Predictions made by the model, can be any data structure containing :class:`torch.Tensor` instances.
        """
        raise NotImplementedError("Forward computation must be implemented in a subclass.")

    def predict(self, features):
        r"""Defines the computation graph.

        Parameters
        ----------
        features : dict[str, torch.Tensor]
            The ground truth features produced by `self.*_data_sources`.

        Returns
        -------
        output_features
            Predictions made by the model, can be any data structure containing :class:`torch.Tensor` instances.
        """
        raise NotImplementedError("Prediction must be implemented in a subclass.")

    def loss(self, features, output_features):
        r"""Defines which predictions should be scored against which ground truth features.

        Typically this method should use :func:`~_loss` to calculate the sequence loss for the target-prediction pairs.

        Parameters
        ----------
        features : dict[str, torch.Tensor]
            The ground truth features produced by `self.*_data_sources`.
        output_features : torch.Tensor or list[torch.Tensor] or dict[str, torch.Tensor]
            Predictions output by user-defined :func:`~predict`.

        Returns
        -------
        float
            Overall loss between (user-defined) pairs of values in `features` and `output_features`.
        """
        raise NotImplementedError("Loss must be implemented in a subclass.")

    def _loss(self, targets, predictions, seq_lens=None, loss_weights=None):
        r"""Defines the sequence loss for multiple target-prediction pairs.

        If `targets` and `predictions` are iterables they must be in the same order, i.e. when zipped corresponding
        elements will be used as a target-prediction pair for calculating the loss.

        The loss value between two frames of the target and prediction is given by :func:`~loss_fn`. Currently
        this must be the same for all target-prediction pairs.

        .. todo::
            Add support for multiple `self.loss_fn`.

        Parameters
        ----------
        targets : list[torch.Tensor] or torch.Tensor, shape (batch_size, seq_len, feat_dim)
            Ground truth tensor(s).
        predictions : list[torch.Tensor] or torch.Tensor, shape (batch_size, seq_len, feat_dim)
            Prediction tensor(s).
        seq_lens : None or list[torch.Tensor] or torch.Tensor, shape (batch_size,)
            Sequence length features. If one tensor is given it will be used for all target-prediction pairs, otherwise
            the length of the list given must match the length of `targets` and `predictions`.
        loss_weights : None or list[float], shape (num_pairs)
            The weight for each target-prediction pair's loss. If `None` then returns the average of all pair's losses.

        Returns
        -------
        float
            Overall (average or weight) loss.

        Raises
        ------
        ValueError
            If `targets`, `predictions`, `seq_len`, or `loss_weights` are lists with non-matching lengths.
        """
        targets = utils.listify(targets)
        predictions = utils.listify(predictions)

        n_feature_streams = len(targets)

        # Ensure there is a sequence length tensor associated with every target/prediction.
        if seq_lens is None:
            seq_lens = [None for _ in range(n_feature_streams)]
        elif not isinstance(seq_lens, (list, tuple)):
            seq_lens = [seq_lens for _ in range(n_feature_streams)]

        # If no cost weights are provided, set all to 1.
        if loss_weights is None:
            loss_weights = [1. for _ in range(n_feature_streams)]

        # Check the number of targets and predictions provided to calculate the loss for.
        if len(predictions) != len(targets) or len(seq_lens) != len(targets) or len(loss_weights) != len(targets):
            raise ValueError("targets, predictions, seq_len, and loss_weights in `_loss` must be the same length.")

        # Calculate the average of the loss for reconstructions.
        loss = 0.
        for i, (feature, prediction, seq_len, weight) in enumerate(zip(targets, predictions, seq_lens, loss_weights)):

            feature_loss = self.loss_fn(feature, prediction)

            if seq_len is None:
                max_num_frames = feature_loss.shape[1]
                feature_loss = torch.sum(feature_loss, dim=1) / max_num_frames
            else:
                mask = utils.sequence_mask(seq_len, max_len=feature_loss.shape[1], dtype=feature_loss.dtype)
                num_valid_frames = torch.sum(mask, dim=1)
                feature_loss = torch.sum(feature_loss * mask, dim=1) / num_valid_frames

            # Average across all batch items and all feature dimensions.
            feature_loss = torch.mean(feature_loss)

            loss += feature_loss * weight
        loss /= n_feature_streams

        self.metrics.accumulate(
            self.mode,
            loss=loss)

        return loss

    def loss_fn(self, target, prediction):
        r"""Defines the frame-wise loss calculation between ground truth and predictions.

        Parameters
        ----------
        target : torch.Tensor, shape (batch_size, seq_len, feat_dim)
            Ground truth feature.
        prediction : torch.Tensor, shape (batch_size, seq_len, feat_dim)
            Predicted feature.

        Returns
        -------
        torch.Tensor, shape (batch_size, seq_len, feat_dim)
            Loss between `feature` and `prediction`.
        """
        raise NotImplementedError("Loss function must be implemented in a subclass.")

    def save_parameters(self, experiment_dir, epoch):
        r"""Saves the model's `state_dict` to a `.pt` file.

        Parameters
        ----------
        experiment_dir : str
            The experiment directory, within which the `checkpoints` directory will be created.
        epoch : int
            The epoch number, used to create the checkpoint file name, `epoch_{}.pt`
        """
        train_epoch_save_path = os.path.join(experiment_dir, 'checkpoints', 'epoch_{}.pt'.format(epoch))
        os.makedirs(os.path.dirname(train_epoch_save_path), exist_ok=True)
        torch.save(self.state_dict(), train_epoch_save_path)

    def load_parameters(self, checkpoint_path, strict=True, device=None):
        r"""Loads a `state_dict` from a `.pt` file.

        Parameters
        ----------
        checkpoint_path : str
            The file path of the `.pt` file containing the `state_dict` to be loaded
        strict : bool
            Whether to strictly enforce that the keys in the loaded `state_dict` match this model's structure.
        device : str or `torch.device` or dict or callable
            Specifies how to remap storage locations, passed to :func:`torch.load`.
        """
        state_dict = torch.load(checkpoint_path, map_location=device)
        super(BaseModel, self).load_state_dict(state_dict, strict=strict)

    def analysis_for_train_batch(self, features, output_features, names, out_dir, **kwargs):
        r"""Hook used by :class:`morgana.experiment_builder.ExperimentBuilder` after training batches for some epochs.

        Can be used to save output or generate visualisations.

        Parameters
        ----------
        features : dict[str, torch.Tensor]
            The ground truth features produced by `self.*_data_sources`.
        output_features : torch.Tensor or list[torch.Tensor] or dict[str, torch.Tensor]
            Predictions output by user-defined `self.predict`.
        names : list[str]
            File base names of each item in the batch.
        out_dir : str
            The directory used to save output (changes for each epoch).
        kwargs : dict
            Additional keyword arguments used for generating output.
        """
        pred_dir = os.path.join(out_dir, 'feats')
        os.makedirs(pred_dir, exist_ok=True)

        n_frames = features['n_frames'].cpu().detach().numpy()
        for feat_name, values in output_features.items():

            if isinstance(values, torch.Tensor):
                values = values.cpu().detach().numpy()

            if isinstance(values, np.ndarray):
                values = [value[:n_frame] for value, n_frame in zip(values, n_frames)]

                tdt.file_io.save_dir(tdt.file_io.save_bin,
                                     path=os.path.join(pred_dir, feat_name),
                                     data=values,
                                     file_ids=names,
                                     feat_ext='.bin')

    def analysis_for_valid_batch(self, features, output_features, names, out_dir, **kwargs):
        r"""Hook used by :class:`morgana.experiment_builder.ExperimentBuilder` after validation batches for some epochs.

        Can be used to save output or generate visualisations.

        Parameters
        ----------
        features : dict[str, torch.Tensor]
            The ground truth features produced by `self.*_data_sources`.
        output_features : torch.Tensor or list[torch.Tensor] or dict[str, torch.Tensor]
            Predictions output by user-defined `self.predict`.
        names : list[str]
            File base names of each item in the batch.
        out_dir : str
            The directory used to save output (changes for each epoch).
        kwargs : dict
            Additional keyword arguments used for generating output.
        """
        self.analysis_for_train_batch(features, output_features, names, out_dir, **kwargs)

    def analysis_for_test_batch(self, features, output_features, names, out_dir, **kwargs):
        r"""Hook used by :class:`morgana.experiment_builder.ExperimentBuilder` after each testing batch.

        Can be used to save output or generate visualisations.

        Parameters
        ----------
        features : dict[str, torch.Tensor]
            The ground truth features produced by `self.*_data_sources`.
        output_features : torch.Tensor or list[torch.Tensor] or dict[str, torch.Tensor]
            Predictions output by user-defined :func:`~predict`.
        names : list[str]
            File base names of each item in the batch.
        out_dir : str
            The directory used to save output (changes for each epoch).
        kwargs : dict
            Additional keyword arguments used for generating output.
        """
        self.analysis_for_valid_batch(features, output_features, names, out_dir, **kwargs)


class BaseSPSS(BaseModel):
    r"""Creates an abstract SPSS acoustic model."""
    def __init__(self):
        super(BaseSPSS, self).__init__()

    def loss_fn(self, target, prediction):
        r"""Mean squared error loss."""
        return F.mse_loss(target, prediction, reduction='none')

    def forward(self, features):
        r"""Prediction and calculation of loss."""
        output_features = self.predict(features)

        loss = self.loss(features, output_features)

        return loss, output_features


class BaseVAE(BaseSPSS):
    r"""Creates an abstract VAE model, where the decoder corresponds to an SPSS model.

    Parameters
    ----------
    z_dim : int
        Dimensionality of the latent space.
    kld_weight : float
        Weight of the Kullback–Leibler divergence cost. Used to mitigate posterior collapse.
    """
    def __init__(self, z_dim=16, kld_weight=1.):
        super(BaseVAE, self).__init__()
        self.z_dim = z_dim
        self.kld_weight = kld_weight

        self.metrics.add_metrics(
            'all',
            kld=metrics.Mean())

    def encode(self, features):
        r"""VAE encoder.

        Parameters
        ----------
        features : dict[str, torch.Tensor]
            The ground truth features produced by `self.*_data_sources`.

        Returns
        -------
        mean : torch.Tensor, shape (batch_size, z_dim)
        log_variance : torch.Tensor, shape (batch_size, z_dim)
        """
        raise NotImplementedError("Encoder must be implemented in a subclass.")

    def sample(self, mean, log_variance):
        r"""Takes one sample from the approximate posterior (an isotropic Gaussian).

        Parameters
        ----------
        mean : torch.Tensor, shape (batch_size, z_dim)
        log_variance : torch.Tensor, shape (batch_size, z_dim)

        Returns
        -------
        latent_sample : torch.Tensor, shape (batch_size, z_dim)
        """
        std_dev = torch.exp(log_variance * 0.5)
        latent_sample = torch.distributions.Normal(mean, std_dev).rsample()
        return latent_sample

    def decode(self, latent, features):
        r"""VAE decoder.

        Parameters
        ----------
        latent : torch.Tensor, shape (batch_size, z_dim)
        features : dict[str, torch.Tensor]
            The ground truth features produced by `self.*_data_sources`.

        Returns
        -------
        output_features : torch.Tensor or list[torch.Tensor] or dict[str, torch.Tensor]
            Reconstructions from the model, can be any data structure containing `torch.Tensor` instances.
        """
        raise NotImplementedError("Decoder must be implemented in a subclass.")

    def forward(self, features):
        r"""Encodes the input features, samples from the encoding, reconstructs the input, and calculates the loss."""
        mean, log_variance = self.encode(features)
        latent_sample = self.sample(mean, log_variance)
        output_features = self.decode(latent_sample, features)

        output_features['latent'] = latent_sample
        output_features['mean'] = mean
        output_features['log_variance'] = log_variance

        loss = self.loss(features, output_features)

        return loss, output_features

    def predict(self, features):
        r"""Runs the model in testing mode (the encoder is not used), but the latent must be provided as an input."""
        latent = features['latent']
        return self.decode(latent, features)

    def _loss(self, targets, predictions, mean, log_variance, seq_lens=None, loss_weights=None):
        r"""Defines the loss helper to calculate the Kullback–Leibler divergence as well as the sequence loss."""
        if loss_weights is None:
            mse_weight = len(utils.listify(targets))
        else:
            mse_weight = sum(loss_weights)

        mse = super(BaseVAE, self)._loss(targets, predictions, seq_lens, loss_weights)
        mse *= mse_weight

        kld = losses.KLD_standard_normal(mean, log_variance)

        self.metrics.accumulate(
            self.mode,
            kld=kld)

        loss = (mse + kld * self.kld_weight) / (mse_weight + 1)

        return loss

