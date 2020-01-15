import os

import numpy as np
import torch
import torch.nn as nn

from morgana import metrics

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

    def finalise_init(self):
        r"""Called at the end of ExperiemntBuilder.__init__"""
        pass

    def normaliser_sources(self):
        r"""Specifies the normalisers that will be initialised and used by `FilesDataset`.

        Only specifies what data will be loaded, but not where from.

        Returns
        -------
        normalisers : data.Normalisers or dict[str, data._FeatureNormaliser or data._SpeakerDependentNormaliser]
            The normalisers used by :class:`morgana.data.FilesDataset`.
        """
        return {}

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
        r"""Defines the loss used to train the model.

        If you are calculating losses on padded sequences, wrap your loss function with `losses.sequence_loss`.

        Parameters
        ----------
        features : dict[str, torch.Tensor]
            The ground truth features produced by `self.*_data_sources`.
        output_features : torch.Tensor or list[torch.Tensor] or dict[str, torch.Tensor]
            Predictions output by user-defined :func:`~predict`.

        Returns
        -------
        float
            Overall loss between `features` and `output_features`.
        """
        raise NotImplementedError("Loss must be implemented in a subclass.")

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

        Returns
        -------
        state_dict : dict
            Parameters and persistent buffers that define the model.
        """
        state_dict = torch.load(checkpoint_path, map_location=device)
        super(BaseModel, self).load_state_dict(state_dict, strict=strict)
        return state_dict

    def analysis_for_train_batch(self, features, output_features, out_dir, **kwargs):
        r"""Hook used by :class:`morgana.experiment_builder.ExperimentBuilder` after training batches for some epochs.

        Can be used to save output or generate visualisations.

        Parameters
        ----------
        features : dict[str, torch.Tensor]
            The ground truth features produced by `self.*_data_sources`.
        output_features : torch.Tensor or list[torch.Tensor] or dict[str, torch.Tensor]
            Predictions output by user-defined `self.predict`.
        out_dir : str
            The directory used to save output (changes for each epoch).
        kwargs : dict
            Additional keyword arguments used for generating output.
        """
        pass

    def analysis_for_valid_batch(self, features, output_features, out_dir, **kwargs):
        r"""Hook used by :class:`morgana.experiment_builder.ExperimentBuilder` after validation batches for some epochs.

        Can be used to save output or generate visualisations.

        Parameters
        ----------
        features : dict[str, torch.Tensor]
            The ground truth features produced by `self.*_data_sources`.
        output_features : torch.Tensor or list[torch.Tensor] or dict[str, torch.Tensor]
            Predictions output by user-defined `self.predict`.
        out_dir : str
            The directory used to save output (changes for each epoch).
        kwargs : dict
            Additional keyword arguments used for generating output.
        """
        self.analysis_for_train_batch(features, output_features, out_dir, **kwargs)

    def analysis_for_test_batch(self, features, output_features, out_dir, **kwargs):
        r"""Hook used by :class:`morgana.experiment_builder.ExperimentBuilder` after each testing batch.

        Can be used to save output or generate visualisations.

        Parameters
        ----------
        features : dict[str, torch.Tensor]
            The ground truth features produced by `self.*_data_sources`.
        output_features : torch.Tensor or list[torch.Tensor] or dict[str, torch.Tensor]
            Predictions output by user-defined :func:`~predict`.
        out_dir : str
            The directory used to save output (changes for each epoch).
        kwargs : dict
            Additional keyword arguments used for generating output.
        """
        self.analysis_for_valid_batch(features, output_features, out_dir, **kwargs)

    def analysis_for_train_epoch(self, out_dir, **kwargs):
        r"""Hook used by :class:`morgana.experiment_builder.ExperimentBuilder` after some training epochs.

        Can be used to save output or generate visualisations.

        Parameters
        ----------
        out_dir : str
            The directory used to save output (changes for each epoch).
        kwargs : dict
            Additional keyword arguments used for generating output.
        """
        pass

    def analysis_for_valid_epoch(self, out_dir, **kwargs):
        r"""Hook used by :class:`morgana.experiment_builder.ExperimentBuilder` after some validation epochs.

        Can be used to save output or generate visualisations.

        Parameters
        ----------
        out_dir : str
            The directory used to save output (changes for each epoch).
        kwargs : dict
            Additional keyword arguments used for generating output.
        """
        self.analysis_for_train_epoch(out_dir, **kwargs)

    def analysis_for_test_epoch(self, out_dir, **kwargs):
        r"""Hook used by :class:`morgana.experiment_builder.ExperimentBuilder` after each testing epoch.

        Can be used to save output or generate visualisations.

        Parameters
        ----------
        out_dir : str
            The directory used to save output (changes for each epoch).
        kwargs : dict
            Additional keyword arguments used for generating output.
        """
        self.analysis_for_valid_epoch(out_dir, **kwargs)


class BaseSPSS(BaseModel):
    r"""Creates an abstract SPSS acoustic model."""
    def __init__(self):
        super(BaseSPSS, self).__init__()

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
        if 'latent' in features:
            latent = features['latent']
        else:
            _, feature = next(iter(features.items()))
            batch_size = feature.shape[0]
            device = feature.device

            # If no latent is provided, default to using the zero vector.
            latent = torch.zeros((batch_size, self.z_dim)).to(device)

        return self.decode(latent, features)

