import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from morgana import utils

from tts_data_tools import file_io
from tts_data_tools.utils import get_file_ids


TO_TORCH_DTYPE = {
    np.dtype('float16'): torch.float16,
    np.dtype('float32'): torch.float32,
    np.dtype('float64'): torch.float64,
    np.dtype('int8'): torch.int8,
    np.dtype('int16'): torch.int16,
    np.dtype('int32'): torch.int32,
    np.dtype('int64'): torch.int64,
    np.dtype('bool'): torch.bool,
    np.dtype('uint8'): torch.uint8,
    int: torch.int64,
    float: torch.float32,
    bool: torch.uint8
}


def batch(data_generator, batch_size=32, shuffle=True, num_data_threads=0, device='cpu'):
    r"""Creates the batched data loader for the dataset given, maps the batches to a given device.

    Parameters
    ----------
    data_generator : torch.utils.data.Dataset or FilesDataset
        Dataset from which to load the batches of data.
    batch_size : int
        Number of samples to load per batch.
    shuffle : bool
        Whether to shuffle the data every epoch.
    num_data_threads : int
        Number of parallel subprocesses to use for data loading.
    device : str
        Name of the device to place the parameters on.

    Returns
    -------
    :class:`torch.utils.data.DataLoader` (in a :class:`ToDeviceWrapper` container)
        An instance with the `__iter__` method, allowing for iteration over batches of the dataset.
    """
    data_loader = DataLoader(data_generator, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_data_threads, collate_fn=data_generator.collate_fn,
                             pin_memory=(torch.device(device) != torch.device('cpu')))

    # Must be separated as a wrapper, since DataLoader uses multiprocessing which doesn't always play nicely with CUDA.
    data_loader = ToDeviceWrapper(data_loader, device)

    return data_loader


class FilesDataset(Dataset):
    r"""Combines multiple :class:`_DataSource` instances, and enables batching of a dictionary of sequence features.

    Parameters
    ----------
    data_sources : dict[str, _DataSource]
        Specification of the different data to be loaded.
    data_dir : str
        The directory containing all data for this dataset split.
    id_list : str
        The name of the file id-list containing base names to load, contained withing `data_root`.
    normalisers : Normalisers or dict[str, _FeatureNormaliser] or dict[str, _SpeakerDependentNormaliser]
        Normaliser instances used to normalise loaded features (and delta features).
    data_root : str
        The directory root for this dataset.

    Attributes
    ----------
    file_ids : list[str]
        List of base names loaded from `id_list`.
    normalisers : Normalisers or dict[str, _FeatureNormaliser] or dict[str, _SpeakerDependentNormaliser]
        The normaliser instances, set automatically by :class:`morgana.experiment_builder.ExperimentBuilder`.

    Notes
    -----
    If any speaker-dependent normalisers are provided, the user must define a data source by the name `speaker_id`.
    """
    def __init__(self, data_sources, data_dir, id_list, normalisers, data_root='.'):
        # Check speaker ids will be generated if they are needed by any speaker dependent normalisers.
        for name, normaliser in normalisers.items():
            if isinstance(normaliser, _SpeakerDependentNormaliser) and 'speaker_id' not in data_sources:
                raise KeyError(f"{name} is a speaker-dependent normaliser, but no 'speaker_id' data_source was defined")

            if normaliser.use_deltas and not data_sources[name].use_deltas:
                raise ValueError(f'To normalise deltas of {name}, set `data_source.use_deltas` to True.')

        self.data_sources = data_sources
        self.data_root = data_root
        self.data_dir = os.path.join(self.data_root, data_dir)

        self.id_list = os.path.join(self.data_root, id_list)
        with open(self.id_list, 'r') as f:
            self.file_ids = list(filter(bool, map(str.strip, f.readlines())))

        self.normalisers = normalisers

    def __getitem__(self, index):
        r"""Combines the features loaded by each data source and adds normalised features where specified.

        Parameters
        ----------
        index : int
            The index of the file in the `id_list` to load.

        Returns
        -------
        features : dict[str, np.array]
            Features loaded by each data source, contained in a non-nested dictionary.
        """
        def _normalise_feature(feature, is_deltas=False):
            if isinstance(self.normalisers[name], _SpeakerDependentNormaliser):
                normalised_feature = \
                    self.normalisers[name].normalise(feature, features['speaker_id'], deltas=is_deltas)
            else:
                normalised_feature = \
                    self.normalisers[name].normalise(feature, deltas=is_deltas)

            return normalised_feature.astype(np.float32)

        base_name = self.file_ids[index]

        features = {'name': base_name}

        # If speaker ids are provided, extract them before the `for` loop so they can be used by `_normalise_feature`.
        if 'speaker_id' in self.data_sources:
            speaker_id = self.data_sources['speaker_id'](base_name, self.data_dir)
            features.update(speaker_id)

        for name, data_source in self.data_sources.items():
            if name == 'speaker_id':
                continue

            data_source_features = data_source(base_name, self.data_dir)

            if name in self.normalisers:
                data_source_features['normalised_{}'.format(name)] = \
                    _normalise_feature(data_source_features[name])

                if self.normalisers[name].use_deltas:
                    data_source_features['normalised_{}_deltas'.format(name)] = \
                        _normalise_feature(data_source_features['{}_deltas'.format(name)], is_deltas=True)

            features.update(data_source_features)

        return features

    def __len__(self):
        return len(self.file_ids)

    @staticmethod
    def collate_fn(batch):
        r"""Collates a list of outputs from `self.__getitem__` into a batched structure.

        Parameters
        ----------
        batch : list[dict[str, object]]
            Each element in the list is a non-nested dictionary containing features loaded by each data source.

        Returns
        -------
        batched_features : dict[str, :class:`torch.Tensor`]
            Batched version of the list of `features` items in `batch`.
            Note, it is possible to provide objects such as strings that will not be converted to `torch.Tensor`, these
            will not be padded or sent to the correct device, but can be accessed in the features dictionary.
        """
        batch_size = len(batch)
        feature_template = batch[0]

        def feature_list_to_batched_tensor(feature_list):
            """Handles padding and type conversion."""
            feature_item = feature_list[0]

            # Sequence feature.
            if isinstance(feature_item, np.ndarray) and feature_item.ndim > 1:
                max_seq_len = max(map(len, feature_list))
                feat_dim = feature_item.shape[-1]
                dtype = TO_TORCH_DTYPE[feature_item.dtype]

                # Padding is handled by creating a zeros tensor using the maximum sequence length.
                batched_feature = torch.zeros((batch_size, max_seq_len, feat_dim), dtype=dtype)

                for i, feature in enumerate(feature_list):
                    seq_len = feature.shape[0]
                    batched_feature[i, :seq_len, ...] = torch.tensor(feature, dtype=dtype)

            # Static 1 dimensional feature.
            elif isinstance(feature_item, np.ndarray) and feature_item.dtype in TO_TORCH_DTYPE:
                dtype = TO_TORCH_DTYPE[feature_item.dtype]
                batched_feature = torch.tensor(feature_list, dtype=dtype)

            # Static 0 dimensional feature.
            elif not isinstance(feature_item, np.ndarray) and type(feature_item) in TO_TORCH_DTYPE:
                dtype = TO_TORCH_DTYPE[type(feature_item)]
                batched_feature = torch.tensor(feature_list, dtype=dtype)

            # Feature that will not be converted to `torch.Tensor`.
            else:
                batched_feature = feature_list

            return batched_feature

        # First transpose the list of dictionaries:
        #   from - [ { feat_name: _DataSource.load_file() } ]
        #   to - { feat_name: [ _DataSource.load_file() ] }
        features = {feat_name: [] for feat_name in feature_template.keys()}
        for i, item_features in enumerate(batch):
            for feat_name, value in item_features.items():
                features[feat_name].append(value)

        # Convert all features in the batch to `torch.Tensors` if possible.
        batched_features = {feat_name: [] for feat_name in feature_template.keys()}
        for feat_name, feature_list in features.items():
            batched_features[feat_name] = feature_list_to_batched_tensor(feature_list)

        return batched_features


class Normalisers(dict):
    r"""A dictionary-like container for normalisers, loads parameters for all the normalisers.

    Parameters
    ----------
    normaliser_sources : dict[str, _FeatureNormaliser]
        Specification of the normalisers.
    normalisation_dir : str
        The directory containing the normalisation parameters (in a JSON file).
    data_root : str
        The directory root for this dataset.
    device : str or `torch.device`
        The name of the device to place the parameters on.
    """
    def __init__(self, normaliser_sources, normalisation_dir, data_root='.', device='cpu'):
        super(Normalisers, self).__init__()

        self.normalisation_dir = os.path.join(data_root, normalisation_dir)
        self.device = device

        for name, normaliser_source in normaliser_sources.items():
            self[name] = normaliser_source
            self[name].load_params(self.normalisation_dir, self.device)


class _FeatureNormaliser(object):
    r"""Abstract feature normaliser class. Exposes the :func:`~normalise` and :func:`~denormalise` methods.

    Normalisers will work on both NumPy arrays and PyTorch tensors. This is necessary to process NumPy arrays in
    :func:`_DataSource.__call__` and to normalise/denormalise PyTorch tensors in batch within the model.

    Parameters
    ----------
    name : str
        Name of the feature.
    use_deltas : bool
        Whether to load normalisation parameters for delta features.
    file_pattern : str
        Format of the JSON file containing the normalisation parameters.

    Attributes
    ----------
    params : dict[str, np.ndarray]
    params_torch : dict[str, torch.Tensor]
    delta_params : dict[str, np.ndarray]
    delta_params_torch : dict[str, torch.Tensor]
    """
    def __init__(self, name, use_deltas=False, file_pattern='{name}.json'):
        self.name = name
        self.use_deltas = use_deltas
        self.file_pattern = file_pattern

        self.params = None
        self.params_torch = None

        if self.use_deltas:
            self.delta_params = None
            self.delta_params_torch = None

    def _normalise(self, feature, **params):
        raise NotImplementedError("Underlying calculation of normalisation should be implemented in a subclass.")

    def _denormalise(self, feature, **params):
        raise NotImplementedError("Underlying calculation of denormalisation should be implemented in a subclass.")

    def normalise(self, feature, deltas=False):
        r"""Normalises the sequence feature.

        Parameters
        ----------
        feature : np.ndarray or torch.Tensor, shape (batch_size, seq_len, feat_dim) or (seq_len, feat_dim)
            Sequence feature to be normalised, can be a NumPy array or a PyTorch tensor, can be batched.
        deltas : bool
            Whether `feature` is a delta feature, and should be normalised using the delta parameters.

        Returns
        -------
        np.ndarray or torch.Tensor, shape (batch_size, seq_len, feat_dim) or (seq_len, feat_dim)
            Normalised sequence feature.
        """
        params = self.fetch_params(type(feature), deltas=deltas)
        return self._normalise(feature, **params)

    def denormalise(self, feature, deltas=False):
        r"""De-normalises the sequence feature.

        Parameters
        ----------
        feature : np.ndarray or torch.Tensor, shape (batch_size, seq_len, feat_dim) or (seq_len, feat_dim)
            Sequence feature to be normalised, can be a NumPy array or a PyTorch tensor, can be batched.
        deltas : bool
            Whether `feature` is a delta feature, and should be normalised using the delta parameters.

        Returns
        -------
        np.ndarray or torch.Tensor, shape (batch_size, seq_len, feat_dim) or (seq_len, feat_dim)
            Normalised sequence feature.
        """
        params = self.fetch_params(type(feature), deltas=deltas)
        return self._denormalise(feature, **params)

    def fetch_params(self, data_type=np.ndarray, deltas=False):
        r"""Gets the normalisation parameters, taking into account the delta flag and type of data."""
        if deltas:
            if data_type == torch.Tensor:
                return self.delta_params_torch
            else:
                return self.delta_params

        else:
            if data_type == torch.Tensor:
                return self.params_torch
            else:
                return self.params

    @staticmethod
    def _from_json(file_path):
        r"""Loads parameters from JSON file and converts to `np.ndarray`s."""
        feat_params = file_io.load_json(file_path)

        params = {}
        for param_name, param in feat_params.items():
            params[param_name] = np.array(param, dtype=np.float32)

        return params

    @staticmethod
    def _to_torch(params, device='cpu'):
        r"""Converts dictionary of parameters to `torch.Tensor`s on the specified device."""
        params_torch = {}
        for param_name, param in params.items():
            params_torch[param_name] = torch.tensor(param).to(device)

        return params_torch

    def load_params(self, data_dir, data_root='.', device='cpu'):
        r"""Loads the parameters from file and converts them to NumPy arrays and PyTorch tensors.

        Parameters
        ----------
        data_dir : str
            Directory containing all data for this dataset split.
        data_root : str
            Directory root for this dataset.
        device : str or torch.device
            Name of the device to place the parameters on.
        """
        params_file = os.path.join(
            data_root, data_dir, self.file_pattern.format(name=self.name))

        self.params = self._from_json(params_file)
        self.params_torch = self._to_torch(self.params, device=device)

        if self.use_deltas:
            delta_params_file = os.path.join(
                data_root, data_dir, self.file_pattern.format(name=self.name + '_deltas'))

            self.delta_params = self._from_json(delta_params_file)
            self.delta_params_torch = self._to_torch(self.delta_params, device=device)


class _SpeakerDependentNormaliser(_FeatureNormaliser):
    r"""Speaker-dependent feature normaliser class, wraps individual normalisers exposing speaker identity argument.

    Parameters
    ----------
    name : str
        Name of the feature.
    speaker_id_list : str
        File name of the id list containing speaker names, used to load parameters for all speakers.
    use_deltas : bool
        Whether to load normalisation parameters for delta features.
    file_pattern : str
        Format of the JSON file containing the normalisation parameters.

    Attributes
    ----------
    speaker_ids : list[str]
        Names of speakers for each batch item in `feature`.
    """
    def __init__(self, name, speaker_id_list, use_deltas=False, file_pattern='{speaker_id}/{name}.json'):
        super(_SpeakerDependentNormaliser, self).__init__(name, use_deltas=use_deltas, file_pattern=file_pattern)

        self.speaker_id_list = speaker_id_list
        self.speaker_ids = None

        self.params = {}
        self.params_torch = {}

        if self.use_deltas:
            self.delta_params = {}
            self.delta_params_torch = {}

    def normalise(self, feature, speaker_ids, deltas=False):
        r"""Normalises the sequence feature based on speaker-dependent normalisation parameters.

        Parameters
        ----------
        feature : np.ndarray or torch.Tensor, shape (batch_size, seq_len, feat_dim) or (seq_len, feat_dim)
            Sequence feature to be normalised, can be a NumPy array or a PyTorch tensor, can be batched.
        speaker_ids : list[str] or str
            Names of speakers for each batch item in `feature`
        deltas : bool
            Whether `feature` is a delta feature, and should be normalised using the delta parameters.

        Returns
        -------
        np.ndarray or torch.Tensor, shape (batch_size, seq_len, feat_dim) or (seq_len, feat_dim)
            Normalised sequence feature.
        """
        params = self.fetch_params(speaker_ids, type(feature), deltas=deltas)
        return self._normalise(feature, **params)

    def denormalise(self, feature, speaker_ids, deltas=False):
        r"""De-normalises the sequence feature based on speaker-dependent normalisation parameters.

        Parameters
        ----------
        feature : np.ndarray or torch.Tensor, shape (batch_size, seq_len, feat_dim) or (seq_len, feat_dim)
            Sequence feature to be normalised, can be a NumPy array or a PyTorch tensor, can be batched.
        speaker_ids : list[str] or str
            Names of speakers for each batch item in `feature`
        deltas : bool
            Whether `feature` is a delta feature, and should be normalised using the delta parameters.

        Returns
        -------
        np.ndarray or torch.Tensor, shape (batch_size, seq_len, feat_dim) or (seq_len, feat_dim)
            Normalised sequence feature.
        """
        params = self.fetch_params(speaker_ids, type(feature), deltas=deltas)
        return self._denormalise(feature, **params)

    def fetch_params(self, speaker_ids, data_type=np.ndarray, deltas=False):
        r"""Gets the speaker-dependent normalisation parameters, taking into account the delta flag and type of data.

        Parameters
        ----------
        speaker_ids : list[str]
            Names of speakers for each batch item.
        data_type : type
            Typically `torch.Tensor` for batched features, or `np.ndarray` for single sentences or visualisation code.
        deltas : bool
            Whether `feature` is a delta feature, and should be normalised using the delta parameters.

        Returns
        -------
        sd_params : dict[str, torch.Tensor] or dict[str, np.ndarray], shape (batch_size, feat_dim) or (feat_dim)
            The speaker dependent parameters
        """
        speaker_ids = utils.listify(speaker_ids)
        speaker_params = super(_SpeakerDependentNormaliser, self).fetch_params(data_type=data_type, deltas=deltas)

        sd_params = {}
        for speaker_id in speaker_ids:

            params = speaker_params[speaker_id]

            for name, param in params.items():
                # For current speaker_id (item in batch) and current parameter (e.g. mean), concatenate along dim=0
                param = param[None, ...]

                if name not in sd_params:
                    sd_params[name] = param

                else:
                    if data_type == torch.Tensor:
                        sd_params[name] = torch.cat((sd_params[name], param))
                    else:
                        sd_params[name] = np.concatenate((sd_params[name], param))

        for name, sd_param in sd_params.items():
            sd_params[name] = sd_param.squeeze(0)

        return sd_params

    def load_params(self, data_dir, data_root='.', device='cpu'):
        r"""Loads the parameters for all speakers from file and stacks them in NumPy arrays and PyTorch tensors.

        Parameters
        ----------
        data_dir : str
            Directory containing all data for this dataset split.
        data_root : str
            Directory root for this dataset.
        device : str or torch.device
            Name of the device to place the parameters on.
        """
        if self.speaker_ids is None:
            self.speaker_ids = get_file_ids(id_list=os.path.join(data_root, self.speaker_id_list))

        for speaker_id in self.speaker_ids:
            params_file = os.path.join(
                data_root, data_dir, self.file_pattern.format(name=self.name, speaker_id=speaker_id))

            self.params[speaker_id] = self._from_json(params_file)
            self.params_torch[speaker_id] = self._to_torch(self.params[speaker_id], device=device)

            if self.use_deltas:
                delta_params_file = os.path.join(
                    data_root, data_dir, self.file_pattern.format(speaker_id=speaker_id, name=self.name + '_deltas'))

                self.delta_params[speaker_id] = self._from_json(delta_params_file)
                self.delta_params_torch[speaker_id] = self._to_torch(self.delta_params[speaker_id], device=device)


def normalise_mvn(feature, mean, std_dev):
    return (feature - mean[..., None, :]) / (std_dev[..., None, :] + 1e-8)


def denormalise_mvn(feature, mean, std_dev):
    return (feature * std_dev[..., None, :]) + mean[..., None, :]


class MeanVarianceNormaliser(_FeatureNormaliser):
    r"""Normalises features such that they have zero mean and unit variance.

    Normalisation:
        `norm_f = (f - mean) / std_dev`
    Denormalisation:
        `f = (norm_f * std_dev) + mean`

    Parameters
    ----------
    name : str
        Name of the feature.
    use_deltas : bool
        Whether to load normalisation parameters for delta features.
    """
    def __init__(self, name, use_deltas=False):
        super(MeanVarianceNormaliser, self).__init__(
            name, use_deltas, '{name}_mvn.json')

    def _normalise(self, feature, **params):
        return normalise_mvn(feature, params['mean'], params['std_dev'])

    def _denormalise(self, feature, **params):
        return denormalise_mvn(feature, params['mean'], params['std_dev'])


class SpeakerDependentMeanVarianceNormaliser(_SpeakerDependentNormaliser):
    def __init__(self, name, speaker_id_list, use_deltas=False):
        super(SpeakerDependentMeanVarianceNormaliser, self).__init__(
            name, speaker_id_list, use_deltas, '{speaker_id}/{name}_mvn.json')

    def _normalise(self, feature, **params):
        return normalise_mvn(feature, params['mean'], params['std_dev'])

    def _denormalise(self, feature, **params):
        return denormalise_mvn(feature, params['mean'], params['std_dev'])


def normalise_minmax(feature, mmin, mmax):
    scale = mmax - mmin
    scale[abs(scale) <= 1e-8] = 1.

    return (feature - mmin[..., None, :]) / scale[..., None, :]


def denormalise_minmax(feature, mmin, mmax):
    scale = mmax - mmin
    scale[abs(scale) <= 1e-8] = 1.

    return (feature * scale[..., None, :]) + mmin[..., None, :]


class MinMaxNormaliser(_FeatureNormaliser):
    r"""Normalises features such that they have a minimum value of 0 and a maximum value of 1.

    Normalisation:
        `norm_f = (f - min) / (max - min)`
    Denormalisation:
        `f = norm_f * (max - min) + min`

    Parameters
    ----------
    name : str
        Name of the feature.
    use_deltas : bool
        Whether to load normalisation parameters for delta features.
    """
    def __init__(self, name, use_deltas=False):
        super(MinMaxNormaliser, self).__init__(
            name, use_deltas, '{name}_minmax.json')

    def _normalise(self, feature, **params):
        return normalise_minmax(feature, params['mmin'], params['mmax'])

    def _denormalise(self, feature, **params):
        return denormalise_minmax(feature, params['mmin'], params['mmax'])


class SpeakerDependentMinMaxNormaliser(_SpeakerDependentNormaliser):
    def __init__(self, name, speaker_id_list, use_deltas=False):
        super(SpeakerDependentMinMaxNormaliser, self).__init__(
            name, speaker_id_list, use_deltas, '{speaker_id}/{name}_minmax.json')

    def _normalise(self, feature, **params):
        return normalise_minmax(feature, params['mmin'], params['mmax'])

    def _denormalise(self, feature, **params):
        return denormalise_minmax(feature, params['mmin'], params['mmax'])


class _DataLoaderWrapper(object):
    r"""Abstract :class:`torch.utils.data.DataLoader` wrapper. Allows attribute reference for underlying data loader."""
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def __getattr__(self, attr):
        """Recursively calls `__getattr__` until `self.data_loader` is the underlying data loader instance."""
        if isinstance(self.data_loader, DataLoader):
            return self.data_loader.__getattribute__(attr)
        else:
            # Recurse down until we get to the actual DataLoader.
            return self.data_loader.__getattr__(attr)

    def __len__(self):
        return len(self.data_loader)


class ToDeviceWrapper(_DataLoaderWrapper):
    r"""Wraps the `__iter__` method of :class:`torch.utils.data.DataLoader`, mapping each batch to a given device."""
    def __init__(self, data_loader, device):
        super(ToDeviceWrapper, self).__init__(data_loader)

        self.torch_device = torch.device(device)

    def to_device(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.torch_device)
        else:
            return tensor

    def __iter__(self):
        for features in self.data_loader:
            yield utils.map_nested(self.to_device, features)

