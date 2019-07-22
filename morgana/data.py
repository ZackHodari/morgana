import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from morgana import utils

from tts_data_tools import file_io


TO_TORCH_DTYPE = {
    np.dtype('float16'): torch.float16,
    np.dtype('float32'): torch.float32,
    np.dtype('float64'): torch.float64,
    np.dtype('int8'): torch.uint8,
    np.dtype('int16'): torch.int16,
    np.dtype('int32'): torch.int32,
    np.dtype('int64'): torch.int64,
    np.dtype('bool'): torch.uint8,
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
    normalisers : Normalisers or dict[str, _FeatureNormaliser]
        Normaliser instances used to normalise loaded features (and delta features).
    data_root : str
        The directory root for this dataset.

    Attributes
    ----------
    file_ids : list[str]
        List of base names loaded from `id_list`.
    normalisers : Normalisers or dict[str, _FeatureNormaliser]
        The normaliser instances, set automatically by :class:`morgana.experiment_builder.ExperimentBuilder`.
    """
    def __init__(self, data_sources, data_dir, id_list, normalisers, data_root='.'):
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
        base_name : str
            The name (without extensions) of the loaded file.
        """
        base_name = self.file_ids[index]

        features = {}
        for name, data_source in self.data_sources.items():
            data_source_features = data_source(base_name, self.data_dir)

            if name in self.normalisers:
                for feature_name, feature in list(data_source_features.items()):
                    is_deltas = feature_name.endswith('_deltas')
                    normalised_feature = self.normalisers[name].normalise(feature, deltas=is_deltas)

                    data_source_features['normalised_{}'.format(feature_name)] = normalised_feature.astype(np.float32)

            features.update(data_source_features)

        return features, base_name

    def __len__(self):
        return len(self.file_ids)

    @staticmethod
    def collate_fn(batch):
        r"""Collates a list of outputs from `self.__getitem__` into a batched structure.

        Parameters
        ----------
        batch : list[tuple[features, base_name]]
            features : dict[str, `np.array`]
                Features loaded by each data source, contained in a non-nested dictionary.
            base_name : str
                The name (without extensions) of the loaded file.

        Returns
        -------
        batched_features : dict[str, :class:`torch.Tensor`]
            Batched version of the list of `features` items in `batch`.
        base_names
            List of `base_name` strings in `batch`.
        """
        batch_size = len(batch)
        feature_template, _ = batch[0]

        def feature_list_to_batched_tensor(feature_list):
            """Handles padding and type conversion."""
            feature_item = feature_list[0]

            # Features can be a single number, or sequence data, in which case find the sequence length.
            if isinstance(feature_item, np.ndarray):
                max_seq_len = max(map(len, feature_list))
                feat_dim = feature_item.shape[-1]
                dtype = TO_TORCH_DTYPE[feature_item.dtype]

                # Padding is handled by creating a zeros tensor using the maximum sequence length.
                batched_feature = torch.zeros((batch_size, max_seq_len, feat_dim), dtype=dtype)

                for i, feature in enumerate(feature_list):
                    # Check if the array contains unsupported types, these need to be explicitly cast.
                    if feature.dtype in [np.bool, np.int8]:
                        feature = feature.astype(np.uint8)

                    seq_len = feature.shape[0]
                    batched_feature[i, :seq_len, ...] = torch.tensor(feature, dtype=dtype)

            else:
                dtype = TO_TORCH_DTYPE[type(feature_item)]
                batched_feature = torch.tensor(feature_list, dtype=dtype)

            return batched_feature

        # First transpose the list of dictionaries:
        #   from - [ { feat_name: _DataSource.load_file() }, base_name ]
        #   to - { feat_name: [ _DataSource.load_file() ] }, [ base_name ]
        features = {feat_name: [] for feat_name in feature_template.keys()}
        base_names = []
        for i, (item_features, base_name) in enumerate(batch):
            for feat_name, value in item_features.items():
                features[feat_name].append(value)
            base_names.append(base_name)

        # For all `feat_name` features in the batch (in feature_list) convert to a tensor.
        batched_features = {feat_name: [] for feat_name in feature_template.keys()}
        for feat_name, feature_list in features.items():
            batched_features[feat_name] = feature_list_to_batched_tensor(feature_list)

        return batched_features, base_names


class Normalisers(dict):
    r"""A dictionary-like container for normalisers, provides an interface for creating the normalisers.

    Parameters
    ----------
    normalisation_dir : str
        The directory containing the normalisation parameters (in a JSON file).
    data_root : str
        The directory root for this dataset.
    device : str or `torch.device`
        The name of the device to place the parameters on.
    """
    def __init__(self, data_sources, normalisation_dir, data_root='.', device='cpu'):
        super(Normalisers, self).__init__()

        self.normalisation_dir = normalisation_dir
        self.data_root = data_root
        self.device = device

        for name, data_source in data_sources.items():
            if data_source.normalisation is not None:
                self[name] = self.create_normaliser(name, data_source)

    def create_normaliser(self, name, data_source):
        r"""Creates the normaliser if one was specified for this data source.

        Parameters
        ----------
        name : str
            Name used to index this data source in the model.
        data_source : _DataSource
            Specification of how to load this feature.

        Returns
        -------
        _FeatureNormaliser
        """
        if data_source.normalisation == 'mvn':
            normaliser = MeanVaraianceNormaliser(
                name, self.normalisation_dir, data_source.use_deltas, self.device, self.data_root)
        elif data_source.normalisation == 'minmax':
            normaliser = MinMaxNormaliser(
                name, self.normalisation_dir, data_source.use_deltas, self.device, self.data_root)
        else:
            raise ValueError("Unknown or unsupported feature normaliser specified, {}"
                             .format(data_source.normalisation))

        return normaliser


class _FeatureNormaliser(object):
    r"""Abstract feature normaliser class. Exposes the :func:`~normalise` and :func:`~denormalise` methods.

    Normalisers will work on both NumPy arrays and PyTorch tensors. This is necessary to process NumPy arrays in
    :func:`_DataSource.__call__` and to normalise/denormalise PyTorch tensors in batch within the model.

    Parameters
    ----------
    feature_name : str
        Name of the feature.
    data_dir : str
        Directory containing all data for this dataset split.
    use_deltas : bool
        Whether to load normalisation parameters for delta features.
    device : str or `torch.device`
        Name of the device to place the parameters on.
    data_root : str
        Directory root for this dataset.
    file_pattern : str
        Format of the JSON file containing the normalisation parameters.
    """
    def __init__(self, feature_name, data_dir, use_deltas=False, device='cpu', data_root='.', file_pattern='{}.json'):
        self.feature_name = feature_name
        self.data_dir = os.path.join(data_root, data_dir)
        self.use_deltas = use_deltas
        self.device = device
        self.file_pattern = file_pattern

        self.params, self.params_torch = self.load_params(self.feature_name,
                                                          self.data_dir, self.device, self.file_pattern)

        if self.use_deltas:
            self.delta_params, self.delta_params_torch = self.load_params('{}_deltas'.format(self.feature_name),
                                                                          self.data_dir, self.device, self.file_pattern)

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
        """Gets the normalisation parameters, taking into account the delta flag and type of data."""
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
    def load_params(feature_name, data_dir, device='cpu', file_pattern='{}.json'):
        r"""Loads the parameters from file and converts them to NumPy arrays and PyTorch tensors."""
        feat_params = file_io.load_json(os.path.join(data_dir, file_pattern.format(feature_name)))

        params = {}
        params_torch = {}
        for param_name, param in feat_params.items():
            param = np.array(param, dtype=np.float32)

            params[param_name] = param
            params_torch[param_name] = torch.tensor(param).to(device)

        return params, params_torch


class MeanVaraianceNormaliser(_FeatureNormaliser):
    r"""Normalises features such that they have zero mean and unit variance.

    Normalisation:
        `norm_f = (f - mean) / std_dev`
    Denormalisation:
        `f = (norm_f * std_dev) + mean`

    Parameters
    ----------
    feature_name : str
        Name of the feature.
    data_dir : str
        Directory containing all data for this dataset split.
    use_deltas : bool
        Whether to load normalisation parameters for delta features.
    device : str or `torch.device`
        Name of the device to place the parameters on.
    data_root : str
        Directory root for this dataset.
    """
    def __init__(self, feature_name, data_dir, use_deltas=False, device='cpu', data_root='.'):
        super(MeanVaraianceNormaliser, self).__init__(
            feature_name, data_dir, use_deltas, device, data_root, '{}_mvn.json')

    def _normalise(self, feature, mean, std_dev):
        return (feature - mean) / (std_dev + 1e-8)

    def _denormalise(self, feature, mean, std_dev):
        return (feature * std_dev) + mean


class MinMaxNormaliser(_FeatureNormaliser):
    r"""Normalises features such that they have a minimum value of 0 and a maximum value of 1.

    Normalisation:
        `norm_f = (f - min) / (max - min)`
    Denormalisation:
        `f = norm_f * (max - min) + min`

    Parameters
    ----------
    feature_name : str
        Name of the feature.
    data_dir : str
        Directory containing all data for this dataset split.
    use_deltas : bool
        Whether to load normalisation parameters for delta features.
    device : str or `torch.device`
        Name of the device to place the parameters on.
    data_root : str
        Directory root for this dataset.
    """
    def __init__(self, feature_name, data_dir, use_deltas=False, device='cpu', data_root='.'):
        super(MinMaxNormaliser, self).__init__(
            feature_name, data_dir, use_deltas, device, data_root, '{}_minmax.json')

    def _normalise(self, feature, mmin, mmax):
        scale = mmax - mmin
        scale[abs(scale) <= 1e-8] = 1.

        return (feature - mmin) / scale

    def _denormalise(self, feature, mmin, mmax):
        scale = mmax - mmin
        scale[abs(scale) <= 1e-8] = 1.

        return (feature * scale) + mmin


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

    def __iter__(self):
        for features, names in self.data_loader:

            features_on_device = utils.map_nested(lambda tensor: tensor.to(self.torch_device), features)

            yield features_on_device, names

