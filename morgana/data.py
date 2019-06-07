import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from morgana import utils

import tts_data_tools as tdt


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


class _DataSource(object):
    r"""Abstract data loading class.

    Parameters
    ----------
    name : str
        Name of directory that will contain this feature.
    normalisation : {None, 'mvn', 'minmax'}
        Type of normalisation to perform. If not None, a JSON file containing the normalisation parameters must be
        present adjacent to the directory that contains the features.
    use_deltas : bool
        Whether to compute delta features. If normalisation is being used it will also perform normalisation of deltas.
    ext : str, optional
        The file extension of the saved features, if not set `self.name` is used.

    Attributes
    ----------
    normaliser : _FeatureNormaliser
        The normaliser instance, set automatically by :class:`morgana.experiment_builder.ExperimentBuilder`.

    Notes
    -----
    The data setup assumes a folder structure such as the following example,

    .. code-block::

        dataset_name (data_root)

            train (data_dir)

                lab (name)
                    *.lab
                lab.dim
                lab_minmax.json

                lf0 (name)
                    *.npy
                lf0.dim
                lf0_mvn.json
                lf0_deltas_mvn.json
                ...

            valid (data_dir)
                ...

            ...

    All data is contained below `data_root`.

    There can be multiple `data_dir` directories, e.g. one for each data split (train, valid, test).

    Each feature should have a directory within `data_dir`, this will contain all files for this feature.

    If the feature is to be loaded with :func:`np.fromfile` a `.dim` text file can be included, this can be used to
    determine the number of feature dimension in the files being loaded.

    There should be JSON files within the `data_dir` used for the training split, these JSON files should contain the
    normalisation parameters. There should be an additional JSON file for delta features if deltas need to be used.
    """
    def __init__(self, name, normalisation=None, use_deltas=False, ext=None):
        if normalisation not in [None, 'mvn', 'minmax']:
            raise ValueError("Normalisation for feature {} not known/supported: {}".format(name, normalisation))

        self.name = name
        self.normalisation = normalisation
        self.use_deltas = use_deltas
        self.ext = ext if ext is not None else name

        # This should be set by `FilesDataset` if a normaliser is to be used.
        self.normaliser = None

    def create_normaliser(self, normalisation_dir, data_root, device):
        r"""Creates the normaliser if one was specified for this data source.

        Parameters
        ----------
        normalisation_dir : str
            The directory containing the normalisation parameters (in a JSON file).
        data_root : str
            The directory root for this dataset.
        device : str or `torch.device`
            The name of the device to place the parameters on.

        Returns
        -------
        None or _FeatureNormaliser
        """
        if self.normalisation == 'mvn':
            normaliser = MeanVaraianceNormaliser(self.name, normalisation_dir, self.use_deltas, device, data_root)
        elif self.normalisation == 'minmax':
            normaliser = MinMaxNormaliser(self.name, normalisation_dir, self.use_deltas, device, data_root)
        else:
            normaliser = None

        return normaliser

    def file_path(self, base_name, data_dir):
        r"""Creates file path for a given base name and data directory."""
        return os.path.join(data_dir, self.name, '{}.{}'.format(base_name, self.ext))

    def load_file(self, base_name, data_dir):
        r"""Loads the contents of a given file. Must either be a sequence feature with 2 dimensions or a scalar value.

        Parameters
        ----------
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.

        Returns
        -------
        int or float or bool or `np.ndarray`, shape (seq_len, feat_dim)
        """
        raise NotImplementedError

    def __call__(self, base_name, data_dir):
        r"""Loads the feature and creates deltas and/or normalised versions if specified by this data source.

        Parameters
        ----------
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.

        Returns
        -------
        dict[str, (int or float or bool or np.ndarray)]
            Loaded feature, and if specified delta and/or normalised versions of the feature.
        """
        feature = self.load_file(base_name, data_dir)
        features = {self.name: feature}

        if self.use_deltas:
            deltas = tdt.wav_features.compute_deltas(feature)
            features['{}_deltas'.format(self.name)] = deltas.astype(np.float32)

        if self.normaliser is not None:
            normalised = self.normaliser.normalise(feature)
            features['normalised_{}'.format(self.name)] = normalised.astype(np.float32)

            if self.use_deltas:
                normalised_deltas = self.normaliser.normalise(deltas, deltas=True)
                features['normalised_{}_deltas'.format(self.name)] = normalised_deltas.astype(np.float32)

        return features


class NumpyBinarySource(_DataSource):
    r"""Data loading class for features saved with `np.ndarray.tofile`, loading is thus performed using `np.fromfile`.

    Parameters
    ----------
    name : str
        Name of directory that will contain this feature.
    normalisation : {None, 'mvn', 'minmax'}
        Type of normalisation to perform. If not None, a JSON file containing the normalisation parameters must be
        present adjacent to the directory that contains the features.
    use_deltas : bool
        Whether to compute delta features. If normalisation is being used it will also perform normalisation of deltas.
    ext : str, optional
        The file extension of the saved features, if not set `name` is used.
    dtype : type
        The numpy dtype to use when loading with `np.fromfile`.
    dim : int, optional
        The dimensionality of the feature being loaded. If None, this will be searched for in the file `{name}.dim`.
    """
    def __init__(self, name, normalisation=None, use_deltas=False, ext=None, dtype=np.float32, dim=None):
        super(NumpyBinarySource, self).__init__(name, normalisation, use_deltas, ext)

        self.dtype = dtype
        self.dim = dim

    def get_feat_dim(self, data_dir):
        r"""Gets the dimensionality of the feature being loaded from the file `{name}.dim`."""
        feat_dim_path = os.path.join(data_dir, '{}.dim'.format(self.name))
        feat_dim = tdt.file_io.load_txt(feat_dim_path).item()
        return feat_dim

    def load_file(self, base_name, data_dir):
        r"""Loads the feature using `np.fromfile`.

        Parameters
        ----------
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.

        Returns
        -------
        int or float or bool or np.ndarray, shape (seq_len, feat_dim)
        """
        if self.dim is None:
            self.dim = self.get_feat_dim(data_dir)

        feat_path = self.file_path(base_name, data_dir)
        feature = tdt.file_io.load_bin(feat_path, feat_dim=self.dim, dtype=self.dtype)

        return feature


class TextSource(_DataSource):
    r"""Loads data from a text file, this can contain integers or floats and will have up to 2 dimensions.

    Parameters
    ----------
    name : str
        Name of directory that will contain this feature.
    normalisation : {None, 'mvn', 'minmax'}
        Type of normalisation to perform. If not None, a JSON file containing the normalisation parameters must be
        present adjacent to the directory that contains the features.
    use_deltas : bool
        Whether to compute delta features. If normalisation is being used it will also perform normalisation of deltas.
    ext : str, optional
        The file extension of the saved features, if not set `name` is used.
    """
    def __init__(self, name, normalisation=None, use_deltas=False, ext=None):
        super(TextSource, self).__init__(name, normalisation, use_deltas, ext)

    def load_file(self, base_name, data_dir):
        r"""Loads the feature using `tts_data_tools.file_io.load_txt`.

        Parameters
        ----------
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.

        Returns
        -------
        int or float or np.ndarray, shape (seq_len, feat_dim)
        """
        feat_path = self.file_path(base_name, data_dir)
        feature = tdt.file_io.load_txt(feat_path)

        # If the sequence length feature is describing a sentence level length, convert it to a scalar.
        if feature.shape[0] == 1:
            feature = feature.item()

        return feature


class StringSource(_DataSource):
    r"""Loads data from a text file, this will be loaded as strings where each item should be on a new line.

    Parameters
    ----------
    name : str
        Name of directory that will contain this feature.
    ext : str, optional
        The file extension of the saved features, if not set `name` is used.
    """
    def __init__(self, name, ext=None):
        super(StringSource, self).__init__(name, normalisation=None, use_deltas=False, ext=ext)

    def load_file(self, base_name, data_dir):
        r"""Loads the lines and converts to ascii codes (np.int8), each line is considered as a sequence item.

        Each line can have a different number of characters, the maximum number of characters will be used to determine
        the shape of the 2nd dimension of the array.

        Parameters
        ----------
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.

        Returns
        -------
        np.ndarray, shape (seq_len, max_num_characters), dtype (np.int8)
        """
        feat_path = self.file_path(base_name, data_dir)
        lines = tdt.file_io.load_lines(feat_path)

        # Convert the strings into ASCII integers. Padding is also partially handled here.
        feature = utils.string_to_ascii(lines)
        return feature


class WavSource(_DataSource):
    r"""Loads wavfiles using `scipy.io.wavfile`.

    Parameters
    ----------
    name : str
        Name of directory that will contain this feature.
    normalisation : {None, 'mvn', 'minmax'}
        Type of normalisation to perform. If not None, a JSON file containing the normalisation parameters must be
        present adjacent to the directory that contains the features.
    use_deltas : bool
        Whether to compute delta features. If normalisation is being used it will also perform normalisation of deltas.
    """
    def __init__(self, name, normalisation=None, use_deltas=False):
        super(WavSource, self).__init__(name, normalisation, use_deltas, ext='wav')

    def load_file(self, base_name, data_dir):
        r"""Loads a wavfile using `scipy.io.wavfile`.

        Parameters
        ----------
        base_name : str
            The name (without extensions) of the file to be loaded.
        data_dir : str
            The directory containing all feature types for this dataset.

        Returns
        -------
        np.ndarray, shape (num_samples,), dtype (np.int16)
        """
        feat_path = self.file_path(base_name, data_dir)
        feature = tdt.file_io.load_wav(feat_path)

        return feature


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
    normalisers : None or dict[str, _FeatureNormaliser]
        Normalisers to be passed to the :class:`_DataSource` instances.
    data_root : str
        The directory root for this dataset.

    Attributes
    ----------
    file_ids : list[str]
        List of base names loaded from `id_list`.
    """
    def __init__(self, data_sources, data_dir, id_list, normalisers=None, data_root='.'):
        self.data_sources = data_sources
        self.data_root = data_root
        self.data_dir = os.path.join(self.data_root, data_dir)
        self.id_list = os.path.join(self.data_root, id_list)

        if normalisers is not None:
            for name, normaliser in normalisers.items():
                self.data_sources[name].normaliser = normaliser

        self.file_ids = tdt.file_io.load_lines(self.id_list)

    def __getitem__(self, index):
        r"""Combines the features loaded by each data source.

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
            features.update(data_source(base_name, self.data_dir))

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
        feat_params = tdt.file_io.load_json(os.path.join(data_dir, file_pattern.format(feature_name)))

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

