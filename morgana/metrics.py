from collections.abc import Iterable

import torch

from morgana import utils


class StatefulMetric(object):
    r"""Abstract class for accumulating information and calculating a result using current stored data.

    Three abstract methods must be implemented so that a metric can be calculated in an online fashion.
        * :func:`reset_state`
        * :func:`accumulate`
        * :func:`result`

    Parameters
    ----------
    hidden : bool
        Whether to hide the metric when being summarised by :class:`Handler`.
    """
    def __init__(self, hidden=False):
        super(StatefulMetric, self).__init__()

        self.hidden = hidden

    def reset_state(self, *args):
        r"""Creates any stateful variables and sets their initial values."""
        raise NotImplementedError

    def accumulate(self, *args, **kwargs):
        r"""Accumulates a batch of values into the stateful variables."""
        raise NotImplementedError

    def result(self, *args):
        r"""Calculates the current result using the stateful variables."""
        raise NotImplementedError

    def result_as_json(self, *args):
        r"""If the result is a :class:`torch.Tensor` then it must be converted to `np.ndarray` to be saved as JSON."""
        tensor = self.result(*args)
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy().tolist()

        return tensor

    def __str__(self):
        return utils.format_float_tensor(self.result())


class Handler(StatefulMetric):
    r"""Container for running a set of metrics.

    Parameters
    ----------
    metrics : dict[str, StatefulMetric]
        Name of metrics and StatefulMetric instances that will be assigned to all metric collections.

    Attributes
    ----------
    collections : dict[str, dict[str, StatefulMetric]]
        Multiple collections, each containing a map of metrics. Metrics can be overlapping between collections. It is
        possible to have multiple collections as different training modes (train/valid) may involve different metrics.
    """
    def __init__(self, **metrics):
        super(Handler, self).__init__(hidden=False)

        self.collections = {
            'all': metrics,
            'train': {},
            'valid': {},
            'test': {}}

        # This is an alias for all metrics, i.e. self.collections['all']
        self.metrics = self.collections['all']

        # Ensure that the values of 'train' and 'valid' are new dictionaries, not copies of 'all'
        self.add_metrics(('train', 'valid'), **metrics)

    def __getitem__(self, name):
        r"""Gets a collection of metrics (a dictionary) by name."""
        if name in self.collections:
            return self.collections[name]
        else:
            raise ValueError("No collection found by the name {}".format(name))

    def add_metrics(self, collections=('all',), **kwargs):
        r"""Updates the given collections with all names and metrics in `kwargs`.

        The new metrics will also be added to `self.metrics`, even if no collections are specified.

        Parameters
        ----------
        collections : str or list[str]
            Name(s) of collections to update with the given metrics.
        kwargs : dict[str, StatefulMetric]
            Name of metrics and :class:`~StatefulMetric` instances to be added.
        """
        if not isinstance(collections, Iterable) or isinstance(collections, str):
            collections = [collections]

        if 'all' in collections:
            collections = self.collections.keys()

        for collection_name in collections:
            self.collections[collection_name].update(kwargs)

        self.metrics.update(kwargs)

    def add_collection(self, collection, from_collections=tuple()):
        r"""Creates a new collection, and copies the metrics of other collection(s).

        Parameters
        ----------
        collection : str
            Name of new collection.
        from_collections
            Name(s) of collections from which to copy existing metrics.
        """
        if not isinstance(from_collections, Iterable) or isinstance(from_collections, str):
            from_collections = [from_collections]

        self.collections[collection] = {}

        for from_collection in from_collections:
            self[collection].update(self[from_collection])

    def reset_state(self, collection, *args):
        for metric_name, metric in self[collection].items():
            metric.reset_state()

    def accumulate(self, collection, **kwargs):
        r"""Accumulates to all metrics in kwargs.

        Parameters
        ----------
        collection : str
            Metrics in this collection will be updated.
        kwargs : dict[str, tuple]
            Names of metrics, and inputs to each metric's accumulate function, e.g. a list of :class:`torch.Tensor`.
        """
        for metric_name, inputs in kwargs.items():
            # Allow multiple inputs to be specified (or one).
            inputs = utils.listify(inputs)

            # If a kwargs dict is specified for this metric.
            if isinstance(inputs[-1], dict):
                inputs, kwinputs = inputs[:-1], inputs[-1]
            else:
                kwinputs = dict()

            self[collection][metric_name].accumulate(*inputs, **kwinputs)

    def result(self, collection='all', *args):
        r"""Gets the result for all metrics in the given collection."""
        results = {}
        for metric_name, metric in self[collection].items():
            results[metric_name] = metric.result(*args)

        return results

    def results_as_json_dict(self, collection='all', prefix=''):
        r"""Gets the result (in a JSON friendly format) for all metrics in the given collection."""
        d = {}
        for name, metric in self[collection].items():
            if not metric.hidden:
                d[prefix + name] = metric.result_as_json()

        return d

    def results_as_str_dict(self, collection='all', prefix=''):
        r"""Gets the result (as a dictionary of strings) for all metrics in the given collection."""
        d = {}
        for name, metric in self[collection].items():
            if not metric.hidden:
                d[prefix + name] = str(metric)

        return d

    def __str__(self):
        r"""Gets the result for all metrics in the given collection, formats this as a single string."""
        d = self.results_as_str_dict('all')
        s = ['{} = {}'.format(name, value) for name, value in d.items()]
        return ' | '.join(s)


class Print(StatefulMetric):
    r"""Class for printing the last reported value.

    Parameters
    ----------
    hidden : bool
        Whether to hide the metric when being summarised by :class:`Handler`.

    Attributes
    ----------
    value
        Most recent accumulated input.
    """
    def __init__(self, hidden=False):
        super(Print, self).__init__(hidden=hidden)

    def reset_state(self, *args):
        self.value = None

    def accumulate(self, tensor):
        self.value = tensor

    def result(self, *args):
        return self.value


class History(StatefulMetric):
    r"""Class for storing the history of any object.

    Parameters
    ----------
    max_len : int
        Maximum length of the history being stored.
    hidden : bool
        Whether to hide the metric when being summarised by :class:`Handler`.

    Attributes
    ----------
    history
        Tensor of (up to `max_len`) previous inputs.
    """
    def __init__(self, max_len=None, hidden=False):
        super(History, self).__init__(hidden=hidden)
        self.max_len = max_len

        self.reset_state()

    def reset_state(self):
        self.history = []

    def accumulate(self, obj):
        self.history.extend(obj)

        # Only save the most recent `self.max_len` tensors.
        if self.max_len is not None:
            self.history = self.history[-self.max_len:]

    def result(self):
        return self.history

    def str_summary(self, result):
        return str(result[-1])

    def result_as_json(self):
        return str(self)

    def __str__(self):
        result = self.result()
        return self.str_summary(result)


class TensorHistory(StatefulMetric):
    r"""Class for storing the history of a tensor.

    Parameters
    ----------
    feat_dim : int
        Dimensionality of each feature vector. If 0, then no axis is used for the feature dimension.
    max_len : int
        Maximum length of the history being stored.
    device : str
        Device (cpu/cuda) to use. If None, this will be inferred from the tensor.
    hidden : bool
        Whether to hide the metric when being summarised by :class:`Handler`.

    Attributes
    ----------
    history
        Tensor of (up to `max_len`) previous inputs.
    """
    def __init__(self, feat_dim, max_len=None, dtype=torch.float32, device=None, hidden=False):
        super(TensorHistory, self).__init__(hidden=hidden)
        self.feat_dim = feat_dim
        self.max_len = max_len

        self.dtype = dtype
        self.device = device
        self.reset_state()

    def reset_state(self):
        if self.feat_dim == 0:
            self.history = torch.empty(0, dtype=self.dtype)
        else:
            self.history = torch.empty((0, self.feat_dim), dtype=self.dtype)

        if self.device is not None:
            self.history = self.history.to(self.device)

    def accumulate(self, tensor, seq_len=None):
        if self.device is None:
            self.device = utils.infer_device(tensor)
            self.history = self.history.to(self.device)

        tensor = tensor.to(self.device)

        if seq_len is None:
            tensor = tensor.reshape(-1, self.feat_dim)
        else:
            # Select only the items with the sequence length for each batch item.
            tensor = utils.batched_masked_select(tensor, seq_len)

        self.history = torch.cat([self.history, tensor])

        # Only save the most recent `self.max_len` tensors.
        if self.max_len is not None:
            self.history = self.history[-self.max_len:]

    def result(self):
        return self.history

    def str_summary(self, result):
        r"""Summarises history of tensors using Gaussian parameters and range."""
        mean = torch.mean(result)
        std = torch.std(result)
        mmin = torch.min(result)
        mmax = torch.max(result)

        if torch.isnan(std):
            std = torch.zeros_like(std)

        return 'N({mean}, {std}) in range [{min}, {max}]'.format(
            mean=utils.format_float_tensor(mean),
            std=utils.format_float_tensor(std),
            min=utils.format_float_tensor(mmin),
            max=utils.format_float_tensor(mmax))

    def result_as_json(self):
        result = self.result()

        if result.numel() == 1:
            return result.item()
        else:
            return self.str_summary(result)

    def __str__(self):
        result = self.result()

        if result.numel() == 1:
            return utils.format_float_tensor(result.item())
        else:
            return self.str_summary(result)


class Mean(StatefulMetric):
    r"""Class for computing the mean in an online fashion.

    Parameters
    ----------
    hidden : bool
        Whether to hide the metric when being summarised by :class:`Handler`.

    Attributes
    ----------
    sum
        Sum of inputs.
    count
        Number of valid frames for all inputs.
    """
    def __init__(self, hidden=False):
        super(Mean, self).__init__(hidden=hidden)
        self.reset_state()

    def reset_state(self):
        self.sum = 0.
        self.count = 0.

    def accumulate(self, tensor, seq_len=None):
        r"""tensor much have shape [batch_size, seq_len, feat_dim]."""
        if seq_len is None:
            self.sum += torch.sum(tensor)
            self.count += tensor.numel()

        else:
            sequence_mask = utils.sequence_mask(seq_len, max_len=tensor.shape[1], dtype=tensor.dtype)
            self.sum += torch.sum(tensor * sequence_mask)
            self.count += torch.sum(sequence_mask).item()

    def result(self, *args):
        return self.sum / (self.count + 1e-8)


class RMSE(Mean):
    r"""Class for computing RMSE in an online fashion.

    Parameters
    ----------
    hidden : bool
        Whether to hide the metric when being summarised by :class:`Handler`.

    Attributes
    ----------
    sum
        Sum of squared difference of `target` and `pred` inputs.
    count
        Number of valid frames for all inputs.
    """
    def __init__(self, hidden=False):
        super(Mean, self).__init__(hidden=hidden)

    def accumulate(self, target, pred, seq_len=None):
        # Accumulate the squared difference.
        square_diff = (target - pred) ** 2
        super(RMSE, self).accumulate(square_diff, seq_len)

    def result(self, *args):
        # Calculate the root mean of the accumulated squared diff.
        return (self.sum / (self.count + 1e-8)) ** 0.5


class F0Distortion(RMSE):
    r"""Class for computing the F0 RMSE in Hz in an online fashion.

    Parameters
    ----------
    hidden : bool
        Whether to hide the metric when being summarised by :class:`Handler`.

    Attributes
    ----------
    sum
        Sum of squared difference of `target` and `pred` inputs.
    count
        Number of valid frames when both `target` and `pred` are voiced.
    """
    def __init__(self, hidden=False):
        super(F0Distortion, self).__init__(hidden=hidden)

    def accumulate(self, f0_target, f0_pred, is_voiced, seq_len=None):
        mask = is_voiced.type(f0_target.dtype)

        if seq_len is not None:
            sequence_mask = utils.sequence_mask(seq_len, max_len=f0_target.shape[1], dtype=f0_target.dtype)
            mask *= sequence_mask

        # Accumulate the squared difference.
        square_diff = (f0_target - f0_pred) ** 2
        self.sum += torch.sum(square_diff * mask)
        self.count += torch.sum(mask).item()


class LF0Distortion(F0Distortion):
    r"""Class for computing the F0 RMSE in Hz in an online fashion.

    Parameters
    ----------
    hidden : bool
        Whether to hide the metric when being summarised by :class:`Handler`.

    Attributes
    ----------
    sum
        Sum of squared difference of `target` and `pred` inputs.
    count
        Number of valid frames when both `target` and `pred` are voiced.
    """
    def __init__(self, hidden=False):
        super(LF0Distortion, self).__init__(hidden=hidden)

    def accumulate(self, lf0_target, lf0_pred, is_voiced, seq_len=None):
        f0_target = torch.exp(lf0_target)
        f0_pred = torch.exp(lf0_pred)

        super(LF0Distortion, self).accumulate(f0_target, f0_pred, is_voiced, seq_len)

