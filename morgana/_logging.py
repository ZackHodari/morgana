import logging
import os
import sys
import time

from tqdm import tqdm


def create_logger(experiment_dir):
    r"""Writes `stdout` and `stderr` to their stream and to files. tqdm progress bars are written to a separate file."""
    curr_time = time.strftime('%y_%m_%d-%H_%M_%S')
    base_dir = os.path.join(experiment_dir, 'log')
    os.makedirs(base_dir, exist_ok=True)

    logger = logging.getLogger('morgana')
    logger.setLevel(logging.DEBUG)

    format_str = '{asctime} – {levelname:8s} – {module}.{funcName}:{lineno} – {message}'
    formatter = logging.Formatter(format_str, style='{')

    # Redirect tqdm's debug output to a separate log file.
    tqdm_file = logging.FileHandler(os.path.join(base_dir, '{}.tqdm'.format(curr_time)))
    tqdm_file.setFormatter(logging.Formatter('{asctime} – {message}', style='{'))
    tqdm_file.addFilter(IsTqdmFilter(include_tqdm=True))
    logger.addHandler(tqdm_file)

    # DEBUG <= level < ERROR redirected to stdout.
    stdout_stream = logging.StreamHandler(sys.stdout)
    stdout_stream.setLevel(logging.DEBUG)
    stdout_stream.setFormatter(formatter)
    stdout_stream.addFilter(IsTqdmFilter())
    stdout_stream.addFilter(LessThanLevelFilter(level=logging.ERROR))
    logger.addHandler(stdout_stream)

    # DEBUG and higher redirected to file.
    stdout_file = logging.FileHandler(os.path.join(base_dir, '{}.stdout'.format(curr_time)))
    stdout_file.setLevel(logging.DEBUG)
    stdout_file.setFormatter(formatter)
    stdout_file.addFilter(IsTqdmFilter())
    logger.addHandler(stdout_file)

    # ERROR and higher redirected to stderr.
    stderr_stream = logging.StreamHandler(sys.stderr)
    stderr_stream.setLevel(logging.ERROR)
    stderr_stream.setFormatter(formatter)
    stderr_stream.addFilter(IsTqdmFilter())
    logger.addHandler(stderr_stream)

    # ERROR and higher redirected to file.
    stderr_file = logging.FileHandler(os.path.join(base_dir, '{}.stderr'.format(curr_time)))
    stderr_file.setLevel(logging.ERROR)
    stderr_file.setFormatter(formatter)
    stderr_file.addFilter(IsTqdmFilter())
    logger.addHandler(stderr_file)

    return logger


class ProgressBar(tqdm):
    r"""Custom tqdm progress bar with additional :func:`~print` method for logging metrics to the progress bar."""
    def __init__(self, num_batches):
        bar_format = '{desc}{percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
        super(ProgressBar, self).__init__(range(num_batches), bar_format=bar_format)

        self.logger = logging.getLogger('morgana')

    def print(self, mode, epoch, **kwargs):
        r"""Format metric outputs as part of tqdm description, and log to a file."""
        desc = '{mode} | epoch {epoch: >2}'.format(mode=mode, epoch=epoch)
        if kwargs:
            desc += ': ' + ' | '.join('{key} = {value}'.format(key=k, value=v) for k, v in kwargs.items())

        self.logger.debug(desc, {'is_tqdm': True})  # allows `logging.Filter` to recognise these messages.
        self.set_description(desc)


class IsTqdmFilter(logging.Filter):
    r"""Allows only tqdm messages, or disallows all tqdm messages."""
    def __init__(self, name='', include_tqdm=False):
        super(IsTqdmFilter, self).__init__(name=name)
        self.include_tqdm = include_tqdm

    def filter(self, record):
        include_record = super(IsTqdmFilter, self).filter(record)

        record_has_kwargs = record.args and not isinstance(record.args, tuple)
        is_tqdm_message = record.args.get('is_tqdm', False) if record_has_kwargs else False

        if self.include_tqdm:
            # Allow only tqdm messages.
            return include_record and is_tqdm_message
        else:
            # Disallow all tqdm messages.
            return include_record and not is_tqdm_message


class LessThanLevelFilter(logging.Filter):
    r"""Only allow messages less than a given level."""
    def __init__(self, name='', level=logging.NOTSET):
        super(LessThanLevelFilter, self).__init__(name=name)
        self.level = level

    def filter(self, record):
        include_record = super(LessThanLevelFilter, self).filter(record)
        not_above_level = record.levelno < self.level

        return include_record and not_above_level

