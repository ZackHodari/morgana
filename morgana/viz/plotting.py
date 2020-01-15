from collections import OrderedDict
import logging
import os

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

from tts_data_tools import file_io

from morgana import utils


logger = logging.getLogger('morgana')


def load_experiment_results(experiment_name, metric_names='loss', mode='train', experiments_base='experiments'):
    r"""Loads metrics from an experiment.

    Returns
    -------
    results : dict[str, collections.OrderedDict[int, float]]
        Dictionary of results with the following structure,

        .. code:: python

            {
                metric_name: OrderedDict(
                    epoch: metric_value
                )
            }
    """
    metric_names = utils.listify(metric_names)
    results = {metric_name: {} for metric_name in metric_names}

    model_path = os.path.join(experiments_base, experiment_name, mode)
    for epoch_str in os.listdir(model_path):

        metric_path = os.path.join(model_path, epoch_str, 'metrics.json')
        if os.path.isfile(metric_path):

            # Load metrics for this epoch from file.
            metrics = file_io.load_json(metric_path)

            epoch = int(epoch_str.split('_')[-1])
            for metric_name in metric_names:
                if metric_name in metrics:
                    results[metric_name][epoch] = metrics[metric_name]

    # Sort the keys of each metric by the epochs, as `os.listdir` will not use the correct numerical order.
    results = {
        metric_name: OrderedDict(sorted(result.items()))
        for metric_name, result in results.items()
    }

    return results


def plot_experiment(experiment_name, metric_names='loss', experiments_base='experiments',
                    axs=None, colour=None, add_labels=True, save=False):
    metric_names = utils.listify(metric_names)
    results_train = load_experiment_results(experiment_name, metric_names, 'train', experiments_base)
    results_valid = load_experiment_results(experiment_name, metric_names, 'valid', experiments_base)

    if axs is None:
        n_axes = len(metric_names)
        fig, axs = plt.subplots(1, n_axes, figsize=(1 + n_axes * (4 + 1), 4))
        if len(metric_names) == 1:
            axs = [axs]

    for ax, metric_name in zip(axs, metric_names):
        metric_values_train = results_train[metric_name]
        ax.plot(list(metric_values_train.keys()), list(metric_values_train.values()), label=experiment_name, c=colour)

        metric_values_valid = results_valid[metric_name]
        ax.plot(list(metric_values_valid.keys()), list(metric_values_valid.values()), '--', c=colour)

        if add_labels:
            ax.set_xlabel('Epoch number')
            ax.set_ylabel(metric_name)

    if save:
        save_path = os.path.join(experiments_base, experiment_name, 'metrics.pdf')

        logger.info('Saving plot of metrics to {}'.format(save_path))
        plt.savefig(save_path, bbox_inches='tight')

    return axs


def plot_experiment_set(experiment_names, metric_names='loss', experiments_base='experiments', file_name=None):
    experiment_names = utils.listify(experiment_names)
    metric_names = utils.listify(metric_names)

    n_axes = len(metric_names)
    fig, axs = plt.subplots(1, n_axes, figsize=(1 + n_axes * (4 + 1), 4))
    if len(metric_names) == 1:
        axs = [axs]
    cmap = plt.get_cmap('Set1')

    for i, experiment_name in enumerate(experiment_names):
        colour = cmap(float(i) / len(experiment_names))
        plot_experiment(experiment_name, metric_names, experiments_base,
                        axs=axs, colour=colour, add_labels=i == 0)

    handles, labels = axs[0].get_legend_handles_labels()
    extra = Rectangle((0, 0), 1, 1, fc='w', fill=False, edgecolor='none', linewidth=0)
    lgd_pos = ((0.5 + 0.1) * n_axes - 0.1, -0.1)  # 0.5 per subplot plus 0.1 between each subplot.
    lgd = axs[0].legend([extra] + handles, ['solid = train, dotted = valid'] + labels, loc='upper center',
                        bbox_to_anchor=lgd_pos, fancybox=True, shadow=True, ncol=min(3, len(experiment_names)+1))

    plt.setp(lgd.get_lines(), linewidth=4.)

    if file_name:
        os.makedirs(os.path.join('plots', os.path.dirname(file_name)), exist_ok=True)
        save_path = os.path.join('plots', file_name)

        logger.info('Saving plot of metrics for multiple experiments to {}'.format(save_path))
        plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)
