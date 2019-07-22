import argparse
import json
import os
import shutil
import sys

from tensorboardX import SummaryWriter
import torch
from torch.optim import Adam
from tqdm import tqdm

from tts_data_tools import file_io

from morgana import data
from morgana import lr_schedules
from morgana import utils
from morgana import viz
from morgana import _logging


def add_boolean_arg(parser, name, help):
    r"""Adds two arguments (one with a \"no-" prefix), allowing for a positive or a negative boolean argument."""
    parser.add_argument("--{}".format(name), dest=name, action="store_true", default=True, help=help)
    parser.add_argument("--no-{}".format(name), dest=name, action="store_false", help=argparse.SUPPRESS)


class DictAction(argparse.Action):
    r"""Uses `eval` to convert a string representation of a dict into a dict instance (including types of values)."""
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, eval(values))


class ExperimentBuilder(object):
    r"""Interface for running training, validation, and generation. Works as glue for machine learning support code.

    Parameters
    ----------
    model_class : morgana.base_models.BaseModel
        Model to be initialised by the experiment builder. Must contain implementations of all abstract methods.
    experiment_name : str
        Name of the experiment, this is used as the directory to save all output (under `experiments_base` directory).
        This is the only command line argument that has no default value and is required.
    kwargs : dict[str, \*]
        Command line arguments. See :func:`~add_args` for all options.

    Attributes
    ----------
    experiment_dir : str
        Directory path to save all output to.
    model : morgana.base_models.BaseModel
        Model instance.
    ema : morgana.utils.ExponentialMovingAverage (if `ema_decay` is not 0.)
        Helper for updating a second model instance with an exponential moving average of the parameters.
    epoch : int
        Current epoch of the model (starting from 1).
    device : str or `torch.device`
        Name of the device to place the mode and parameters on.
    _lr_schedule : `torch.optim.lr_scheduler._LRScheduler`
        Partially initialised learning rate schedule. Depending on the schedule, this will be used per epoch or batch.
    logger : _logging.Logger
        Python Logger. Copies `stdout`, `stderr`, and all tqdm output to separate files.

    train_loader : :class:`torch.utils.data.DataLoader` (in a :class:`data._DataLoaderWrapper` container).
        Attribute is only present if `self.train` is `True`.
        Produces batches from the training data (on a given device) when used as an iterator.
    valid_loader : :class:`torch.utils.data.DataLoader` (in a :class:`data._DataLoaderWrapper` container).
        Attribute is only present if `self.valid` is `True`.
        Produces batches from the validation data (on a given device) when used as an iterator.
    test_loader : :class:`torch.utils.data.DataLoader` (in a :class:`data._DataLoaderWrapper` container).
        Attribute is only present if `self.test` is `True`.
        Produces batches from the testing data (on a given device) when used as an iterator.

    Notes
    -----
    All arguments provided as command line arguments (including `experiment_name`) are saved as instance attributes.
    """

    @classmethod
    def get_experiment_args(cls):
        r"""Creates a command line argument parser and returns the dictionary of arguments."""
        parser = argparse.ArgumentParser(description="Experiment builder for TTS model training and generation.")
        cls.add_args(parser)
        args = parser.parse_args()

        return vars(args)

    @classmethod
    def add_args(cls, parser):
        r"""Adds command line arguments to a parser, see `usage <command_line_arguments.html>`_."""
        parser.add_argument("--model_kwargs",
                            dest="model_kwargs", action=DictAction, type=str, default={},
                            help="Settings for the model, a Python dictionary written in quotes.")

        # Training options
        add_boolean_arg(parser, "train", help="If True, model will be trained for --num_epochs on --train_id_list.")
        add_boolean_arg(parser, "valid", help="If True, model will be evaluated on --valid_id_list every epoch.")
        parser.add_argument("--test",
                            dest="test", action="store_true", default=False,
                            help="If True, generation for --test_id_list will be performed after training.")

        parser.add_argument("--start_epoch",
                            dest="start_epoch", action="store", type=int, default=1,
                            help="The epoch number to start training at (will effect checkpoint saves).")
        parser.add_argument("--end_epoch",
                            dest="end_epoch", action="store", type=int, default=50,
                            help="Epoch to end training at.")
        parser.add_argument("--checkpoint_path",
                            dest="checkpoint_path", action="store", type=str, default=None,
                            help="If specified, the model will first load parameters from an existing checkpoint.")
        parser.add_argument("--ema_checkpoint_path",
                            dest="ema_checkpoint_path", action="store", type=str, default=None,
                            help="If specified, the EMA model will first load parameters from an existing checkpoint.")

        parser.add_argument("--batch_size",
                            dest="batch_size", action="store", type=int, default=32,
                            help="Batch size used for iteration over train/valid data.")
        parser.add_argument("--learning_rate",
                            dest="learning_rate", action="store", type=float, default=0.01,
                            help="Learning rate for Adam optimiser to use during training.")
        parser.add_argument("--lr_schedule_name",
                            dest="lr_schedule_name", action="store", type=str, default='constant',
                            help="Learning rate schedule to use during training.")
        parser.add_argument("--lr_schedule_kwargs",
                            dest="lr_schedule_kwargs", action=DictAction, type=str, default={},
                            help="Settings for learning rate schedule, a Python dictionary written in quotes.")
        parser.add_argument("--weight_decay",
                            dest="weight_decay", action="store", type=float, default=0.,
                            help="L2 regularisation weight, default of 0 indication no L2 loss term.")
        parser.add_argument("--ema_decay",
                            dest="ema_decay", action="store", type=float, default=0.,
                            help="If not 0, track exponential moving average of model parameters, used for generation.")

        parser.add_argument("--num_data_threads",
                            dest="num_data_threads", action="store", type=int, default=1,
                            help="Number of threads used to load the data with.")

        parser.add_argument("--model_checkpoint_interval",
                            dest="model_checkpoint_interval", action="store", type=int, default=1,
                            help="The number of epochs to wait between saving the model.")
        parser.add_argument("--train_output_interval",
                            dest="train_output_interval", action="store", type=int, default=10,
                            help="The number of epochs to wait between generating output for training data.")
        parser.add_argument("--valid_output_interval",
                            dest="valid_output_interval", action="store", type=int, default=10,
                            help="The number of epochs to wait between generating output for validation data.")
        parser.add_argument("--test_output_interval",
                            dest="test_output_interval", action="store", type=int, default=10,
                            help="The number of epochs to wait between generating output for test data.")

        # Paths for data, and output
        parser.add_argument("--data_root",
                            dest="data_root", action="store", type=str, default='data',
                            help="Base directory containing all data.")

        parser.add_argument("--train_dir",
                            dest="train_dir", action="store", type=str, default='train',
                            help="Name of the sub-directory in --data_root containing training data.")
        parser.add_argument("--valid_dir",
                            dest="valid_dir", action="store", type=str, default='valid',
                            help="Name of the sub-directory in --data_root containing validation data.")
        parser.add_argument("--test_dir",
                            dest="test_dir", action="store", type=str, default='test',
                            help="Name of the sub-directory in --data_root containing test data.")

        parser.add_argument("--train_id_list",
                            dest="train_id_list", action="store", type=str, default='train_file_id_list.scp',
                            help="File name in --train_dir containing basenames of training samples.")
        parser.add_argument("--valid_id_list",
                            dest="valid_id_list", action="store", type=str, default='valid_file_id_list.scp',
                            help="File name in --valid_dir containing basenames of validation samples.")
        parser.add_argument("--test_id_list",
                            dest="test_id_list", action="store", type=str, default='test_file_id_list.scp',
                            help="File name in --test_dir containing basenames of test files.")

        parser.add_argument("--normalisation_dir",
                            dest="normalisation_dir", action="store", type=str, default='train',
                            help="Name of the sub-directory in --data_root containing normalisation data.")

        parser.add_argument("--experiments_base",
                            dest="experiments_base", action="store", type=str, default='experiments',
                            help="Base directory where all experiments direct their output.")
        parser.add_argument("--experiment_name",
                            dest="experiment_name", action="store", type=str, required=True,
                            help="Name of the sub-directory in --output_dir used for any output.")

        # Synthesis options
        parser.add_argument("--sample_rate",
                            dest="sample_rate", action="store", type=str, default=16000,
                            help="Sample rate of the waveforms generated.")

    def __init__(self, model_class, experiment_name, **kwargs):
        self.model_class = model_class
        self.model_kwargs = kwargs['model_kwargs']
        self.experiment_name = experiment_name

        self.train = kwargs['train']
        self.valid = kwargs['valid']
        self.test = kwargs['test']

        self.start_epoch = kwargs['start_epoch']
        self.end_epoch = kwargs['end_epoch']
        self.checkpoint_path = kwargs['checkpoint_path']
        self.ema_checkpoint_path = kwargs['ema_checkpoint_path']

        self.batch_size = kwargs['batch_size']
        self.learning_rate = kwargs['learning_rate']
        self.lr_schedule_name = kwargs['lr_schedule_name']
        self.lr_schedule_kwargs = kwargs['lr_schedule_kwargs']
        self.weight_decay = kwargs['weight_decay']
        self.ema_decay = kwargs['ema_decay']

        self.num_data_threads = kwargs['num_data_threads']

        self.model_checkpoint_interval = kwargs['model_checkpoint_interval']
        self.train_output_interval = kwargs['train_output_interval']
        self.valid_output_interval = kwargs['valid_output_interval']
        self.test_output_interval = kwargs['test_output_interval']

        self.data_root = kwargs['data_root']
        self.train_dir = kwargs['train_dir']
        self.valid_dir = kwargs['valid_dir']
        self.test_dir = kwargs['test_dir']
        self.train_id_list = kwargs['train_id_list']
        self.valid_id_list = kwargs['valid_id_list']
        self.test_id_list = kwargs['test_id_list']

        self.normalisation_dir = kwargs['normalisation_dir']

        self.experiments_base = kwargs['experiments_base']

        self.sample_rate = kwargs['sample_rate']

        #
        # Add/modify settings and attributes.
        #

        self.experiment_dir = os.path.join(self.experiments_base, self.experiment_name)
        self.logger = _logging.create_logger(self.experiment_dir)

        self._lr_schedule = lr_schedules.init_lr_schedule(self.lr_schedule_name, **self.lr_schedule_kwargs)

        if self.ema_checkpoint_path is None:
            self.ema_checkpoint_path = self.checkpoint_path

        #
        # Check settings have no "conflicts".
        #

        self.resolve_setting_conflicts()

        #
        # Finish setup of model and data, ready for procedures to be run.
        #

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.logger.info('Using device {}'.format(self.device))

        # Create the model, loading from a checkpoint if specified and providing the normalisers to the model instance.
        self.model = self.build_model(self.model_class, self.model_kwargs, checkpoint_path=self.checkpoint_path)

        # Create normalisers using the feature specification given by the model.
        train_data_sources = self.model.train_data_sources()
        normalisers = data.Normalisers(train_data_sources, self.normalisation_dir, self.data_root, self.device)
        self.model.normalisers = normalisers

        # Create a duplicate model for EMA-based generation, load parameters for this if necessary.
        if self.ema_decay:
            averaged_model = self.build_model(
                self.model_class, self.model_kwargs, checkpoint_path=self.ema_checkpoint_path)
            averaged_model.normalisers = normalisers

            self.ema = utils.ExponentialMovingAverage(model=averaged_model, decay=self.ema_decay)

        # Prepare the data by collating the data sources - pads, batches, and sends their outputs to the chosen device.
        if self.train:
            self.train_loader = self.load_data(
                train_data_sources, self.train_dir, self.train_id_list, normalisers, name='train')
        if self.valid:
            valid_data_sources = self.model.valid_data_sources()
            self.valid_loader = self.load_data(
                valid_data_sources, self.valid_dir, self.valid_id_list, normalisers, name='valid', shuffle=False)
        if self.test:
            test_data_sources = self.model.test_data_sources()
            self.test_loader = self.load_data(
                test_data_sources, self.test_dir, self.test_id_list, normalisers, name='test', shuffle=False)

        # Print and log the configuration provided.
        self.log_initial_setup(experiment_name=experiment_name, **kwargs)

        # Create a TensorboardX SummaryWriter.
        self.model.tensorboard = SummaryWriter(self.experiment_dir)

    def log_initial_setup(self, **kwargs):
        r"""Copies model definition if the experiment is new. Logs the model summary and config options."""
        # If this experiment has not been run before, then copy the model's file into experiment_dir.
        if not os.path.exists(self.experiment_dir):
            # Check that the code was run from a file (as opposed to an interpreter.
            if hasattr(sys.modules['__main__'], '__file__'):
                # The file from which ExperimentBuilder was called.
                model_class_file = sys.modules['__main__'].__file__
                os.makedirs(self.experiment_dir)
                shutil.copy2(model_class_file, self.experiment_dir)

        else:
            # The model has been run before, ideally if this is the case the file that was run should be the .py file in
            # experiment_dir. However, this is not a requirement; note that if a different .py file is used the model
            # may not load correctly (or at all), or it may cause the .py file in experiment_dir to become incompatible.
            pass

        self.logger.info('\n\n{}\n\n'.format(self.model))
        with open(os.path.join(self.experiment_dir, 'model_summary.txt'), 'w') as f:
            f.write(str(self.model))

        self.logger.info('\n\n{}\n\n'.format(json.dumps(kwargs, indent=4)))
        with open(os.path.join(self.experiment_dir, 'config.json'), 'w') as f:
            json.dump(kwargs, f, indent=4)

    def resolve_setting_conflicts(self):
        r"""Checks settings and modify any that are inconsistent. Errors for incorrect setting should be raised here.

        If a checkpoint file is given, and :class:`torch.optim.lr_scheduler.ReduceLROnPlateau` (early stopping) learning
        rate schedule is being used validation will be forced to be turned on.

        If training is off, the epoch number will be extracted from the checkpoint file and set as the current epoch.

        Raises
        -------
        ValueError
            If no procedures are specified (e.g. training, validation, or testing).
        ValueError
            If a checkpoint file is given and `self.start_epoch` is less than the epoch of this file. This may be overly
            restrictive, but it will catch some off by one errors. Can be avoided by renaming the checkpoint file.
        ValueError
            If no checkpoint file is given and training is turned off.
        """
        # Ensure some procedure is specified.
        if not (self.train or self.valid or self.test):
            raise ValueError('No process specified, use --train, --valid, or --test.')

        # If we are training from a checkpoint, make sure no checkpoints will be overwritten.
        if self.train:
            if self.checkpoint_path:
                checkpoint_epoch = utils.get_epoch_from_checkpoint_path(self.checkpoint_path)

                if self.start_epoch <= checkpoint_epoch:
                    raise ValueError(
                        'Warning: --start_epoch is less than or equal to --args.checkpoint_path, this may cause '
                        'checkpoints to be overwritten. Either rename the checkpoint or increase start_epoch,\n'
                        '\tcheckpoint_path: {path}\n'
                        '\tstart_epoch\t <= checkpoint_epoch\n'
                        '\t{s_epoch}\t\t <= {c_epoch}'
                        .format(path=self.checkpoint_path, s_epoch=self.start_epoch, c_epoch=checkpoint_epoch))

            # If our LR schedule is based on validation performance, then ensure validation is run.
            if self.lr_schedule_name == 'plateau':
                self.valid = True

        # If we are not training, then a checkpoint must be specified.
        if (not self.train) and (self.valid or self.test):
            if self.checkpoint_path:
                # Since we are not training, we must override the epoch counter for use in validation or testing.
                self.epoch = utils.get_epoch_from_checkpoint_path(self.checkpoint_path)
            else:
                raise ValueError('If we are performing evaluations without training a checkpoint must be specified '
                                 'using --load_from_epoch.')

    def build_model(self, model_class, model_kwargs, checkpoint_path=None):
        r"""Creates model instance. Loads parameters from a checkpoint file, if given. Moves the model to the device."""
        model = model_class(**model_kwargs)

        if checkpoint_path:
            self.logger.info('Loading model checkpoint from\n'
                             '\t{path}\n'.format(path=checkpoint_path))
            model.load_parameters(checkpoint_path, device=self.device)

        model.to(self.device)
        return model

    def load_data(self, data_sources, data_dir, id_list, normalisers=None, name='', shuffle=True):
        r"""Creates a dataset using the data sources and bathces this as a data loader.

        Parameters
        ----------
        data_sources : dict[str, _DataSource]
            Specification of the different data to be loaded.
        data_dir : str
            The directory containing all data for this dataset split.
        id_list : str
            The name of the file id-list containing base names to load, contained withing `self.data_root`.
        normalisers : None or dict[str, _FeatureNormaliser]
            Normalisers to be passed to the :class:`morgana.data._DataSource` instances.
        name : str
            An identifier used for logging.
        shuffle : bool
            Whether to shuffle the data every epoch.

        Returns
        -------
        torch.utils.data.DataLoader (in a :class:`morgana.data._DataLoaderWrapper` container)
            An instance with the `__iter__` method, allowing for iteration over batches of the dataset.
        """
        self.logger.info('Loading {name} data using {id_list} from\n'
                         '\t{root}/{dir}'.format(name=name, id_list=id_list, root=self.data_root, dir=data_dir))

        dataset = data.FilesDataset(data_sources, data_dir, id_list, normalisers, self.data_root)

        data_loader = data.batch(dataset, batch_size=self.batch_size, shuffle=shuffle,
                                 num_data_threads=self.num_data_threads, device=self.device)

        return data_loader

    def train_epoch(self, data_loader, optimizer, lr_schedule=None, gen_output=False, out_dir=None):
        r"""Trains the model once on all batches in the data loader given.

        * Gradient updates, and EMA gradient updates.
        * Batch level learning rate schedule updates.
        * Logging metrics to tqdm and to a `metrics.json` file.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader (in a :class:`data._DataLoaderWrapper` container)
            An instance with the `__iter__` method, allowing for iteration over batches of the dataset.
        optimizer : torch.optim.Optimizer
        lr_schedule : torch.optim.lr_scheduler._LRScheduler
            Learning rate schedule, only used if it is a member of `morgana.lr_schedules.BATCH_LR_SCHEDULES`.
        gen_output : bool
            Whether to generate output for this training epoch. Output is defined by
            :func:`morgana.base_models.BaseModel.analysis_for_train_batch`.
        out_dir : str
            Directory used to save output (changes for each epoch).

        Returns
        -------
        loss : float
            Average loss for entire batch.
        """
        self.model.mode = 'train'
        self.model.metrics.reset_state('train')

        loss = 0.0
        pbar = _logging.ProgressBar(len(data_loader))
        for i, (features, names) in zip(pbar, data_loader):
            self.model.step = (self.epoch - 1) * len(data_loader) + i + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            batch_loss, output_features = self.model(features)

            batch_loss.backward()
            optimizer.step()

            # Update the learning rate.
            if lr_schedule is not None and self.lr_schedule_name in lr_schedules.BATCH_LR_SCHEDULES:
                lr_schedule.step()

            loss += batch_loss.item()

            # Update the exponential moving average model if it exists.
            if self.ema_decay:
                self.ema.update_params(self.model)

            # Log metrics.
            pbar.print('train', self.epoch,
                       batch_loss=tqdm.format_num(batch_loss),
                       **self.model.metrics.results_as_str_dict('train'))

            if gen_output:
                self.model.analysis_for_train_batch(features, output_features, names,
                                                    out_dir=out_dir, sample_rate=self.sample_rate)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            file_io.save_json(self.model.metrics.results_as_json_dict('train'),
                              os.path.join(out_dir, 'metrics.json'))

        self.model.mode = ''

        return loss / (i + 1)

    def run_train(self):
        r"""Runs training from `start_epoch` to `end_epoch`.

        * Parameter checkpointing, and EMA parameter checkpointing
        * Validation and generation.
        * Epoch level learning rate schedule updates
        """
        self.logger.info('epoch {epoch: >2}: Beginning training'.format(epoch=self.start_epoch))

        optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_schedule = self._lr_schedule(optimizer)

        for self.epoch in range(self.start_epoch, self.end_epoch + 1):
            epoch_train_dir = os.path.join(self.experiment_dir, 'train', 'epoch_{}'.format(self.epoch))

            # Train model.
            gen_train_output = self.epoch % self.train_output_interval == 0
            train_loss = self.train_epoch(self.train_loader, optimizer, lr_schedule,
                                          gen_output=gen_train_output, out_dir=epoch_train_dir)

            # Save model.
            if self.epoch % self.model_checkpoint_interval == 0:
                self.logger.info(
                    'epoch {e: >2}: loss {loss:.3f}:  Saving model to\n'
                    '\t{dir}/checkpoints/epoch_{e}.pt'.format(e=self.epoch, loss=train_loss, dir=self.experiment_dir))
                self.model.save_parameters(self.experiment_dir, self.epoch)

                if self.ema_decay:
                    self.logger.info(
                        'epoch {e: >2}:  Saving EMA model to\n'
                        '\t{dir}/checkpoints/epoch_{e}_ema.pt'.format(e=self.epoch, dir=self.experiment_dir))
                    self.ema.model.save_parameters(self.experiment_dir, '{}_ema'.format(self.epoch))

            # Run validation.
            if self.valid:
                gen_valid_output = self.epoch % self.valid_output_interval == 0
                val_loss = self.run_valid(gen_valid_output)

                # Update the learning rate.
                if self.lr_schedule_name == 'plateau':
                    lr_schedule.step(metrics=val_loss)

            # Run test.
            gen_test_output = self.epoch % self.test_output_interval == 0
            if self.test and gen_test_output:
                self.run_test()

            # Update the learning rate.
            if self.lr_schedule_name in lr_schedules.EPOCH_LR_SCHEDULES:
                lr_schedule.step()

    @torch.no_grad()
    def valid_epoch(self, data_loader, model=None, gen_output=False, out_dir=None):
        r"""Evaluates model once on all batches in data loader given. Performs analysis of model output if requested.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader (in a :class:`data._DataLoaderWrapper` container)
            An instance with the `__iter__` method, allowing for iteration over batches of the dataset.
        model : morgana.base_models.BaseModel
            Model instance. If `self.ema_decay` is non-zero this will be the ema model.
        gen_output : bool
            Whether to generate output for this validation epoch. Output is defined by
            :func:`morgana.base_models.BaseModel.analysis_for_valid_batch`.
        out_dir : str
            Directory used to save output (changes for each epoch).

        Returns
        -------
        loss : float
            Average loss for entire batch.
        """
        if model is None:
            model = self.model

        model.mode = 'valid'
        model.metrics.reset_state('valid')

        loss = 0.0
        pbar = _logging.ProgressBar(len(data_loader))
        for i, (features, names) in zip(pbar, data_loader):
            self.model.step = (self.epoch - 1) * len(data_loader) + i + 1

            batch_loss, output_features = model(features)

            loss += batch_loss.item()

            # Log metrics.
            pbar.print('valid', self.epoch,
                       batch_loss=tqdm.format_num(batch_loss),
                       **model.metrics.results_as_str_dict('valid'))

            if gen_output:
                model.analysis_for_valid_batch(features, output_features, names,
                                               out_dir=out_dir, sample_rate=self.sample_rate)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            file_io.save_json(model.metrics.results_as_json_dict('valid'),
                              os.path.join(out_dir, 'metrics.json'))

        model.mode = ''

        return loss / (i + 1)

    def run_valid(self, gen_output):
        r"""Runs evaluation for the current epoch."""
        epoch_valid_dir = os.path.join(self.experiment_dir, 'valid', 'epoch_{}'.format(self.epoch))
        self.logger.info('epoch {e: >2}: Evaluating loaded model on validation set'.format(e=self.epoch))
        if gen_output:
            self.logger.info('\toutput being saved to\n\t{dir}'.format(dir=epoch_valid_dir))

        if self.ema_decay:
            model = self.ema.model
        else:
            model = self.model

        valid_loss = self.valid_epoch(self.valid_loader, model=model, gen_output=gen_output, out_dir=epoch_valid_dir)
        self.logger.info('epoch {e: >2}: valid_loss {loss:.3f}'.format(e=self.epoch, loss=valid_loss))

        return valid_loss

    @torch.no_grad()
    def test_epoch(self, data_loader, model=None, out_dir=None):
        r"""Evaluates the model once on all batches in the data loader given. Performs analysis of model predictions.

        Parameters
        ----------
        data_loader : :class:`torch.utils.data.DataLoader` (in a :class:`morgana.data._DataLoaderWrapper` container)
            An instance with the `__iter__` method, allowing for iteration over batches of the dataset.
        model : morgana.base_models.BaseModel
            Model instance. If `self.ema_decay` is non-zero this will be the ema model.
        out_dir : str
            Directory used to save output (changes for each epoch).
        """
        if model is None:
            model = self.model

        model.mode = 'test'
        model.metrics.reset_state('test')

        pbar = _logging.ProgressBar(len(data_loader))
        for i, (features, names) in zip(pbar, data_loader):
            self.model.step = (self.epoch - 1) * len(data_loader) + i + 1

            output_features = model.predict(features)

            model.analysis_for_test_batch(features, output_features, names,
                                          out_dir=out_dir, sample_rate=self.sample_rate)

            # Log metrics.
            pbar.print('test', self.epoch,
                       **model.metrics.results_as_str_dict('test'))

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            file_io.save_json(model.metrics.results_as_json_dict('test'),
                              os.path.join(out_dir, 'metrics.json'))

        model.mode = ''

    def run_test(self):
        r"""Runs generation for the current epoch."""
        epoch_test_dir = os.path.join(self.experiment_dir, 'test', 'epoch_{}'.format(self.epoch))
        self.logger.info('epoch {e: >2}: Running synthesis for the test set, output being saving to\n'
                         '\t{dir}.'.format(e=self.epoch, dir=epoch_test_dir))

        if self.ema_decay:
            model = self.ema.model
        else:
            model = self.model

        self.test_epoch(self.test_loader, model=model, out_dir=epoch_test_dir)

    def run_experiment(self):
        r"""Runs all procedures requested for the experiment."""
        if self.train:
            try:
                self.run_train()

                if self.valid:
                    viz.plotting.plot_experiment(self.experiment_name, 'loss', self.experiments_base, save=True)

            except KeyboardInterrupt:
                if self.valid:
                    viz.plotting.plot_experiment(self.experiment_name, 'loss', self.experiments_base, save=True)
                raise

        if (not self.train) and self.valid:
            self.run_valid(gen_output=True)

        if (not self.train) and self.test:
            self.run_test()

