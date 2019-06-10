=======
morgana
=======

Morgana (`GitHub <https://github.com/ZackHodari/morgana>`_) (`docs <https://zackhodari.github.io/morgana/>`_) is a
toolkit for defining and training Text-to-Speech voices in PyTorch.


Installation
------------

To install as a package, from source:

.. code-block:: bash

    pip install git+https://github.com/ZackHodari/morgana


To clone as a git repo, allowing for local modifications to the source code:

.. code-block:: bash

    git clone https://github.com/ZackHodari/morgana
    cd morgana
    python setup.py develop


Design
------

The support code necessary for creating Text-to-Speech (TTS) voices is not over complicated, but does require time to
piece together. Existing packages like `Merlin <https://github.com/CSTR-Edinburgh/merlin>`_ provide an easy to use
toolkit for training voices, but Merlin obfuscates too many details, and makes modifications hard without changing the
source code. Similar to `nnmnkwii <https://github.com/r9y9/nnmnkwii>`_, Morgana aims to provide flexibility in defining
your own models. However, Morgana attempts to automate as much of the support code as possible, while still allowing for
customisation where necessary.


Defining a model
----------------

For most use-cases, the classes provided in `base_models.py
<https://github.com/ZackHodari/morgana/blob/master/morgana/base_models.py>`_ is all that needs to be extended. For
example, create a subclass of :class:`morgana.base_models.BaseSPSS` and implement `train_data_sources`, `predict`, and
`loss`.

* `__init__`: Create the layers for your model.
* `train_data_sources`: Define a dictionary containing instances that are subclasses of `morgana.data._DataSource`.
* `predict`: Define any arbitrary computation in PyTorch.
* `loss`: Define the targets (from `features`) and predictions (output of `predict`) used to calculate the loss. A
  sequence length feature can also be specified (for each target-prediction pair).

.. code-block::

    import torch.nn as nn
    from morgana import data
    from morgana import utils
    from morgana.base_models import BaseSPSS
    from morgana.experiment_builder import ExperimentBuilder

    class F0Model(BaseSPSS):
        def __init__(self, normalisers=None):
            super(F0Model, self).__init__(normalisers=normalisers)

            self.layers = nn.Sequential(
                nn.Linear(600, 512),
                nn.Sigmoid(),
                nn.Linear(512, 128),
                nn.Sigmoid(),
                nn.Linear(128, 32),
                nn.Sigmoid(),
                nn.Linear(32, 1)
            )

        @classmethod
        def train_data_sources(cls):
            return {
                'n_frames': data.TextSource('n_frames'),
                'dur': data.TextSource('dur', normalisation='mvn'),
                'lab': data.NumpyBinarySource('lab', normalisation='minmax'),
                'lf0': data.NumpyBinarySource('lf0', normalisation='mvn'),
            }

        def predict(self, features):
            norm_lab_at_frame_rate = utils.upsample_to_repetitions(features['normalised_lab'], features['dur'])
            pred_norm_lf0 = self.layers(norm_lab_at_frame_rate, seq_len=features['n_frames'])
            pred_lf0 = self.normalisers['lf0'].denormalise(pred_norm_lf0)

            return {'pred_norm_lf0': pred_norm_lf0,
                    'pred_lf0': pred_lf0}

        def loss(self, features, output_features):
            target_norm_lf0 = features['normalised_lf0']
            pred_norm_lf0 = output_features['pred_norm_lf0']
            seq_len = features['n_frames']

            return self._loss(target_norm_lf0, pred_norm_lf0, seq_len)


Running an experiment
---------------------

Most models can be run using classes provided in `experiment_builder.py
<https://github.com/ZackHodari/morgana/blob/master/morgana/experiment_builder.py>`_. If different training procedures
are needed, then a new :class:`morgana.experiment_builder.ExperimentBuilder` subclass may be required. An
`ExperimentBuilder` contains the following important methods,

* `add_args`: Defines the command lines arguments supported for experiments of this type.
* `__init__`:
    * Saves command line arguments as instance attributes.
    * Calls `resolve_setting_conflicts`.
    * Loads normalisers and data specified in `train_data_sources`.
    * Creates the model. Loads from a checkpoint. Creates an exponential moving average (EMA) instance of the model.
    * Sets up Python logging, saves stdout and stderr to files. Saves tqdm output to a separate log file.
* `resolve_setting_conflicts`: Check (and modify) any command line arguments that are incorrect (or inconsistent).
* `train_epoch`: Epoch loop that iterates through `ExperimentBuilder.train_iter`.
* `run_train`: Training loop that calls `train_epoch` until `ExperimentBuilder.epoch` reaches
  `ExperimentBuilder.end_epoch`.
* `valid_epoch`: Epoch loop that iterates through `ExperimentBuilder.valid_iter`.
* `run_valid`: Runs validation of the current model (or EMA model), and reports the validation loss.
* `test_epoch`: Epoch loop that iterates through `ExperimentBuilder.test_iter`.
* `run_test`: Runs generation of the current model (or EMA model), no loss will be reported (no labels are given).
* `run_experiment`: Runs `run_train`, `run_valid`, and `run_test` according to the command line arguments.

At the bottom of the file containing `F0Model` place the following,

.. code-block::

    def main():
        args = ExperimentBuilder.get_experiment_args()
        experiment = ExperimentBuilder(F0Model, **args)
        experiment.run_experiment()


    if __name__ == "__main__":
        main()


The model can then be trained using the following command (see `tts_data_tools
<https://github.com/ZackHodari/tts_data_tools>`_ for guidance on pre-processing of data),

.. code-block:: bash

    python acoustic_model.py \
        --experiment_name DNN_voice \
        --data_root ~/data/Blizzard2017 \
        --train_dir train \
        --train_id_list train_file_id_list.scp \
        --valid_dir valid \
        --valid_id_list valid_file_id_list.scp

