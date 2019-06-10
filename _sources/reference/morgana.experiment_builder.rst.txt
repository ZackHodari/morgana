experiment_builder
==================

.. currentmodule:: morgana.experiment_builder


After defining a model (as described in :ref:`base_models`) you can train your model (or generate from a checkpoint)
using an experiment builder, this provides a `Command line interface <command_line_arguments.html>`_ to
train/validation/test loops.

.. toctree::
   :maxdepth: 1

   Command line usage <command_line_arguments>

If you want to define custom train/validation/test loops you can create a subclass of :class:`ExperimentBuilder` and
override the methods you wish to change.


ExperimentBuilder
-----------------

.. autoclass:: ExperimentBuilder
   :members:
   :private-members:
   :show-inheritance:

