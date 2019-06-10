metrics
=======

.. currentmodule:: morgana.metrics


To allow logging of model performance during each epoch of training we provide an interface to compute metrics in an
online (streaming) fashion. This is not only more memory/time efficient than loading all generated files and calculating
performance after an epoch, but it allows for performance to be reported at each batch.

See :class:`StatefulMetric` for details on how to define a new streaming metric.

* `StatefulMetric`_
* `Handler`_

The following are metrics provided by Morgana. See `F0Model
<https://github.com/ZackHodari/morgana/blob/master/models/f0_test_model.py>`_ for example usage.

* `Print`_
* `TensorHistory`_
* `Mean`_
* `RMSE`_
* `F0Distortion`_
* `LF0Distortion`_


StatefulMetric
--------------

.. autoclass:: StatefulMetric
   :members:
   :private-members:
   :show-inheritance:


Handler
-------

This metric is used to maintain multiple collections of metrics, it is created automatically by
:class:`morgana.experiment_builder.ExperimentBuilder` and added to your model instance as an attribute called `metrics`.

.. autoclass:: Handler
   :members:
   :private-members:
   :show-inheritance:


Print
-----

.. autoclass:: Print
   :members:
   :private-members:
   :show-inheritance:


TensorHistory
-------------

.. autoclass:: TensorHistory
   :members:
   :private-members:
   :show-inheritance:


Mean
----

.. autoclass:: Mean
   :members:
   :private-members:
   :show-inheritance:


RMSE
----

.. autoclass:: RMSE
   :members:
   :private-members:
   :show-inheritance:


F0Distortion
------------

.. autoclass:: F0Distortion
   :members:
   :private-members:
   :show-inheritance:


LF0Distortion
-------------

.. autoclass:: LF0Distortion
   :members:
   :private-members:
   :show-inheritance:


