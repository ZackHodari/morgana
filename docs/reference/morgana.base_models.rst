base_models
===========

.. currentmodule:: morgana.base_models


You can defined your model by implementing the abstract methods, :func:`BaseModel.train_data_sources`,
:func:`BaseModel.predict`, and :func:`BaseModel.loss`. See :ref:`Defining a model` for instructions. Currently, there
are three base models that you can inherit from,

* `BaseModel`_
* `BaseSPSS`_
* `BaseVAE`_


BaseModel
---------

.. autoclass:: BaseModel
   :members:
   :private-members:
   :show-inheritance:


BaseSPSS
--------

.. autoclass:: BaseSPSS
   :members:
   :private-members:
   :show-inheritance:


BaseVAE
-------

.. autoclass:: BaseVAE
   :members:
   :private-members:
   :show-inheritance:

