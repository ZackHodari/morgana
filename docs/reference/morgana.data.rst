data
====

.. currentmodule:: morgana.data


Specification of what data the model will load is given as part of your (:ref:`base_models`) model
class, this does not specify where the data will be loaded from, but what data (e.g. the given name, the directory name,
normalisation used, delta and delta-delta inclusion, and file extension). The location of the data is given when running
an experiment using the :ref:`Command line arguments`.

If you need to load a file with a custom function create a subclass of :class:`_DataSource` (loading can include
preprocessing if needed).

The following sections describe the provided utilities for loading code in Morgana, however
these are all used internally by :ref:`experiment_builder`, typically the only thing you need to be aware of are the
available data sources and feature normalisers.

* `Batching utility`_
* `DataSource specification`_
* `Dataset to combine DataSource instances`_
* `Feature normalisers`_
* `Wrappers to change existing DataLoader instance`_


Batching utility
----------------

.. autofunction:: batch


DataSource specification
------------------------

.. note::
   Supported `Feature normalisers`_ are limited to `mvn` and `minmax`. To define a new normalisers you should override
   :func:`create_normaliser` for your data source class.

_DataSource
+++++++++++

.. autoclass:: _DataSource
   :members:
   :private-members:
   :special-members:
   :show-inheritance:


NumpyBinarySource
+++++++++++++++++

.. autoclass:: NumpyBinarySource
   :members:
   :private-members:
   :show-inheritance:


TextSource
++++++++++

.. autoclass:: TextSource
   :members:
   :private-members:
   :show-inheritance:


StringSource
++++++++++++

.. autoclass:: StringSource
   :members:
   :private-members:
   :show-inheritance:


WavSource
+++++++++

.. autoclass:: WavSource
   :members:
   :private-members:
   :show-inheritance:


Dataset to combine DataSource instances
---------------------------------------


FilesDataset
++++++++++++

.. autoclass:: FilesDataset
   :members:
   :private-members:
   :show-inheritance:


Feature normalisers
-------------------


_FeatureNormaliser
++++++++++++++++++

.. autoclass:: _FeatureNormaliser
   :members:
   :private-members:
   :show-inheritance:


MeanVaraianceNormaliser
+++++++++++++++++++++++

.. autoclass:: MeanVaraianceNormaliser
   :members:
   :private-members:
   :show-inheritance:


MinMaxNormaliser
++++++++++++++++

.. autoclass:: MinMaxNormaliser
   :members:
   :private-members:
   :show-inheritance:


Wrappers to change existing DataLoader instance
-----------------------------------------------


_DataLoaderWrapper
++++++++++++++++++

.. autoclass:: _DataLoaderWrapper
   :members:
   :private-members:
   :show-inheritance:


ToDeviceWrapper
+++++++++++++++

.. autoclass:: ToDeviceWrapper
   :members:
   :private-members:
   :show-inheritance:

