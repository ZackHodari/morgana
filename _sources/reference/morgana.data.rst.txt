data
====

.. currentmodule:: morgana.data

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

