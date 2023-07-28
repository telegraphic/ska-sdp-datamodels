.. _helper_functions:

Data Model Helper Functions
============================

Xarray Coordinate Support
-------------------------

We have provided coordinate support functions for the WCS coordinate system used in Image, GridData and ConvolutionFunction.
See :py:mod:`ska_sdp_datamodels.xarray_coordinate_support`

Data Models IO Functions
-------------------------

Each data model directory comes with a set of read-write and convert functions. See :ref:`api`.


MsgPack Support
---------------

Two new functions are now available to be able to encode and decode :py:mod:`xarray.Dataset` objects.

These functions are in the :py:mod:`ska_sdp_datamodels.utilities` module.
