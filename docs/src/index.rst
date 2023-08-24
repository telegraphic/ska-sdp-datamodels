.. _documentation_master:

.. toctree::

SKA SDP Python-based Data Models
################################

This is a `repository <https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels.git>`_
for the Python-based Data Models used in the SKA SDP. The aim of this repository is to
provide a set of data models involved in radio astronomy visibility processing.
The models are specifically meant to facilitate passing data between services
and processing components within the SDP.

Eventually this should cover:

- In-memory communication within the same process, both between Python software as
  well as Python and C++ software (such as `ska-sdp-func <https://gitlab.com/ska-telescope/sdp/ska-sdp-func>`_)

- In-memory communication between different processes, such as via shared memory
  (e.g. as done using Apache Plasma in real-time processing)

- Network communication between different processes for the purpose of distributed computing
  (e.g. via Dask or Kafka)

- Communication via storage, both internal to the SDP as well as for delivery of
  final data products (i.e. conversion into standard established data formats).

The package also provides tools for easy access and manipulation (especially slicing
and reordering) of the data, including providing tools for making the data
self-describing using metadata.

The code is written in Python. The structure is modeled after the
standard data models used in `RASCIL <https://gitlab.com/ska-telescope/external/rascil-main.git>`_.
The interfaces all operate with familiar data structures such as image,
visibility table, gain table, etc. The python source code is directly accessible from these documentation pages:
see the source link in the top right corner.

The data classes are built on the `xarray <https://docs.xarray.dev/en/stable/#>`_ library, offering a
rich API for applications. For more details including how to update existing scripts, see
:ref:`Use of xarray <xarray_doc>`.


Installation Instructions
=========================

The package is installable via pip.

If you would like to view the source code or install from git, use::

    git clone https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels.git

Please ensure you have all the dependency packages installed. The installation is managed
through `poetry <https://python-poetry.org/docs/>`_.
Refer to their page for instructions.

Currently, the package supports Python 3.10 and above.

.. toctree::
   :maxdepth: 1
   :caption: Sections

   data_structure
   polarisation_handling
   xarray
   helper_functions
   api
