.. _documentation_master:

.. toctree::

SKA SDP Python-based Data Models
################################

This is a `repository <https://gitlab.com/ska-telescope/sdp/ska-sdp-datamodels.git>`_
for the Python-based Data Models used in the SKA SDP. The aim of this repository is to
provide a set of universal data models that can be used across various workflows
and pipelines in the SDP architecture.

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

.. toctree::
   :maxdepth: 1
   :caption: Sections

   data_structure
   polarisation_handling
   xarray
   helper_functions
   api
