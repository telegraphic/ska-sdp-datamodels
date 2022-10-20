.. _data_structure:

.. toctree::
   :maxdepth: 2

Data containers used in ska-sdp-datamodels
==============================

ska-sdp-datamodels holds data in python Classes. The bulk data and attributes are usually kept in a xarray.Dataset.
For each xarray based class there is an accessor which holds class specific methods and properties.

See :py:mod:`ska_sdp_datamodels.memory_data_models` for the following definitions:

* Image (data and WCS header): :py:class:`ska_sdp_datamodels.memory_data_models.Image`
* SkyComponent (data for a point source or a Gaussian source): :py:class:`ska_sdp_datamodels.memory_data_models.SkyComponent`
* SkyModel (collection of SkyComponents and Images): :py:class:`ska_sdp_datamodels.memory_data_models.SkyModel`
* Baseline-based visibility table, shape (ntimes, nbaselines, nchan, npol), length ntime): :py:class:`ska_sdp_datamodels.memory_data_models.Visibility`
* Telescope Configuration: :py:class:`ska_sdp_datamodels.memory_data_models.Configuration`
* GainTable for gain solutions (as e.g. output from solve_gaintable): :py:class:`ska_sdp_datamodels.memory_data_models.GainTable`
* PointingTable for pointing information: :py:class:`ska_sdp_datamodels.memory_data_models.PointingTable`
* FlagTable for flagging information: :py:class:`ska_sdp_datamodels.memory_data_models.FlagTable`

Polarisation-specific data models are introduced in :py:mod:`ska_sdp_datamodels.polarisation_data_models`
* ReceptorFrame: :py:class:`ska_sdp_datamodels.polarisation_data_models.ReceptorFrame`
* PolarisationFrame: :py:class:`ska_sdp_datamodels.polarisation_data_models.PolarisationFrame`
