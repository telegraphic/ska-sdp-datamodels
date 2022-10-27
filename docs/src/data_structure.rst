.. _data_structure:

Data containers used in ska-sdp-datamodels
===========================================

ska-sdp-datamodels holds data in python Classes. The bulk data and attributes are usually kept in a xarray.Dataset.
For each xarray based class there is an accessor which holds class specific methods and properties.

The structure follows the "Memory Data Models" element of the
`Processing Functions Module view <https://confluence.skatelescope.org/pages/viewpage.action?pageId=161359520>`_.

Visibility-related models
-------------------------

* Baseline-based visibility table, shape (ntimes, nbaselines, nchan, npol), length ntime): :py:class:`ska_sdp_datamodels.visibility.Visibility`
* FlagTable for flagging information: :py:class:`ska_sdp_datamodels.visibility.FlagTable`

Image model
-----------

* Image (data and WCS header): :py:class:`ska_sdp_datamodels.image.Image`

Calibration-related models
--------------------------

* GainTable for gain solutions (as e.g. output from solve_gaintable): :py:class:`ska_sdp_datamodels.calibration.GainTable`
* PointingTable for pointing information: :py:class:`ska_sdp_datamodels.calibration.PointingTable`

Sky-related models
------------------

* SkyComponent (data for a point source or a Gaussian source): :py:class:`ska_sdp_datamodels.sky_model.SkyComponent`
* SkyModel (collection of SkyComponents and Images): :py:class:`ska_sdp_datamodels.sky_model.SkyModel`

Gridded visibility-related models
---------------------------------

* GridData: :py:class:`ska_sdp_datamodels.gridded_visibility.GridData`
* ConvolutionFunction: :py:class:`ska_sdp_datamodels.gridded_visibility.ConvolutionFunction`

Configuration model
-------------------

* Telescope Configuration: :py:class:`ska_sdp_datamodels.configuration.Configuration`

Note that the package contains a set of example configuration files, which can be used
for testing purposes. These are in `ska_sdp_datamodels/configuration/example_antenna_files`.

Science data models
-------------------

* ReceptorFrame: :py:class:`ska_sdp_datamodels.science_data_model.ReceptorFrame`
* PolarisationFrame: :py:class:`ska_sdp_datamodels.science_data_model.PolarisationFrame`
* QualityAssessment: :py:class:`ska_sdp_datamodels.science_data_model.QualityAssessment`
