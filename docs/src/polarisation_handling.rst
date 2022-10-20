.. polarisation_handling:

Polarisation handling
=====================

Polarisation handling is intended to implement the Hamaker-Bregman-Sault formalism.

For imaging:

 * Types of polarisation allowed are stokesIQUV, stokesI, linear, circular.
 * These are defined in :py:class:`ska_sdp_datamodels.polarisation_data_models.PolarisationFrame`
 * Images may be defined as stokesI, stokesIQUV, linear, or circular
 * SkyComponents may be defined as stokesI, stokesIQUV, linear, or circular
 * Visibility may be defined as stokesI, stokesIQUV, linear, or circular.
 * Dish/station voltage patterns are described by images in which each pixel is a 2 x 2 complex matrix.
 * For converting different polarisation frames, see functions in polarisation_convert.py.

For calibration, the Jones matrices allowed are:

 * T = scalar phase-only term i.e. complex unit-amplitude phasor times the identity [2,2] matrix
 * G = vector complex gain i.e. diagonal [2, 2] matrix with different phasors
 * B = Same as G but frequency dependent

