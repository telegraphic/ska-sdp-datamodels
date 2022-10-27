""" Unit tests for SkyModel class
"""

import pytest
import numpy
import pandas
from xarray import DataArray
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.wcs import FITSFixedWarning

from src.ska_sdp_datamodels.polarisation_data_models import (
    PolarisationFrame,
    ReceptorFrame,
)

from src.ska_sdp_datamodels.image_model import (
    Image
)

from src.ska_sdp_datamodels.memory_data_models import (
    Configuration,
    GainTable,
    SkyModel
)
from src.ska_sdp_datamodels.xarray_coordinate_support import (
    conv_func_wcs,
    griddata_wcs,
    image_wcs,
)

#Create an Image object for SkyModel

N_CHAN = 100
N_POL = 2
Y = 512
X = 256
CLEAN_BEAM = {"bmaj": 0.1, "bmin": 0.1, "bpa": 0.1}

WCS_HEADER = {
    "CTYPE1": "RA---SIN",
    "CTYPE2": "DEC--SIN",
    "CTYPE3": "STOKES",  # no units, so no CUNIT3
    "CTYPE4": "FREQ",
    "CUNIT1": "deg",
    "CUNIT2": "deg",
    "CUNIT4": "Hz",
    "CRPIX1": 120,  # CRPIX1-4 are reference pixels
    "CRPIX2": 120,
    "CRPIX3": 1,
    "CRPIX4": 1,
    "CRVAL1": 40.0,  # RA in deg
    "CRVAL2": 0.0,  # DEC in deg
    "CDELT1": -0.1,
    "CDELT2": 0.1,  # abs(CDELT2) = cellsize in deg
    "CDELT3": 3,  # delta between polarisation values (I=0, V=4)
    "CDELT4": 10.0,  # delta between frequency values
}
data = numpy.ones((N_CHAN, N_POL, Y, X))
pol_frame = PolarisationFrame("stokesIV")
wcs = WCS(header=WCS_HEADER, naxis=4)
image = Image.constructor(data, pol_frame, wcs, clean_beam=CLEAN_BEAM)
components=None

# Create a GainTable object for SkyModel
name = "MID"
location = (5109237.71471275, 2006795.66194638, -3239109.1838011)
names = "M000"
xyz = 222
mount = "altaz"
frame=None
receptor_frame=ReceptorFrame("linear")
diameter = 13.5
offset = 0.0
stations = 0
vp_type = "MEERKAT" 

# Create a Configuration object for GainTable

configuration = Configuration.constructor(name, location, 
    names, xyz, mount, frame, receptor_frame, diameter,
    offset, stations, vp_type)

gain = numpy.ones((1,1,1,1,1))
time = numpy.ones((1))
interval = numpy.ones((1))
weight = numpy.ones((1,1,1,1,1))
residual = numpy.ones((1,1,1,1))
frequency = numpy.ones((1))
receptor_frame = ReceptorFrame("linear")
phasecentre = (180., -35.)
jones_type = "T"
gaintable = GainTable.constructor(gain, time, interval, 
    weight, residual, frequency, receptor_frame, phasecentre, 
    configuration, jones_type)

mask = "Test_mask"
fixed = True

@pytest.fixture(scope="module", name="result_skyModel")
def fixture_skyModel():
    """
    Generate a simple image using __init__.
    """
    
    skyModel = SkyModel(image, components, gaintable,
        mask, fixed)
    return skyModel


def test__str__(result_skyModel):
    s = f"SkyModel: fixed: {self.fixed}\n"
    for _, sc in enumerate(self.components):
        s += str(sc)
    s += "\n"
    s += str(self.image)
    s += "\n"
    s += str(self.mask)
    s += "\n"
    s += str(self.gaintable)
    assert str(result_skyModel) == s