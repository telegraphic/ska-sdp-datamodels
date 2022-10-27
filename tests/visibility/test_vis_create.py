import numpy
import pytest
from astropy import units
from astropy.coordinates import SkyCoord

from ska_sdp_datamodels.configuration import create_named_configuration
from ska_sdp_datamodels.science_data_model import PolarisationFrame
from ska_sdp_datamodels.visibility import create_visibility

LOW_CORE = create_named_configuration("LOW-AA0.5")
TIMES = (numpy.pi / 43200.0) * numpy.arange(0.0, 300.0, 30.0)
FREQUENCY = numpy.linspace(0.8e8, 1.2e8, 5)
CHANNEL_BANDWIDTH = numpy.array([1e7, 1e7, 1e7, 1e7, 1e7])
POLARISATION_FRAME = PolarisationFrame("linear")
PHASE_CENTRE = SkyCoord(
    ra=+180.0 * units.deg, dec=-35.0 * units.deg, frame="icrs", equinox="J2000"
)

# TODO: in the future, add tests for time and flagging related
#  calculations in create_visibility


def test_create_visibility():
    """
    Visibility is created with minimum input to function.
    """
    vis = create_visibility(
        LOW_CORE,
        TIMES,
        FREQUENCY,
        channel_bandwidth=CHANNEL_BANDWIDTH,
        phasecentre=PHASE_CENTRE,
    )
    # shape: (time, baseline, frequency, polarisation)
    # polarisation is 4, because the code defaults to using the frame
    # that matches the configuration receptor frame, which, by default
    # is linear
    assert vis.vis.shape == (len(TIMES), 21, len(FREQUENCY), 4)
    # default polarisation is linear; coming from defailt receptor frame
    # of configuration
    assert (vis.polarisation.data == ["XX", "XY", "YX", "YY"]).all()


def test_create_visibility_no_phase_centre():
    """
    Function raises ValueError if
    phasecentre is None.
    """
    with pytest.raises(ValueError) as error:
        create_visibility(
            LOW_CORE,
            TIMES,
            FREQUENCY,
            channel_bandwidth=CHANNEL_BANDWIDTH,
            phasecentre=None,
            weight=1.0,
        )
    assert str(error.value) == "Must specify phase centre"


def test_create_visibility_polarisation():
    """
    The created visibility hase the
    specified polarisation frame.
    """
    vis = create_visibility(
        LOW_CORE,
        TIMES,
        FREQUENCY,
        channel_bandwidth=CHANNEL_BANDWIDTH,
        phasecentre=PHASE_CENTRE,
        weight=1.0,
        polarisation_frame=PolarisationFrame("stokesI"),
    )
    assert vis.vis.shape[-1] == 1  # polarisation dimension is 1
    assert (vis.polarisation.data == ["I"]).all()
