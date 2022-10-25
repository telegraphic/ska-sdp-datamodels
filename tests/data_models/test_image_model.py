"""
Unit tests for Image()
"""
import tempfile

import numpy
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from xarray import DataArray

from ska_sdp_datamodels.image import Image
from ska_sdp_datamodels.science_data_model import PolarisationFrame

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


@pytest.fixture(scope="module", name="result_image")
def fixture_image():
    """
    Generate a simple image using Image.constructor.
    """
    data = numpy.ones((N_CHAN, N_POL, Y, X))
    pol_frame = PolarisationFrame("stokesIV")
    wcs = WCS(header=WCS_HEADER, naxis=4)
    image = Image.constructor(data, pol_frame, wcs, clean_beam=CLEAN_BEAM)
    return image


def test_constructor_coords(result_image):
    """
    Constructor correctly generates coordinates
    from simple wcs input.
    """
    expected_coords_keys = ["polarisation", "frequency", "x", "y"]

    result_coords = result_image.coords

    assert sorted(result_coords.keys()) == sorted(expected_coords_keys)
    assert result_coords["frequency"].shape == (N_CHAN,)
    assert (result_coords["polarisation"].data == ["I", "V"]).all()
    assert result_coords["x"].shape == (X,)
    assert result_coords["y"].shape == (Y,)
    assert result_coords["x"][0] != result_coords["x"][-1]
    assert result_coords["y"][0] != result_coords["y"][-1]


def test_constructor_attrs(result_image):
    """
    Constructor correctly generates attributes.
    """
    result_attrs = result_image.attrs

    assert result_attrs["data_model"] == "Image"
    assert result_attrs["_polarisation_frame"] == "stokesIV"
    assert result_attrs["clean_beam"] == {"bmaj": 0.1, "bmin": 0.1, "bpa": 0.1}
    assert result_attrs["channel_bandwidth"] == WCS_HEADER["CDELT4"]
    assert result_attrs["ra"].round(3) == 0.698  # rad of WCS_HEADER["CRVAL1"]
    assert result_attrs["dec"] == 0.0  # rad of WCS_HEADER["CRVAL2"]
    assert (
        result_attrs["refpixel"]
        == [
            WCS_HEADER["CRPIX1"],
            WCS_HEADER["CRPIX2"],
            WCS_HEADER["CRPIX3"],
            WCS_HEADER["CRPIX4"],
        ]
    ).all()


def test_constructor_data_vars(result_image):
    """
    Constructor correctly generates data variables.
    """
    result_data_vars = result_image.data_vars

    assert len(result_data_vars) == 1  # only var is pixels
    assert "pixels" in result_data_vars.keys()
    assert isinstance(result_data_vars["pixels"], DataArray)
    assert (result_data_vars["pixels"].data == 1.0).all()


def test_constructor_clean_beam_missing_key():
    """
    Constructor throws an error if clean_beam
    doesn't contain all the keys of ["bmaj", "bmin", "bpa"].
    """

    data = numpy.ones((N_CHAN, N_POL, Y, X))
    pol_frame = PolarisationFrame("stokesIV")
    wcs = WCS(header=WCS_HEADER, naxis=4)

    clean_beam = {"bmaj": 0.1}

    with pytest.raises(KeyError) as error:
        Image.constructor(data, pol_frame, wcs, clean_beam=clean_beam)

    error_message = error.value.args[0]
    assert (
        error_message == "Image: clean_beam must have key(s): ['bmin', 'bpa']"
    )


def test_is_canonical_true(result_image):
    """
    WCS Header contains canonical information
    --> image is canonical.
    """
    assert result_image.image_acc.is_canonical() is True


def test_is_canonical_false():
    """
    WCS Header contains non-canonical information
    --> image is not canonical.
    """
    new_header = WCS_HEADER.copy()
    new_header["CTYPE1"] = "GLON-CAR"
    new_header["CTYPE2"] = "GLAT-CAR"
    del new_header["CTYPE3"]
    del new_header["CTYPE4"]
    del new_header["CUNIT4"]

    data = numpy.ones((N_CHAN, N_POL, Y, X))
    pol_frame = PolarisationFrame("stokesIV")
    wcs = WCS(header=new_header, naxis=4)
    image = Image.constructor(data, pol_frame, wcs, clean_beam=None)

    assert image.image_acc.is_canonical() is False


def test_export_to_fits(result_image):
    """
    export_to_fits generates a FITS file, which when
    read back in, contains the data from result_image
    and header values match those of result_image's WCS.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        test_fits = f"{temp_dir}/test_export_to_fits_result.fits"
        result_image.image_acc.export_to_fits(test_fits)

        imported_image = fits.open(test_fits)[0]

        # pylint: disable=no-member
        assert (
            imported_image.data == result_image.data_vars["pixels"].data
        ).all()
        for key, value in WCS_HEADER.items():
            assert imported_image.header[key] == value, f"{key} mismatch"

        # when clean_beam is part of the image, those values
        # are also added to the FITS header
        for key, value in CLEAN_BEAM.items():
            assert imported_image.header[key] == value, f"{key} mismatch"


def test_qa_image(result_image):
    """
    QualityAssessment of object data values
    are derived correctly.

    Note: input "result_image" contains 1.0 for
    every pixel value, which is used to determine most
    of QA data values.
    """
    expected_data = {  # except "size"
        "shape": f"({N_CHAN}, {N_POL}, {Y}, {X})",
        "max": 1.0,
        "min": 1.0,
        "maxabs": 1.0,
        "rms": 0.0,
        "sum": 26214400.0,
        "medianabs": 1.0,
        "medianabsdevmedian": 0.0,
        "median": 1.0,
    }

    result_qa = result_image.image_acc.qa_image(context="Test")

    assert result_qa.context == "Test"
    del result_qa.data[
        "size"
    ]  # we are not testing the size determined from __sizeof__
    for key, value in expected_data.items():
        assert result_qa.data[key] == value, f"{key} mismatch"


def test_property_accessor(result_image):
    """
    Image.image_acc (xarray accessor) returns
    properties correctly.
    """
    accessor_object = result_image.image_acc

    assert accessor_object.shape == (N_CHAN, N_POL, Y, X)
    assert accessor_object.nchan == 100
    assert accessor_object.npol == 2
    assert accessor_object.polarisation_frame == PolarisationFrame("stokesIV")
    assert accessor_object.projection == (
        WCS_HEADER["CTYPE1"],
        WCS_HEADER["CTYPE2"],
    )
    assert (
        accessor_object.phasecentre.ra.value == WCS_HEADER["CRVAL1"]
    )  # in deg
    assert (
        accessor_object.phasecentre.dec.value == WCS_HEADER["CRVAL2"]
    )  # in deg

    for key, value in WCS_HEADER.items():
        assert accessor_object.wcs.to_header()[key] == value, f"{key} mismatch"
