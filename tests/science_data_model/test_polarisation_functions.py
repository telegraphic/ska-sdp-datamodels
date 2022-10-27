"""
Unit processing_components for polarisation_convert
"""
import numpy
import pytest
from numpy import random
from numpy.testing import assert_array_almost_equal

from ska_sdp_datamodels.science_data_model import (
    PolarisationFrame,
    ReceptorFrame,
    congruent_polarisation,
    convert_circular_to_stokes,
    convert_linear_to_stokes,
    convert_pol_frame,
    convert_stokes_to_circular,
    convert_stokes_to_linear,
    correlate_polarisation,
    polarisation_frame_from_names,
)


@pytest.mark.parametrize("frame_type", ["linear", "circular", "stokesI"])
def test_correlate(frame_type):
    """
    Correct PolarisationFrame is given for a ReceptorFrame
    """
    rec_frame = ReceptorFrame(frame_type)
    expected = PolarisationFrame(frame_type)
    result = correlate_polarisation(rec_frame)
    assert result == expected


@pytest.mark.parametrize("frame_type", ["linear", "circular", "stokesI"])
def test_congruent(frame_type):
    """
    Function correctly determines if PolFrame and ReceptorFrame match.
    """
    rec_frame = ReceptorFrame(frame_type)
    pol_frame = PolarisationFrame(frame_type)
    result = congruent_polarisation(rec_frame, pol_frame)
    assert result is True


@pytest.mark.parametrize("frame_type", ["linear", "circular", "stokesI"])
def test_congruent_not(frame_type):
    """
    Function correctly determines that PolFrame and
    ReceptorFrame do not match.
    """
    rec_frame = ReceptorFrame(frame_type)
    pol_frame = PolarisationFrame("stokesIQ")
    result = congruent_polarisation(rec_frame, pol_frame)
    assert result is False


@pytest.mark.parametrize(
    "frame_type",
    [
        "circular",
        "circularnp",
        "linear",
        "linearnp",
        "stokesIQUV",
        "stokesIV",
        "stokesIQ",
        "stokesI",
    ],
)
def test_extract_polarisation_frame(frame_type):
    """
    Frame type is correctly determined from frame name
    """
    polarisation_frame = PolarisationFrame(frame_type)
    assert polarisation_frame.type == frame_type
    names = polarisation_frame.names

    result = polarisation_frame_from_names(names)
    assert result == frame_type


def test_extract_polarisation_frame_fail():
    """
    Frame type cannot be determined from incorrect name.
    Raises ValueError.
    """
    with pytest.raises(ValueError):
        fake_name = ["foo", "bar"]
        polarisation_frame_from_names(fake_name)


def test_stokes_linear_conversion():
    """
    Convert StokesIQUV to Linear correctly.
    """
    stokes = numpy.array([1.0, 0.0, 0.0, 0.0])
    result = convert_stokes_to_linear(stokes, 0)
    assert_array_almost_equal(
        result, numpy.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j])
    )

    stokes = numpy.array([1.0, 0.0])
    result = convert_stokes_to_linear(stokes, 0)
    assert_array_almost_equal(result, numpy.array([1.0 + 0j, 1.0 + 0j]))

    stokes = numpy.array([0.0, 1.0, 0.0, 0.0])
    result = convert_stokes_to_linear(stokes, 0)
    assert_array_almost_equal(
        result, numpy.array([1.0 + 0j, 0j, 0j, -1.0 + 0j])
    )

    stokes = numpy.array([0.0, 0.0, 1.0, 0.0])
    result = convert_stokes_to_linear(stokes, 0)
    assert_array_almost_equal(
        result, numpy.array([0.0 + 0j, 1.0 + 0j, 1.0 + 0j, 0.0 + 0j])
    )

    stokes = numpy.array([0.0, 0.0, 0.0, 1.0])
    result = convert_stokes_to_linear(stokes, 0)
    assert_array_almost_equal(
        result, numpy.array([0.0 + 0j, +1.0j, -1.0j, 0.0 + 0j])
    )

    stokes = numpy.array([1.0, -0.8, 0.2, 0.01])
    result = convert_stokes_to_linear(stokes, 0)
    assert_array_almost_equal(
        result,
        numpy.array([0.2 + 0.0j, 0.2 + 0.01j, 0.2 - 0.01j, 1.8 + 0.0j]),
    )


def test_stokes_circular_conversion():
    """
    Convert StokesIQUV to circular correctly.
    """
    stokes = numpy.array([1.0, 0.0, 0.0, 0.0])
    result = convert_stokes_to_circular(stokes, 0)
    assert_array_almost_equal(
        result, numpy.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j])
    )

    stokes = numpy.array([1.0, 0.0])
    result = convert_stokes_to_circular(stokes, 0)
    assert_array_almost_equal(result, numpy.array([1.0 + 0j, 1.0 + 0j]))

    stokes = numpy.array([0.0, 1.0, 0.0, 0.0])
    result = convert_stokes_to_circular(stokes, 0)
    assert_array_almost_equal(
        result, numpy.array([0.0 + 0j, -1j, -1j, 0.0 + 0j])
    )

    stokes = numpy.array([0.0, 0.0, 1.0, 0.0])
    result = convert_stokes_to_circular(stokes, 0)
    assert_array_almost_equal(
        result, numpy.array([0.0 + 0j, 1.0 + 0j, -1.0 + 0j, 0.0 + 0j])
    )

    stokes = numpy.array([0.0, 0.0, 0.0, 1.0])
    result = convert_stokes_to_circular(stokes, 0)
    assert_array_almost_equal(
        result, numpy.array([1.0 + 0j, +0.0j, 0.0j, -1.0 + 0j])
    )

    stokes = numpy.array([1.0, -0.8, 0.2, 0.01])
    linear = convert_stokes_to_circular(stokes, 0)
    assert_array_almost_equal(
        linear,
        numpy.array([1.01 + 0.0j, 0.2 + 0.8j, -0.2 + 0.8j, 0.99 + 0.0j]),
    )


def test_stokes_linear_stokes_conversion():
    """
    Converting from stokes to linear and then
    back to stokes gives the starting array.
    """
    stokes = numpy.array([1, 0.5, 0.2, -0.1])
    linear = convert_stokes_to_linear(stokes, 0)

    result = convert_linear_to_stokes(linear, 0).real
    assert_array_almost_equal(result, stokes, 15)


def test_stokes_linearnp_stokes_iq_conversion():
    """
    Converting from stokesIQ to linearnp and then
    back to stokesIQ gives the starting array.
    """
    stokes = numpy.array([1, 0.5])
    linearnp = convert_stokes_to_linear(stokes, 0)

    result = convert_linear_to_stokes(linearnp, 0).real
    assert_array_almost_equal(result, stokes, 15)


def test_stokes_circular_stokes_conversion():
    """
    Converting from stokes to circular and then
    back to stokes gives the starting array.
    """
    stokes = numpy.array([1, 0.5, 0.2, -0.1])
    circular = convert_stokes_to_circular(stokes, 0)

    result = convert_circular_to_stokes(circular, 0).real
    assert_array_almost_equal(result, stokes, 15)


def test_stokes_circularnp_stokes_iv_conversion():
    """
    Converting from stokesIV to circularnp and then
    back to stokesIV gives the starting array.
    """
    stokes = numpy.array([1, 0.5])
    circularnp = convert_stokes_to_circular(stokes, 0)

    result = convert_circular_to_stokes(circularnp, 0).real
    assert_array_almost_equal(result, stokes, 15)


def test_image_conversion():
    """
    Do image conversion
    (These are legacy tests, not sure what they
    meant by "image_conversion")
    """
    stokes = numpy.array(random.uniform(-1.0, 1.0, [3, 4, 128, 128]))
    cir = convert_stokes_to_circular(stokes, 1)

    result = convert_circular_to_stokes(cir).real
    assert_array_almost_equal(result, stokes, 15)


@pytest.mark.parametrize(
    "frame_in, frame_out, pol_dims",
    [
        ["stokesIQUV", "circular", 4],
        ["stokesIV", "circularnp", 2],
        ["stokesIQUV", "linear", 4],
        ["stokesIQ", "linearnp", 2],
        ["stokesI", "stokesI", 4],
    ],
)
def test_image_auto_conversion_circular(frame_in, frame_out, pol_dims):
    """
    convert_pol_frame automatically determines
    the conversion between polarisation frames.
    """
    stokes = numpy.array(random.uniform(-1.0, 1.0, [3, pol_dims, 128, 128]))
    input_pf = PolarisationFrame(frame_in)
    output_pf = PolarisationFrame(frame_out)

    first_convert = convert_pol_frame(stokes, input_pf, output_pf, polaxis=1)
    result = convert_pol_frame(first_convert, output_pf, input_pf, polaxis=1)
    assert_array_almost_equal(result.real, stokes, 15)


def test_image_conversion_stokes_iquv_to_i():
    """
    Convert StokesIQUV frame to StokesI
    """
    flux = numpy.array(
        [
            [1.0, 0.0, 1.13, 0.4],
            [2.0, 0.0, 11.13, 2.4],
            [-1.0, 0.0, -1.3, -0.72],
            [10.0, 0.0, 2.4, 1.1],
        ]
    )
    expected_flux = flux[:, 0]
    expected_flux = expected_flux.reshape((len(expected_flux), 1))
    ipf = PolarisationFrame("stokesIQUV")
    opf = PolarisationFrame("stokesI")

    result = convert_pol_frame(flux, ipf, opf)

    assert result.shape == expected_flux.shape
    assert (result == expected_flux).all()


def test_image_auto_conversion_stokes_i_to_iquv():
    """
    Convert StokesI frame to StokesIQUV
    """
    expected_flux = numpy.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],
        ]
    )
    input_flux = expected_flux[:, 0]
    input_flux = input_flux.reshape((len(input_flux), 1))
    ipf = PolarisationFrame("stokesI")
    opf = PolarisationFrame("stokesIQUV")

    result = convert_pol_frame(input_flux, ipf, opf)

    assert result.shape == expected_flux.shape
    assert (result == expected_flux).all()


def test_vis_conversion():
    """
    Do visibility conversion
    (These are legacy tests, not sure what they
    meant by "vis_conversion")
    """
    stokes = numpy.array(random.uniform(-1.0, 1.0, [1000, 3, 4]))
    cir = convert_stokes_to_circular(stokes, polaxis=2)

    result = convert_circular_to_stokes(cir, polaxis=2)
    assert_array_almost_equal(result.real, stokes, 15)


@pytest.mark.parametrize(
    "frame_in, frame_out", [["stokesIQUV", "circular"], ["stokesI", "stokesI"]]
)
def test_vis_auto_conversion(frame_in, frame_out):
    """
    convert_pol_frame automatically determines
    the conversion between polarisation frames.
    """
    stokes = numpy.array(random.uniform(-1.0, 1.0, [1000, 3, 4]))
    input_pf = PolarisationFrame(frame_in)
    output_pf = PolarisationFrame(frame_out)
    cir = convert_pol_frame(stokes, input_pf, output_pf, polaxis=2)

    result = convert_pol_frame(cir, output_pf, input_pf, polaxis=2)
    assert_array_almost_equal(result.real, stokes, 15)


def test_circular_to_linear_fail():
    """
    Conversion with wrong pol_frame raises ValueError
    """
    stokes = numpy.array(random.uniform(-1.0, 1.0, [3, 4, 128, 128]))
    ipf = PolarisationFrame("stokesIQUV")
    opf = PolarisationFrame("circular")
    cir = convert_pol_frame(stokes, ipf, opf)
    wrong_pf = PolarisationFrame("linear")
    with pytest.raises(ValueError):
        convert_pol_frame(cir, opf, wrong_pf)
