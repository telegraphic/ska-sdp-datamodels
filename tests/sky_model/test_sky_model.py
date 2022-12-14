"""
Unit tests for the Sky Model
"""


def test_skymodel_copy(sky_model):
    """
    Test copy SkyModel
    """
    new_model = sky_model.copy()
    assert new_model.mask == sky_model.mask
    assert (
        new_model.components[0].frequency.all()
        == sky_model.components[0].frequency.all()
    )
    assert (
        new_model.gaintable["gain"].data.all()
        == sky_model.gaintable["gain"].data.all()
    )
    assert (
        new_model.image["pixels"].data.all()
        == sky_model.image["pixels"].data.all()
    )


def test_sky_model__str__(sky_model):
    """
    Check __str__() returns the correct string
    """
    # Assume copy works
    sky_copy = sky_model.copy()
    sky_copy.components = ""

    sky_model_text = "SkyModel: fixed: False\n"
    sky_model_text += "\n"  # SkyComponent is None
    sky_model_text += f"{sky_copy.image}\n"
    sky_model_text += f"{sky_copy.mask}\n"
    sky_model_text += f"{sky_copy.gaintable}"
    assert str(sky_copy) == sky_model_text


def test_sky_component_nchan(sky_component):
    """
    Check nchans returns correct data
    """
    nchans = sky_component.nchan
    assert nchans == 3


def test_sky_component_npol(sky_component):
    """
    Check npols returns correct data
    """
    npols = sky_component.npol
    assert npols == 4


def test_sky_component__str__(sky_component):
    """
    Check __str__() returns the correct string
    """
    params = {}
    sky_comp_text = "SkyComponent:\n"
    sky_comp_text += "\tName: None\n"
    sky_comp_text += f"\tFlux: {sky_component.flux}\n"
    sky_comp_text += f"\tFrequency: {sky_component.frequency}\n"
    sky_comp_text += f"\tDirection: {sky_component.direction}\n"
    sky_comp_text += "\tShape: Point\n"
    sky_comp_text += f"\tParams: {params}\n"
    sky_comp_text += "\tPolarisation frame: stokesIQUV\n"
    assert str(sky_component) == sky_comp_text


def test_sky_component_copy(sky_component):
    """
    Test copying SkyComponent
    """
    new_sc = sky_component.copy()
    assert new_sc.direction == sky_component.direction
    assert new_sc.frequency.all() == sky_component.frequency.all()
    assert new_sc.flux.all() == sky_component.flux.all()
    assert new_sc.params == sky_component.params
    assert new_sc.polarisation_frame == sky_component.polarisation_frame
