"""
Utils for testing
"""


def data_model_equals(ds_new, ds_ref):
    """Check if two xarray objects are identical except to values

    Precision in lost in HDF files at close to the machine
    precision so we cannot reliably use xarray.equals().
    So this function is specific to this set of tests

    Throws AssertionError or returns True

    :param ds_ref: xarray Dataset or DataArray
    :param ds_new: xarray Dataset or DataArray
    :return: True or False
    """
    for coord in ds_ref.coords:
        assert coord in ds_new.coords
    for coord in ds_new.coords:
        assert coord in ds_ref.coords
    for var in ds_ref.data_vars:
        assert var in ds_new.data_vars
    for var in ds_new.data_vars:
        assert var in ds_ref.data_vars
    for attr in ds_ref.attrs.keys():
        assert attr in ds_new.attrs.keys()
    for attr in ds_new.attrs.keys():
        assert attr in ds_ref.attrs.keys()
