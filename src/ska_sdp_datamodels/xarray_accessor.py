# pylint: disable=invalid-name

"""
Base class for xarray accessor classes
implemented to be used with othere memory data models
that inherit from xarray.Dataset
"""


class XarrayAccessorMixin:
    """Convenience methods to access the fields of the xarray"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def size(self):
        """Return size in GB"""
        size = self._obj.nbytes
        return size / 1024.0 / 1024.0 / 1024.0

    def datasizes(self):
        """Return string describing sizes of data variables
        :return: string
        """
        s = f"Dataset size: {self._obj.nbytes / 1024 / 1024 / 1024:.3f} GB\n"
        for var in self._obj.data_vars:
            s += (
                f"\t[{var}]: "
                f"\t{self._obj[var].nbytes / 1024 / 1024 / 1024:.3f} GB\n"
            )
        return s
