.. _xarray:

.. toctree::
   :maxdepth: 3

Use of xarray
*************

ska-sdp-datamodels uses the `Xarray <https:/www.dask.org>`_ library instead of numpy in the
data classes. All data classes are derived from xarray.Dataset. This change is motivated
by the large range of capabilities available from xarray. These include:

 - Named dimensions and coordinates, allowing access via quantities such as time. frequency, polarisation, receptor
 - Indexing, selection, iteration, and conditions
 - Support of split-apply-recombine operations
 - Interpolation in coordinates, including missing values
 - Automatic invocation of Dask for array operations
 - Arbitrary meta data as attributes

We have chosen to make data classes derive from xarray.Dataset. Instead of adding
some of the class methods to the data class, which would introduce some interface fragility
as xarray changes over time, we have used data accessors to control access to
these methods specific to the class. This design is suggested in the xarray documentation
on extending xarray.


Examples::

    # Flagged visibility
    vis.visibility_acc.flagged_vis

    # UVW in wavelengths
    vis.visibility_acc.uvw_lambda

    # DataArray sizes
    vis.visibility_acc.datasizes

    # Phasecentre as an astropy.SkyCoord
    im.image_acc.phasecentre

    # Image RA, Dec grid
    im.image_acc.ra_dec_mesh

    # Gaintable number of receptors
    gt.gaintable_acc.nrec

Usage Examples
*************************************

The steps required are:

 - For Image, GridData, and ConvolutionFunction, the name of the data variable is pixels so for example::

    # The previous numpy format
    im.data
    # as an Xarray becomes
    im["pixels"]
    # as an numpy array or Dask array becomes
    im["pixels"].data
    # as an numpy array becomes
    im["pixels"].values
    # The properties now require using the accessor class. For example:
    im.nchan
    # becomes
    im.image_acc.nchan
    # or directly to the attributes of the xarray.Dataset
    im.attrs["nchan"]


 - For Visibility, the various columns become data variables::

    # The numpy format
    bvis.data["vis"]
    # becomes
    bvis["vis"]
    # as an numpy array or Dask array becomes
    bvis["vis"].data
    # as an numpy array becomes
    bvis["vis"].values
    # The properties now require using the accessor class. For example:
    bvis.nchan
    # becomes
    bvis.visibility_acc.nchan
    # or directly to the attributes of the xarray.Dataset
    bvis.attrs["nchan"]
    # The convenience methods for handling flags also require the accessor:
    bvis.flagged_vis
    # becomes
    bvis.visibility_acc.flagged_vis

Usage with Xarray
*************************************

Here is a simple example of how the capabilities of xarray can be used:

.. code:: ipython3

    vis = create_visibility_from_ms(ms)[0]

    # Don't squeeze out the unit dimensions because we will want
    # them for the concat
    chan_vis = [v[1] for v in vis.groupby_bins(dim, bins=2)]

    # Predict visibility from a model.
    chan_vis = [predict_ng(vis, model) in chan_vis]

    # Now concatenate
    newvis = xarray.concat(chan_vis, dim=dim, data_vars="minimal")

Warnings and Limitations
*************************************

The current main limitation of the xarray implementation is that, some of the class methods and operations
can return a raw xarray.Dataset instead of a subclass instance. So please make sure variables
return the expected data structure when using xarray.