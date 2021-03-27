""" Unit tests for image deconvolution


"""
import os
import logging
import unittest

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame
from rascil.data_models import Skycomponent

from rascil.processing_components.arrays.cleaners import overlapIndices
from rascil.processing_components.skycomponent.operations import restore_skycomponent

from rascil.processing_components.image.deconvolution import (
    deconvolve_cube,
    restore_cube,
    fit_psf,
)
from rascil.processing_components.image.operations import export_image_to_fits, qa_image
from rascil.processing_components.simulation import create_test_image
from rascil.processing_components.simulation import create_named_configuration
from rascil.processing_components.visibility.base import create_blockvisibility
from rascil.processing_components.imaging.base import (
    predict_2d,
    invert_2d,
    create_image_from_visibility,
)

log = logging.getLogger("rascil-logger")

log.setLevel(logging.INFO)


class TestImageDeconvolution(unittest.TestCase):
    def setUp(self):

        self.persist = os.getenv("RASCIL_PERSIST", True)

        from rascil.data_models.parameters import rascil_path, rascil_data_path

        self.dir = rascil_path("test_results")
        self.lowcore = create_named_configuration("LOWBD2-CORE")
        self.times = (numpy.pi / (12.0)) * numpy.linspace(-3.0, 3.0, 7)
        self.frequency = numpy.array([1e8])
        self.channel_bandwidth = numpy.array([1e6])
        self.phasecentre = SkyCoord(
            ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame="icrs", equinox="J2000"
        )
        self.vis = create_blockvisibility(
            self.lowcore,
            self.times,
            self.frequency,
            channel_bandwidth=self.channel_bandwidth,
            phasecentre=self.phasecentre,
            weight=1.0,
            polarisation_frame=PolarisationFrame("stokesI"),
            zerow=True,
        )
        self.vis["vis"].data *= 0.0

        # Create model
        self.test_model = create_test_image(
            cellsize=0.001, frequency=self.frequency, phasecentre=self.vis.phasecentre
        )
        self.vis = predict_2d(self.vis, self.test_model)
        assert numpy.max(numpy.abs(self.vis.vis)) > 0.0
        self.model = create_image_from_visibility(
            self.vis,
            npixel=512,
            cellsize=0.001,
            polarisation_frame=PolarisationFrame("stokesI"),
        )
        self.dirty, sumwt = invert_2d(self.vis, self.model)
        self.psf, sumwt = invert_2d(self.vis, self.model, dopsf=True)

    def overlaptest(self, a1, a2, s1, s2):
        #
        a1[s1[0] : s1[1], s1[2] : s1[3]] = 1
        a2[s2[0] : s2[1], s2[2] : s2[3]] = 1
        return numpy.sum(a1) == numpy.sum(a2)

    def test_overlap(self):
        res = numpy.zeros([512, 512])
        psf = numpy.zeros([100, 100])
        peak = (499, 249)
        s1, s2 = overlapIndices(res, psf, peak[0], peak[1])
        assert len(s1) == 4
        assert len(s2) == 4
        self.overlaptest(res, psf, s1, s2)
        assert s1 == (449, 512, 199, 299)
        assert s2 == (0, 63, 0, 100)

    def test_restore(self):
        self.model["pixels"].data[0, 0, 256, 256] = 1.0
        self.cmodel = restore_cube(self.model, self.psf)
        assert numpy.abs(numpy.max(self.cmodel["pixels"].data) - 1.0) < 1e-7, numpy.max(
            self.cmodel["pixels"].data
        )
        if self.persist:
            export_image_to_fits(self.cmodel, "%s/test_restore.fits" % (self.dir))

    def test_restore_clean_beam(self):
        """ Test restoration with specified beam beam
        
        :return:
        """
        self.model["pixels"].data[0, 0, 256, 256] = 1.0
        # The beam is specified in degrees
        bmaj = 0.006 * 180.0 / numpy.pi
        self.cmodel = restore_cube(
            self.model, self.psf, clean_beam={"bmaj": bmaj, "bmin": bmaj, "bpa": 0.0}
        )
        assert numpy.abs(numpy.max(self.cmodel["pixels"].data) - 1.0) < 1e-7, numpy.max(
            self.cmodel["pixels"].data
        )
        if self.persist:
            export_image_to_fits(
                self.cmodel, "%s/test_restore_6mrad_beam.fits" % (self.dir)
            )

    def test_restore_skycomponent(self):
        """ Test restoration of single pixel and skycomponent
        """
        self.model["pixels"].data[0, 0, 256, 256] = 0.5

        sc = Skycomponent(
            flux=numpy.array([[1.0]]),
            direction=SkyCoord(
                ra=+180.0 * u.deg, dec=-61.0 * u.deg, frame="icrs", equinox="J2000"
            ),
            shape="Point",
            frequency=self.frequency,
            polarisation_frame=PolarisationFrame("stokesI")
        )
        bmaj = 0.012 * 180.0 / numpy.pi
        clean_beam = {"bmaj": bmaj, "bmin": bmaj / 2.0, "bpa": 15.0}
        self.cmodel = restore_cube(self.model, clean_beam=clean_beam)
        self.cmodel = restore_skycomponent(
            self.cmodel, sc, clean_beam=clean_beam
        )
        self.persist = True
        if self.persist:
            export_image_to_fits(
                self.cmodel, "%s/test_restore_skycomponent.fits" % (self.dir)
            )
        assert numpy.abs(numpy.max(self.cmodel["pixels"].data) - 0.9935351105096656) < 1e-7, numpy.max(
            self.cmodel["pixels"].data
        )

    def test_fit_psf(self):
        clean_beam = fit_psf(self.psf)
        if self.persist:
            export_image_to_fits(self.psf, "%s/test_fit_psf.fits" % (self.dir))

        assert numpy.abs(clean_beam["bmaj"] - 1048.7435130499214) < 1.0e-7, clean_beam
        assert numpy.abs(clean_beam["bmin"] - 1003.1945574030732) < 1.0e-7, clean_beam
        assert numpy.abs(clean_beam["bpa"] + 1.0098903330636544) < 1.0e-7, clean_beam

    def test_deconvolve_hogbom(self):
        self.comp, self.residual = deconvolve_cube(
            self.dirty,
            self.psf,
            niter=10000,
            gain=0.1,
            algorithm="hogbom",
            threshold=0.01,
        )
        if self.persist:
            export_image_to_fits(
                self.residual, "%s/test_deconvolve_hogbom-residual.fits" % (self.dir)
            )
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist:
            export_image_to_fits(
                self.cmodel, "%s/test_deconvolve_hogbom-clean.fits" % (self.dir)
            )
        assert numpy.max(self.residual["pixels"].data) < 1.2

    def test_deconvolve_msclean(self):
        self.comp, self.residual = deconvolve_cube(
            self.dirty,
            self.psf,
            niter=1000,
            gain=0.7,
            algorithm="msclean",
            scales=[0, 3, 10, 30],
            threshold=0.01,
        )
        if self.persist:
            export_image_to_fits(
                self.comp, "%s/test_deconvolve_msclean-comp.fits" % (self.dir)
            )
        if self.persist:
            export_image_to_fits(
                self.residual, "%s/test_deconvolve_msclean-residual.fits" % (self.dir)
            )
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist:
            export_image_to_fits(
                self.cmodel, "%s/test_deconvolve_msclean-clean.fits" % (self.dir)
            )
        assert numpy.max(self.residual["pixels"].data) < 1.2

    def test_deconvolve_msclean_1scale(self):

        self.comp, self.residual = deconvolve_cube(
            self.dirty,
            self.psf,
            niter=10000,
            gain=0.1,
            algorithm="msclean",
            scales=[0],
            threshold=0.01,
        )
        if self.persist:
            export_image_to_fits(
                self.comp, "%s/test_deconvolve_msclean_1scale-comp.fits" % (self.dir)
            )
        if self.persist:
            export_image_to_fits(
                self.residual,
                "%s/test_deconvolve_msclean_1scale-residual.fits" % (self.dir),
            )
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist:
            export_image_to_fits(
                self.cmodel, "%s/test_deconvolve_msclean_1scale-clean.fits" % (self.dir)
            )
        assert numpy.max(self.residual["pixels"].data) < 1.2

    def test_deconvolve_hogbom_no_edge(self):
        self.comp, self.residual = deconvolve_cube(
            self.dirty,
            self.psf,
            window_shape="no_edge",
            niter=10000,
            gain=0.1,
            algorithm="hogbom",
            threshold=0.01,
        )
        if self.persist:
            export_image_to_fits(
                self.residual,
                "%s/test_deconvolve_hogbom_noedge-residual.fits" % (self.dir),
            )
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist:
            export_image_to_fits(
                self.cmodel, "%s/test_deconvolve_hogbom_no_edge-clean.fits" % (self.dir)
            )
        assert numpy.max(self.residual["pixels"].data) < 1.2

    def test_deconvolve_hogbom_inner_quarter(self):
        self.comp, self.residual = deconvolve_cube(
            self.dirty,
            self.psf,
            window_shape="quarter",
            niter=10000,
            gain=0.1,
            algorithm="hogbom",
            threshold=0.01,
        )
        if self.persist:
            export_image_to_fits(
                self.residual,
                "%s/test_deconvolve_hogbom_innerquarter-residual.fits" % (self.dir),
            )
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist:
            export_image_to_fits(
                self.cmodel,
                "%s/test_deconvolve_hogbom_innerquarter-clean.fits" % (self.dir),
            )
        assert numpy.max(self.residual["pixels"].data) < 1.2

    def test_deconvolve_msclean_inner_quarter(self):

        self.comp, self.residual = deconvolve_cube(
            self.dirty,
            self.psf,
            window_shape="quarter",
            niter=1000,
            gain=0.7,
            algorithm="msclean",
            scales=[0, 3, 10, 30],
            threshold=0.01,
        )
        if self.persist:
            export_image_to_fits(
                self.comp,
                "%s/test_deconvolve_msclean_innerquarter-comp.fits" % (self.dir),
            )
        if self.persist:
            export_image_to_fits(
                self.residual,
                "%s/test_deconvolve_msclean_innerquarter-residual.fits" % (self.dir),
            )
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist:
            export_image_to_fits(
                self.cmodel,
                "%s/test_deconvolve_msclean_innerquarter-clean.fits" % (self.dir),
            )
        assert numpy.max(self.residual["pixels"].data) < 1.2

    def test_deconvolve_hogbom_subpsf(self):

        self.comp, self.residual = deconvolve_cube(
            self.dirty,
            psf=self.psf,
            psf_support=200,
            window_shape="quarter",
            niter=10000,
            gain=0.1,
            algorithm="hogbom",
            threshold=0.01,
        )
        if self.persist:
            export_image_to_fits(
                self.residual,
                "%s/test_deconvolve_hogbom_subpsf-residual.fits" % (self.dir),
            )
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist:
            export_image_to_fits(
                self.cmodel, "%s/test_deconvolve_hogbom_subpsf-clean.fits" % (self.dir)
            )
        assert numpy.max(self.residual["pixels"].data[..., 56:456, 56:456]) < 1.2

    def test_deconvolve_msclean_subpsf(self):

        self.comp, self.residual = deconvolve_cube(
            self.dirty,
            psf=self.psf,
            psf_support=200,
            window_shape="quarter",
            niter=1000,
            gain=0.7,
            algorithm="msclean",
            scales=[0, 3, 10, 30],
            threshold=0.01,
        )
        if self.persist:
            export_image_to_fits(
                self.comp, "%s/test_deconvolve_msclean_subpsf-comp.fits" % (self.dir)
            )
        if self.persist:
            export_image_to_fits(
                self.residual,
                "%s/test_deconvolve_msclean_subpsf-residual.fits" % (self.dir),
            )
        self.cmodel = restore_cube(self.comp, self.psf, self.residual)
        if self.persist:
            export_image_to_fits(
                self.cmodel, "%s/test_deconvolve_msclean_subpsf-clean.fits" % (self.dir)
            )
        assert numpy.max(self.residual["pixels"].data[..., 56:456, 56:456]) < 1.0


if __name__ == "__main__":
    unittest.main()
