import itertools
import random
import unittest

from numpy.testing import assert_allclose

from arl.fourier_transforms.fft_support import *
from arl.fourier_transforms.convolutional_gridding import w_kernel_function, kernel_oversample, \
    coordinates2, anti_aliasing_transform, anti_aliasing_calculate, fixed_kernel_degrid, fixed_kernel_grid


class TestConvolutionalGridding(unittest.TestCase):
    
    def assertAlmostEqualScalar(self, a, result=1.0):
        w = result * numpy.ones_like(result)
        
    def _test_pattern(self, npixel):
        return coordinates2(npixel)[0] + coordinates2(npixel)[1] * 1j

    
    def test_anti_aliasing_transform(self):
        for shape in [(64, 64), (128, 128), (256, 256)]:
            _, aaf = anti_aliasing_transform(shape, 8)
            self.assertAlmostEqual(numpy.max(aaf[..., aaf.shape[1] // 2, aaf.shape[0] // 2]), 1.0)


    def test_anti_aliasing_calculate(self):
        for shape in [(64, 64), (128, 128), (256, 256)]:
            _, aaf = anti_aliasing_calculate(shape, 8)
            self.assertAlmostEqual(numpy.max(aaf[..., aaf.shape[1] // 2, aaf.shape[0] // 2]), 1.0)

    def test_w_kernel_function(self):
        assert_allclose(w_kernel_function(5, 0.1, 0), 1.0)
        self.assertAlmostEqualScalar(w_kernel_function(5, 0.1, 100)[2, 2], 1)
        self.assertAlmostEqualScalar(w_kernel_function(10, 0.1, 100)[5, 5], 1)
        self.assertAlmostEqualScalar(w_kernel_function(11, 0.1, 1000)[5, 5], 1)
    
    def test_kernel_oversampled_subgrid(self):
        # Oversampling should produce the same values where sub-grids overlap
        for npixel in range(3, 30):
            pat = self._test_pattern(npixel)
            kern = kernel_oversample(pat, npixel, 1, npixel - 2)
            kern2 = kernel_oversample(pat, npixel, 2, npixel - 2)
            assert_allclose(kern[0, 0], kern2[0, 0], atol=1e-15)
            kern3 = kernel_oversample(pat, npixel, 3, npixel - 2)
            assert_allclose(kern[0, 0], kern3[0, 0], atol=1e-15)
            kern4 = kernel_oversample(pat, npixel, 4, npixel - 2)
            for ux, uy in itertools.product(range(2), range(2)):
                assert_allclose(kern2[uy, ux], kern4[2 * uy, 2 * ux], atol=1e-15)
            kern8 = kernel_oversample(pat, npixel, 8, npixel - 2)
            for ux, uy in itertools.product(range(3), range(3)):
                assert_allclose(kern4[uy, ux], kern8[2 * uy, 2 * ux], atol=1e-15)
    
    def test_kernel_scale(self):
        # Scaling the grid should not make a difference
        npixel = 10
        wff = numpy.zeros((npixel, npixel))
        wff[npixel // 2, npixel // 2] = 1  # Not the most interesting kernel...
        k = kernel_oversample(wff, npixel, 1, npixel)
        k2 = kernel_oversample(4 * wff, npixel * 2, 1, npixel)
        assert_allclose(k, k2)
    
    def test_w_kernel_normalisation(self):
        # Test w-kernel normalisation. This isn't quite perfect.
        # TODO: Address very poor normalisation.
        for kernel_oversampling in [4, 5, 6]:
            for npixel in [3, 5, 9, 16, 20, 24, 32, 64]:
                k = kernel_oversample(w_kernel_function(npixel + 2, 0.1, npixel * 10), npixel + 2,
                                      kernel_oversampling, npixel)
                assert_allclose(numpy.sum(k), kernel_oversampling ** 2, rtol=0.07)

    def test_convolutional_grid(self):
        shape = (7, 7)
        npixel = 256
        kernel_oversampling = 16
        nvis = 100
        nchan = 1
        npol = 4
        uvgrid = numpy.zeros([nchan, npol, npixel, npixel], dtype='complex')
        gcf, kernel = anti_aliasing_transform((npixel, npixel), 8)
        # kernel has shape [kernel_oversampling, kernel_oversampling, npixel, npixel] The fractional
        # part of the coordinate maps onto the first two axes.
        # Make some uv coordinates with random locations
        uvcoords = numpy.array([[random.uniform(-0.25, 0.25), random.uniform(-0.25, 0.25)] for ivis in range(nvis)])
        # Make some visibilities, all complex unity
        vis = numpy.ones([nchan, npol, nvis], dtype='complex')
        visweights = numpy.ones([nchan, npol, nvis])
        uvscale = numpy.array([1.0])
        fixed_kernel_grid(kernel, uvgrid, uvcoords, uvscale, vis, visweights)

    def test_convolutional_degrid(self):
        shape = (7, 7)
        npixel = 256
        kernel_oversampling = 16
        nvis = 100
        nchan = 1
        npol = 4
        uvgrid = numpy.ones([nchan, npol, npixel, npixel], dtype='complex')
        gcf, kernel = anti_aliasing_transform((npixel, npixel), 8)
        # gcf has shape [kernel_oversampling, kernel_oversampling, npixel, npixel] The fractional
        # part of the coordinate maps onto the first two axes.
        # Make some uv coordinates with random locations
        uvcoords = numpy.array([[random.uniform(-0.25, 0.25), random.uniform(-0.25, 0.25)] for ivis in range(nvis)])
        # Degrid the visibilities
        uvscale = numpy.array([1.0])
        vis = fixed_kernel_degrid(kernel, uvgrid, uvcoords, uvscale)
        assert vis.shape[0] == nvis
        assert vis.shape[1] == nchan
        assert vis.shape[2] == npol
        
    def test_w_kernel_lambda(self):
        

if __name__ == '__main__':
    unittest.main()
