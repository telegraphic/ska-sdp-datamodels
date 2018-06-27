""" Unit tests for visibility operations


"""
import sys
import unittest
import logging

from data_models.parameters import arl_path

from processing_components.visibility.base import create_blockvisibility_from_ms, create_visibility_from_ms
from processing_components.visibility.operations import integrate_visibility_by_channel

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))
log.addHandler(logging.StreamHandler(sys.stderr))

class TestCreateMS(unittest.TestCase):
    
    def setUp(self):
    
        try:
            from casacore.tables import table  # pylint: disable=import-error
            self.casacore_available = True
#            except ModuleNotFoundError:
        except:
            self.casacore_available = False

    def test_create_list(self):
        
        if not self.casacore_available:
            return
        
        msfile = arl_path("data/vis/xcasa.ms")
        self.vis = create_blockvisibility_from_ms(msfile)
        
        for v in self.vis:
            assert v.vis.data.shape[-1] == 4
            assert v.polarisation_frame.type == "circular"

    def test_create_list_spectral(self):
        if not self.casacore_available:
            return
    
        msfile = arl_path("data/vis/ASKAP_example.ms")
    
        vis_by_channel = list()
        nchan_ave = 16
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan + nchan_ave)
            v = create_visibility_from_ms(msfile, range(schan, max_chan))
            vis_by_channel.append(v[0])
    
        assert len(vis_by_channel) == 12
        for v in vis_by_channel:
            assert v.vis.data.shape[-1] == 4
            assert v.polarisation_frame.type == "linear"

    def test_create_list_spectral_average(self):
        if not self.casacore_available:
            return
    
        msfile = arl_path("data/vis/ASKAP_example.ms")

        vis_by_channel = list()
        nchan_ave = 16
        nchan = 192
        for schan in range(0, nchan, nchan_ave):
            max_chan = min(nchan, schan+nchan_ave)
            v = create_blockvisibility_from_ms(msfile, range(schan, max_chan))
            vis_by_channel.append(integrate_visibility_by_channel(v[0]))
        
        assert len(vis_by_channel) == 12
        for v in vis_by_channel:
            assert v.vis.data.shape[-1] == 4
            assert v.vis.data.shape[-2] == 1
            assert v.polarisation_frame.type == "linear"


    def test_create_list_spectral_average_arlexecute(self):
        if not self.casacore_available:
            return
    
        msfile = arl_path("data/vis/ASKAP_example.ms")
        
        from workflows.arlexecute.processing_component_interface.execution_support.arlexecute import arlexecute
        arlexecute.set_client(use_dask=False)
    
        nchan_ave = 16
        nchan = 192
        
        def create_and_average(schan):
            max_chan = min(nchan, schan + nchan_ave)
            bv = create_blockvisibility_from_ms(msfile, range(schan, max_chan))
            return integrate_visibility_by_channel(bv[0])
                
        vis_by_channel_workflow = \
            [arlexecute.execute(create_and_average)(schan) for schan in range(0, nchan, nchan_ave)]
        
        vis_by_channel = arlexecute.compute(vis_by_channel_workflow)
        arlexecute.close()
    
        assert len(vis_by_channel) == 12
        for v in vis_by_channel:
            assert v.vis.data.shape[-1] == 4
            assert v.polarisation_frame.type == "linear"
            assert v.vis.data.shape[-2] == 1

if __name__ == '__main__':
    unittest.main()
