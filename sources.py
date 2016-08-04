import numpy as num
from pyrocko import moment_tensor
from pyrocko.gf import RectangularSource, DCSource, meta
from pyrocko.guts import Float, String, Int
from brune import Brune


class RectangularBrunesSource(RectangularSource):
    brunes = Brune.T(optional=True, default=None)
    def __init__(self, **kwargs):
        RectangularSource.__init__(self, **kwargs)
        #self.stf = get_stf(self.magnitude)
        #print 'check if STF was applied!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

#class CircularBrunesSource(CircularSource):
#    brunes = Brune.T(optional=True, default=None)
#    def __init__(self, **kwargs):
#        CircularSource.__init__(self, **kwargs)
#        #self.stf = get_stf(self.magnitude)
#        #print 'check if STF was applied!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'


class DCSourceWid(DCSource):
    id = String.T(optional=True, default=None)
    brunes = Brune.T(optional=True, default=None)
    def __init__(self, **kwargs):
        DCSource.__init__(self, **kwargs)
        #self.stf = get_stf(self.magnitude)
        #print 'check if STF was applied!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'


class LineSource(DCSource):
    discretized_source_class = meta.DiscretizedMTSource
    i_sources = Int.T()
    velocity = Float.T()
    brunes = Brune.T(optional=True, default=None)

    def base_key(self):
        return DCSource.base_key(self)+(self.i_sources, self.velocity)

    def discretize_basesource(self, store):

        c = store.config
        dz = c.source_depth_delta

        #i_sources sollte eher i_sources/2. sein!
        north_shifts = num.arange(-dz*self.i_sources, dz*self.i_sources, dz)
        print 'line_source_points ', north_shifts

        times = num.ones(len(north_shifts)) * (north_shifts[-1]-north_shifts[0])/self.velocity
        n = times.size

        mot = moment_tensor.MomentTensor(strike=self.strike, dip=self.dip, rake=self.rake,
                                                            scalar_moment=1.0/n)
        amplitudes = num.ones(n)/n
        m6s = num.repeat(mot.m6()[num.newaxis, :], n, axis=0)
        m6s[:, :] *= amplitudes[:, num.newaxis]
        ds = meta.DiscretizedMTSource(
            lat=self.lat,
            lon=self.lon,
            times=times,
            north_shifts=self.north_shift + north_shifts,
            east_shifts=self.east_shift,
            depths=self.depth,
            m6s=m6s)
        return ds


