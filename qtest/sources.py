import numpy as num
import math

from pyrocko import moment_tensor
from pyrocko.gf import RectangularSource, DCSource, meta
from pyrocko.guts import Float, String, Int

from brune import Brune
import matplotlib as mpl

d2r = math.pi/180.


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

def discretize_extended_source(deltas, deltat, strike, dip, velocity,
                               length=0.0, width=0.0, mask=None, taper=None,
                               stf=None, nucleation_x=None, nucleation_y=None,
                               tref=0.0):

    if stf is None:
        stf = STF()

    mindeltagf = num.min(deltas)
    mindeltagf = min(mindeltagf, deltat * velocity)

    l = length
    w = width

    nl = 2 * num.ceil(l / mindeltagf) + 1
    nw = 2 * num.ceil(w / mindeltagf) + 1
    #ntau = 2 * num.ceil(tau / deltat) + 1

    n = int(nl*nw)

    dl = l / nl
    dw = w / nw
    #dtau = tau / ntau

    xl = num.linspace(-0.5*(l-dl), 0.5*(l-dl), nl)
    xw = num.linspace(-0.5*(w-dw), 0.5*(w-dw), nw)
    #xtau = num.linspace(-0.5*(tau-dtau), 0.5*(tau-dtau), ntau)

    points = num.empty((n, 3), dtype=num.float)
    points[:, 0] = num.tile(xl, nw)
    points[:, 1] = num.repeat(xw, nl)
    points[:, 2] = 0.0

    if mask is not None:
        points = points[mask(points)]
        n = len(points)

    if nucleation_x is not None:
        dist_x = num.abs(nucleation_x - points[:, 0])
    else:
        dist_x = num.zeros(n)

    if nucleation_y is not None:
        dist_y = num.abs(nucleation_y - points[:, 1])
    else:
        dist_y = num.zeros(n)

    dist = num.sqrt(dist_x**2 + dist_y**2)
    times = dist / velocity
    if taper:
        fac_taper = taper(points)

    #times -= num.mean(times)

    rotmat = num.asarray(
        moment_tensor.euler_to_matrix(dip*d2r, strike*d2r, 0.0))

    points = num.dot(rotmat.T, points.T).T

    xtau, amplitudes = stf.discretize_t(deltat, tref)
    nt = xtau.size
    points2 = num.repeat(points, nt, axis=0)
    times2 = num.repeat(times, nt) + num.tile(xtau, n)
    amplitudes2 = num.tile(amplitudes, n)
    if taper:
        amplitudes2 *= fac_taper

    #print num.sum(amplitudes2)
    amplitudes2 /= num.sum(amplitudes2)
    #print num.sum(amplitudes2)
    #print num.cumsum(amplitudes2)
    return points2, times2, amplitudes2

    #return points2, times2




class CircularSource(DCSource):
    '''
    Coin shaped Haskell source model modified for bilateral rupture.

TODO test moment!

    '''

    discretized_source_class = meta.DiscretizedMTSource

    radius = Float.T(
        default=0.,
        help='radius of circular source area [m]')

    nucleation_x = Float.T(
        optional=True,
        help='horizontal position of rupture nucleation in normalized fault '
             'plane coordinates (-1 = left edge, +1 = right edge)')

    nucleation_y = Float.T(
        optional=True,
        help='down-dip position of rupture nucleation in normalized fault '
             'plane coordinates (-1 = upper edge, +1 = lower edge)')

    velocity = Float.T(
        default=3500.,
        help='speed of rupture front [m/s]')

    def base_key(self):
        return DCSource.base_key(self) + (
            self.radius,
            self.nucleation_x,
            self.nucleation_y,
            self.velocity)

    def mask(self, points):
        return num.where(num.linalg.norm(points, axis=1)<self.radius)

    def taper(self, points):
        amplitudes = num.ones(points.shape[0])
        dist = num.linalg.norm(points, axis=1)
        dist_thresh = self.radius*0.7
        iindx = num.where(dist>dist_thresh)
        m = -1./(self.radius-dist_thresh)
        amplitudes[iindx] = amplitudes[iindx] + (m * (dist[iindx]-dist_thresh))
        return amplitudes

    def discretize_basesource(self, store, target=None):

        if self.nucleation_x is not None:
            nucx = self.nucleation_x * 0.5 * self.radius
        else:
            nucx = None

        if self.nucleation_y is not None:
            nucy = self.nucleation_y * 0.5 * self.radius
        else:
            nucy = None

        stf = self.effective_stf_pre()

        points, times, amplitudes = discretize_extended_source(
            store.config.deltas, store.config.deltat, self.strike, self.dip,
            width=self.radius*2, length=self.radius*2,
            velocity=self.velocity, mask=self.mask, taper=self.taper, stf=stf,
            nucleation_x=nucx, nucleation_y=nucy)

        n = times.size

        mot = moment_tensor.MomentTensor(strike=self.strike, dip=self.dip, rake=self.rake,
                              scalar_moment=1.0/num.sum(amplitudes))

        m6s = num.repeat(mot.m6()[num.newaxis, :], n, axis=0)
        m6s[:, :] *= amplitudes[:, num.newaxis]

        ds = meta.DiscretizedMTSource(
            lat=self.lat,
            lon=self.lon,
            times=times,
            north_shifts=self.north_shift + points[:, 0],
            east_shifts=self.east_shift + points[:, 1],
            depths=self.depth + points[:, 2],
            m6s=m6s)

        return ds

    def plot(self, store):
        if self.nucleation_x is not None:
            nucx = self.nucleation_x * 0.5 * self.radius
        else:
            nucx = None

        if self.nucleation_y is not None:
            nucy = self.nucleation_y * 0.5 * self.radius
        else:
            nucy = None

        stf = self.effective_stf_pre()

        points, times, amplitudes = discretize_extended_source(
            store.config.deltas, store.config.deltat, self.strike, self.dip,
            width=self.radius*2, length=self.radius*2,
            velocity=self.velocity, mask=self.mask, taper=self.taper, stf=stf,
            nucleation_x=nucx, nucleation_y=nucy)


        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        mappable = ax.scatter(points.T[0], points.T[1], points.T[2],
                   s=50*amplitudes/amplitudes.max(),
                   c=times, marker='o',
                              vmin=times.min(), vmax=times.max(),
                              cmap=mpl.cm.YlGnBu)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        plt.colorbar(mappable)
        plt.show()




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


__all__ = '''
DCSourceWid
'''.split()
