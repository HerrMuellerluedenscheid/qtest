import numpy as num
from pyrocko import moment_tensor
from pyrocko.gf import Target
from matplotlib import pyplot as plt
from brune import Brune
from rupture_size import radius as source_radius
from sources import DCSourceWid, RectangularBrunesSource
try:
    from pyrocko.gf import BoxcarSTF, TriangularSTF, HalfSinusoidSTF
except ImportError as e:
    print 'CHANGE BRANCHES'
    raise e


def M02tr(Mo, stress, vr):
    #stress=stress drop in MPa
    #vr=rupture velocity in m/s
    #Mo = seismic moment calculated with Hanks and Kanamori,1979    
    #Mo=(10.**((3./2.)*(Mw+10.73))/1E+7 #Mo in Nm
    #Calculate rupture length or source radio (m) with Madariaga(1976), stress drop on a circular fault
    Lr = ((7.*Mo)/(16.*stress*1E+6))**(1./3.) #stress Mpa to Nm2
    #Calculate ruputure time in seconds with the rupture velocity
    tr = Lr/vr
    return tr


def radius2fc(r, k=0.32, beta=3500):
    # Source parameters of the swarm earthquakes in West Bohemia/Vogtland,
    # Michalek
    # k = 0.32 (Madariaga, also in Michaleks paper)
    # beta = 3500 (Michaleks paper)
    fc = k*beta/r
    return fc

def fmin_by_magnitude(magnitude, stress=10., vr=3500):
    Mo = moment_tensor.magnitude_to_moment(magnitude)
    #duration = M02tr(Mo, stress, vr)
    #print duration
    #return 1./duration
    # 
    # Source parameters of the swarm earthquakes in West Bohemia/Vogtland,
    # Michalek:
    r = 0.155 * Mo** 0.206
    return radius2fc(r)
    #print Mo, r
    #return r


def e2extendeds(e, north_shift=0., east_shift=0., nucleation_radius=None, stf_type=None):
    if e.moment_tensor:
        mt = e.moment_tensor
        mag = mt.magnitude
        s, d, r = mt.both_strike_dip_rake()[0]
    else:
        mt = False
        mag = e.magnitude
        s, d, r = None, None, None

    a = source_radius([mag])
    if nucleation_radius is not None:
        nucleation_x, nucleation_y = (num.random.random(2)-0.5)*2.*nucleation_radius
        nucleation_x = float(nucleation_x)
        nucleation_y = float(nucleation_y)
    else:
        nucleation_x, nucleation_y = None, None
    velocity = 3500.
    if stf_type=='brunes':
        # mu nachschauen!
        # beta aus Modell
        brunes = Brune(sigma=2.9E6, mu=3E10, beta=velocity)
    else:
        brunes = None

    stf = get_stf(mag, type=stf_type)
    return RectangularBrunesSource(
       lat=float(e.lat), lon=float(e.lon), depth=float(e.depth), north_shift=float(north_shift),
       east_shift=float(east_shift), time=float(e.time), width=float(a[0]), length=float(a[0]),
       strike=float(s), dip=float(d), rake=float(r), magnitude=float(mag), brunes=brunes,
        velocity=float(velocity),
       nucleation_x=float(nucleation_x), nucleation_y=float(nucleation_y), stf=stf)
    #return RectangularSource(
    #   lat=e.lat, lon=e.lon, depth=e.depth, north_shift=north_shift,
    #   east_shift=east_shift, time=e.time, width=float(a[0]), length=float(a[0]),
    #   strike=mt.strike1, dip=mt.dip1, rake=mt.rake1, magnitude=mag,
    #   nucleation_x=nucleation_x, nucleation_y=nucleation_y, stf=stf)


def e2circulars(e, north_shift=0., east_shift=0., nucleation_radius=None, stf_type=None):
    if e.moment_tensor:
        mt = e.moment_tensor
        mag = mt.magnitude
    else:
        mt = False
        mag = e.magnitude
    a = source_radius([mag])
    #d = num.sqrt(a[0])
    print 'magnitude: ', mag
    print 'source radius: ', a
    if nucleation_radius is not None:
        nucleation_x, nucleation_y = (num.random.random(2)-0.5)*2.*nucleation_radius
        nucleation_x = float(nucleation_x)
        nucleation_y = float(nucleation_y)
    else:
        nucleation_x, nucleation_y = None, None
    #nucleation_x = 0.95
    #nucleation_y = 0.
    stf = get_stf(mag, type=stf_type)
    print nucleation_x, nucleation_y
    print mt.strike1, mt.strike2
    print mt.dip1, mt.dip2
    print mt.rake1, mt.rake2
    print '.'*80
    return CircularBrunesSource(
       lat=e.lat, lon=e.lon, depth=e.depth, north_shift=north_shift,
       east_shift=east_shift, time=e.time, radius=float(a[0]),
       strike=mt.strike1, dip=mt.dip1, rake=mt.rake1, magnitude=mag,
       nucleation_x=nucleation_x, nucleation_y=nucleation_y, stf=stf)


def e2s(e, north_shift=0., east_shift=0., stf_type=None):
    if e.moment_tensor:
        mt = e.moment_tensor
        mag = mt.magnitude
    else:
        mt = False
        mag = e.magnitude

    stf = get_stf(mag, type=stf_type)
    if stf_type=='brunes':
        # mu nachschauen!
        # beta aus Modell
        velocity = 3500.
        brunes = Brune(sigma=2.9E6, mu=3E10, beta=velocity)
    else:
        brunes = None
    s = DCSourceWid.from_pyrocko_event(e)
    s.brunes = brunes
    s.north_shift = north_shift
    s.east_shift = east_shift
    s.magnitude = mag
    s.stf = stf
    return s


def e2linesource(e, north_shift=0., east_shift=0., nucleation_radius=None, stf_type=None):
    if e.moment_tensor:
        mt = e.moment_tensor
        mag = mt.magnitude
        s, d, r = mt.both_strike_dip_rake()[0]
    else:
        mt = False
        mag = e.magnitude
        s, d, r = None, None, None

    #a = source_radius([mag])                                                                                                      
    #d = num.sqrt(a[0])
    velocity = 3500.
    if stf_type=='brunes':
        brunes = Brune(sigma=2.9E6, mu=3E10, beta=velocity)
    else:
        brunes = None

    stf = get_stf(mag, type=stf_type)
    return LineSource(
       lat=float(e.lat), lon=float(e.lon), depth=float(e.depth), north_shift=float(north_shift),
       east_shift=float(east_shift), time=float(e.time), i_sources=10,
       strike=float(s), dip=float(d), rake=float(r), magnitude=float(mag), brunes=brunes,
        velocity=velocity, stf=stf)



def s2t(s, channel='Z', store_id=None): 
    return Target(lat=s.lat, lon=s.lon, depth=s.depth, elevation=s.elevation, 
                  codes=(s.network, s.station, s.location, channel), store_id=store_id) 



def get_stf(magnitude=0, stress=0.1, vr=2750., type=None):
    Mo = moment_tensor.magnitude_to_moment(magnitude)
    duration = M02tr(Mo, stress, vr)
    print 'CHECK THE DURATION: %s' % duration
    if type==None:
        stf = None
    elif type=='boxcar':
        stf = BoxcarSTF(duration=duration)
    elif type=='triangular':
        stf = TriangularSTF(duration=duration)
    elif type=='halfsin':
        stf = HalfSinusoidSTF(duration=duration)
    elif type=='brunes':
        stf = None
    else:
        raise Exception('unknown STF type: %s' % type)
    return stf


class Magnitude2fmin():
    def __init__(self, stress, vr, lim, fcoffset=5):
        self.fcoffset = fcoffset
        self.stress = stress
        self.vr = vr
        self.lim = lim

    def __call__(self, magnitude):
        return max(fmin_by_magnitude(magnitude, self.stress, self.vr),
                   self.lim) + self.fcoffset

    @classmethod
    def setup(cls, stress=0.1, vr=2750, lim=0.):
        return cls(stress, vr, lim)

    def plot(self):
        mags = num.linspace(-1, 4, 50)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(mags, self(mags))
        ax.set_xlabel('magnitude')
        ax.set_ylabel('fmin')


class Magnitude2Window():
    def __init__(self, t_static, t_factor):
        self.t_static = t_static
        self.t_factor = t_factor

    def __call__(self, magnitude):
        return self.t_static+self.t_factor/fmin_by_magnitude(magnitude)

    @classmethod
    def setup(cls, t_static=0.1, t_factor=5.):
        return cls(t_static, t_factor)

    def plot(self):
        mags = num.linspace(-1, 4, 50)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(mags, self(mags))
        ax.set_xlabel('magnitude')
        ax.set_ylabel('time')


