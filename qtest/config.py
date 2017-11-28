from pyrocko.guts import Object, String, Float, Dict, Tuple, List, Int, Bool
from pyrocko import model, util


class SyntheticTestConfig(Object):
    store_id = String.T()
    store_superdirs = List.T(String.T())
    noise_files = String.T(optional=True)


class QConfig(Object):
    want_phase = String.T()
    earthmodel = String.T()
    markers = String.T()
    events = String.T()
    traces = String.T()
    stations = String.T()
    outdir = String.T()
    channel = String.T()
    window_length = Float.T()
    tstart = String.T(optional=True)
    tstop = String.T(optional=True)
    type = String.T(default='real', optional=True)
    position = Float.T(default=0.9)
    mag_delta_max = Float.T(default=0.5)
    method = String.T(default='mtspec')
    fmax_lim = Float.T(default=85.)
    fmin_lim = Float.T(default=40.)
    fminrange = Float.T(default=30.)
    snr = Float.T(default=5.)
    cc_min = Float.T(default=None, optional=True)
    mag_min = Float.T(default=1.)
    mag_max =  Float.T(default=3.)
    time_bandwidth = Float.T(default=5.)
    ntapers = Int.T(default=3)
    rsquared_min = Float.T(default=None, optional=True)
    rmse_max = Float.T(default=None, optional=True)
    whitelist = List.T(String.T(), optional=True)
    traversing_distance_min = Float.T(default=1000.)
    traversing_ratio = Float.T(default=2.)
    file_format = String.T(optional=True, default='guess')
    synthetic_config = SyntheticTestConfig.T(optional=True)
    fn_couples = String.T(optional=True, default='/tmp/couples')
    plot = Bool.T(default=False)
    noise_window_shift = Float.T(default=0.,
        help='if zero, noise measure taken from window preceding phase window.'
                                 '[seconds]')

    def __init__(self, *args, **kwargs):
        Object.__init__(self, *args, **kwargs)

    @property
    def filtered_events(self):
        e = model.load_events(self.events)
        e = filter(lambda x: self.max_magnitude>x.magnitude>self.min_magnitude, e)
        return e

    @property
    def filtered_stations(self):
        stations = model.load_stations(self.stations)
        f = []
        wl = [tuple(w[:3]) for w in self.whitelist]
        for s in stations:
            if s.nsl() in wl:
                f.append(s)
        return f

    @classmethod
    def example(cls):
        c = cls(
            want_phase='P',
            earthmodel='/data/models/earthmodel_malek_alexandrakis.nd',
            markers='/home/marius/josef_dd/hypodd_markers_josef.pf',
            events='/home/marius/josef_dd/events_from_sebastian_check_M1.pf',
            traces='/data/webnet/gse2/2008Oct',
            stations='/data/meta/stations.pf',
            outdir='qopher.out',
            channel='SHZ')

        return c
