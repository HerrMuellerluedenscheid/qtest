from pyrocko.guts import Object, String, Float, Dict, Tuple, List
from pyrocko import model, util


class SyntheticTestConfig(Object):
    store_id = String.T()
    store_superdirs = List.T(String.T())
    noise_files = String.T(optional=True)


class QConfig(Object):
    type = String.T(default='real', optional=True)
    want_phase = String.T()
    earthmodel = String.T()
    markers = String.T()
    events = String.T()
    traces = String.T()
    stations = String.T()
    output = String.T()
    magdiffmax = Float.T(default=0.5)
    method = String.T(default='mtspec')
    fmax_lim = Float.T(default=85.)
    fmin_lim = Float.T(default=40.)
    fminrange = Float.T(default=30.)
    snr_min = Float.T(default=1800)
    cc_min = Float.T(default=0.75)
    min_magnitude = Float.T(default=1.)
    max_magnitude =  Float.T(default=4.)
    whitelist = List.T(String.T(), optional=True)
    traversing_distance_min = Float.T(default=1000.)
    traversing_ratio = Float.T(default=2.)
    file_format = String.T(optional=True)
    synthetic_config = SyntheticTestConfig.T(optional=True)
    window_length = Float.T(optional=True)
    channel = String.T()
    fn_couples = String.T(optional=True)

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
