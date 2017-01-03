from pyrocko.guts import Object, String, Float, Dict, Tuple, List


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
    whitelist = List.T(Tuple.T(4, String.T()), optional=True)
    traversing_distance_min = Float.T(default=1000.)

    synthetic_config = SyntheticTestConfig.T(optional=True)

