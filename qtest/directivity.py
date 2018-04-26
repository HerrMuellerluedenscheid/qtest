import numpy as num
from scipy import stats


from qtest.util import e2s, s2t, reset_events, get_spectrum
from pyrocko import model
from pyrocko.gui.marker import PhaseMarker
from pyrocko.pile import make_pile
try:
    # python2
    import cPickle as pickle
except ImportError:
    # python3
    import pickle

def run_directivity(config, snuffle=False):
    events_load = model.load_events(config.events)
    markers_load = PhaseMarker.load_markers(config.markers)
    reset_events(markers_load, events_load)
    markers_by_event = {}
    p = config.get_pile()

    tstop = 0.2
    tstart = -0.2

    want_stations = ['SKC', 'LBC']

    for m in markers_load:
        if not p.is_relevant(m.tmin, m.tmax):
            continue
        if not m.one_nslc()[1] in want_stations:
            continue
        e = m.get_event()
        if e.magnitude < config.mag_min:
            continue
        mlist = markers_by_event.get(e, [])
        mlist.append(m)
        markers_by_event[e] = mlist

    slope_ratios_by_events = {}
    keys = markers_by_event.keys()
    keys.sort(key=lambda x: x.time)
    for imarker, e in enumerate(keys):
        markers = markers_by_event[e]
        # for imarker, (e, markers) in enumerate(markers_by_event.items()):
        print(imarker+1, len(markers_by_event))

        slopes = {}
        for m in markers:
            trs = p.chopper(tmin=m.tmin+tstart,
                tmax=m.tmax+tstop,
                trace_selector=lambda x: m.match_nslc(x.nslc_id))

            tr = [tr for _tr in trs for tr in _tr if tr.channel == config.channel]
            if len(tr) == 0:
                continue
            if len(tr) > 1:
                raise Exception('more than one trace matched')
            tr = tr[0]
            f, a = get_spectrum(tr.ydata, tr.deltat, config)
            idx = config.get_valid_frequency_idx(f)

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                num.log(f[idx]), num.log(a[idx]))

            slopes[tr.station] = slope

        try:
            slope_ratios_by_events[e.name] = slopes[want_stations[1]] / slopes[want_stations[0]]
        except KeyError as e:
            continue


    with open('differential_fall_offs.pickl', 'wb') as f:
        pickle.dump(slope_ratios_by_events, f, protocol=2)
    print(slope_ratios_by_events)


if __name__ == '__main__':
    from qtest import config
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--snuffle', action='store_true')
    args = parser.parse_args()
    c = config.QConfig.load(filename=args.config)
    run_directivity(c, args.snuffle)
